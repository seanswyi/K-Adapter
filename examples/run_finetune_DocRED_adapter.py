# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" k-adapter for docred"""

from __future__ import absolute_import, division, print_function

import argparse
from collections import Counter
import logging
import os
import random
import sys
import time

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from create_data import load_and_cache_examples
from docred_evaluation import convert_to_official_format, docred_official_evaluate
from docred_model import DocREDModel
from pytorch_transformers import (AdamW,
                                  BertConfig, BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaModel, RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  WarmupLinearSchedule)
from pytorch_transformers.modeling_bert import BertEncoder
from pytorch_transformers.modeling_roberta import gelu
from pytorch_transformers.my_modeling_roberta import RobertaForTACRED
from utils_glue import compute_metrics, ENTITY_MARKER, output_modes, processors


logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())


MODEL_CLASSES = {'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
                 'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
                 'roberta': (RobertaConfig, RobertaForTACRED, RobertaTokenizer)}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataloader, model, tokenizer):
    """ Train the model """
    pretrained_model = model[0]
    docred_model = model[1]

    if args.max_steps > 0:
        t_total = args.max_steps
        # args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if args.freeze_bert:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in docred_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in docred_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in docred_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in docred_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        if args.freeze_bert:
            docred_model = torch.nn.DataParallel(docred_model)
        else:
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            docred_model = torch.nn.DataParallel(docred_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        if args.freeze_bert:
            docred_model = torch.nn.parallel.DistributedDataParallel(docred_model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        else:
            docred_model = torch.nn.parallel.DistributedDataParallel(docred_model, device_ids=[args.local_rank],
                                                                     output_device=args.local_rank,
                                                                     find_unused_parameters=True)
            pretrained_model = torch.nn.parallel.DistributedDataParallel(pretrained_model, device_ids=[args.local_rank],
                                                                     output_device=args.local_rank,
                                                                     find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    logger.info("Try resume from checkpoint")
    if args.restore:
        if os.path.exists(os.path.join(args.output_dir, 'global_step.bin')):
            logger.info("Load last checkpoint data")
            global_step = torch.load(os.path.join(args.output_dir, 'global_step.bin'))
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            logger.info("Load from output_dir {}".format(output_dir))

            optimizer.load_state_dict(torch.load(os.path.join(output_dir, 'optimizer.bin')))
            scheduler.load_state_dict(torch.load(os.path.join(output_dir, 'scheduler.bin')))
            # args = torch.load(os.path.join(output_dir, 'training_args.bin'))
            if hasattr(pretrained_model,'module'):
                pretrained_model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_pretrained_model.bin')))
            else: # Take care of distributed/parallel training
                pretrained_model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_pretrained_model.bin')))
            if hasattr(docred_model,'module'):
                docred_model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
            else:
                docred_model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))

            global_step += 1
            start_epoch = int(global_step / len(train_dataloader))
            start_step = global_step-start_epoch*len(train_dataloader)-1

            logger.info("Start from global_step={} epoch={} step={}".format(global_step, start_epoch, start_step))
        else:
            global_step = 0
            start_epoch = 0
            start_step = 0

            logger.info("Start from scratch")
    else:
        global_step = 0
        start_epoch = 0
        start_step = 0

        logger.info("Start from scratch")
    # global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    pretrained_model.zero_grad()
    docred_model.zero_grad()

    # train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(start_epoch, int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            start = time.time()

            inputs = {'input_ids': batch['word_ids'], # [batch_size, seq_length]
                      'attention_mask': batch['word_attention_mask'], # [batch_size, seq_length]
                      'entity_position_ids': batch['entity_position_ids'], # [batch_size, num_entities, num_mentions, start_end]
                      'labels': batch['labels'], # [batch_size, num_head_tail_pairs, num_labels]
                      'head_tail_idxs': batch['head_tail_idxs']} # [batch_size, num_head_tail_pairs, head_tail]

            inputs = {key: value.to(args.device) for key, value in inputs.items()}

            if args.restore and (step < start_step):
                continue

            if args.freeze_bert:
                pretrained_model.eval()
            else:
                pretrained_model.train()

            docred_model.train()

            pretrained_model_outputs = pretrained_model(**inputs)
            outputs = docred_model(pretrained_model_outputs, **inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # epoch_iterator.set_description("loss {}".format(loss))
            logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs} - Iter {step}/{len(train_dataloader)} | loss = {loss.item():.5f} | time used = {time.time() - start:.3f}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(docred_model.parameters(), args.max_grad_norm)

            wandb.log({'loss': loss.item()})

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                # model.zero_grad()
                pretrained_model.zero_grad()
                docred_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # model_to_save = docred_model.module if hasattr(docred_model, 'module') else docred_model  # Take care of distributed/parallel training
                    # model_to_save.save_pretrained(output_dir)
                    # model_to_save = pretrained_model.module if hasattr(pretrained_model, 'module') else pretrained_model  # Take care of distributed/parallel training
                    # model_to_save.save_pretrained(output_dir)

                    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    # logger.info("Saving model checkpoint to %s", output_dir)
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.bin'))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.bin'))
                    # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    # torch.save(global_step, os.path.join(args.output_dir, 'global_step.bin'))

                    # logger.info("Saving model checkpoint, optimizer, global_step to %s", output_dir)

                if global_step % args.eval_steps == 0:  # Only evaluate when single GPU otherwise metrics may not average well
                    model = (pretrained_model, docred_model)
                    results = evaluate(args, model, tokenizer)

                # if args.save_model_iteration:
                #     if (global_step + 1) % args.save_model_iteration == 0:
                #         output_dir = args.output_dir

                #         model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #         model_to_save.save_pretrained(output_dir)

                #         torch.save(args, os.path.join(output_dir, 'pytorch_model_{}.bin'.format(global_step + 1)))

            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break

        # wandb.log({'loss': tr_loss})

        if args.max_steps > 0 and global_step > args.max_steps:
            # train_iterator.close()
            break

        model = (pretrained_model,docred_model)
        normal_scores, ignore_scores = evaluate(args, model, tokenizer, prefix='')

    return global_step, tr_loss / global_step, normal_scores, ignore_scores


save_results=[]
def evaluate(args, model, tokenizer, prefix=''):
    pretrained_model = model[0]
    docred_model = model[1]

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for dataset_type in ['dev']:
        for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
            eval_features, eval_dataloader = load_and_cache_examples(args, eval_task, tokenizer, dataset_type, evaluate=True)

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            # Eval!
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataloader))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            index = 0

            all_predictions = []

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                pretrained_model.eval()
                docred_model.eval()

                index += 1
                with torch.no_grad():
                    inputs = {'input_ids': batch['word_ids'],
                              'attention_mask': batch['word_attention_mask'],
                              'entity_position_ids': batch['entity_position_ids'],
                              'labels': batch['labels'],
                              'head_tail_idxs': batch['head_tail_idxs']}

                    inputs['input_ids'] = inputs['input_ids'].to(args.device)
                    inputs['attention_mask'] = inputs['attention_mask'].to(args.device)

                    pretrained_model_outputs = pretrained_model(**inputs)
                    outputs = docred_model(pretrained_model_outputs,**inputs)

                    tmp_eval_loss, predictions = outputs

                    eval_loss += tmp_eval_loss.mean().item()
                    all_predictions.append(predictions)

                nb_eval_steps += 1

                index += 1

            eval_loss = eval_loss / nb_eval_steps
            wandb.log({'eval_loss': eval_loss})

            all_predictions = torch.cat(all_predictions, dim=0)
            all_submissions = convert_to_official_format(all_predictions, eval_features)

            prediction_counts = np.nonzero(all_predictions.detach().cpu().numpy())[1]
            print(f"Model predictions: {Counter(prediction_counts)}")

            if len(all_submissions) > 0:
                normal_scores, ignore_scores = docred_official_evaluate(all_submissions, args.data_dir)
            else:
                normal_scores = {'precision': 0.0,
                                'recall': 0.0,
                                'f1': 0.0}
                ignore_scores = {'ign_precision': 0.0,
                                'ign_recall': 0.0,
                                'ign_f1': 0.0}

            wandb.log(normal_scores)
            wandb.log(ignore_scores)

            logger.info('{} result:{}'.format(dataset_type, results))

    return normal_scores, ignore_scores


class PretrainedModel(nn.Module):
    def __init__(self, args):
        super(PretrainedModel, self).__init__()
        self.args = args

        self.model = RobertaModel.from_pretrained('roberta-large', output_hidden_states=True, output_attentions=True)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        self.config = self.model.config
        self.config.freeze_adapter = args.freeze_adapter

        self.config.output_attentions = True

        if args.freeze_bert:
            for p in self.parameters():
                p.requires_grad = False

    def encode_sequence(self, input_ids, attention_mask, start_tokens, end_tokens):
        n, c = input_ids.size()

        start_tokens = torch.tensor(start_tokens).to(input_ids)
        end_tokens = torch.tensor(end_tokens).to(input_ids)

        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)

        if c <= 512:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            new_input_ids = []
            new_hidden_states = []
            new_attention_mask = []
            num_seg = []

            seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
            for i, l_i in enumerate(seq_len):
                if l_i <= 512:
                    new_input_ids.append(input_ids[i, :512])
                    new_attention_mask.append(attention_mask[i, :512])

                    num_seg.append(1)
                else:
                    input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                    input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)

                    attention_mask1 = attention_mask[i, :512]
                    attention_mask2 = attention_mask[i, (l_i - 512): l_i]

                    new_input_ids.extend([input_ids1, input_ids2])
                    new_attention_mask.extend([attention_mask1, attention_mask2])

                    num_seg.append(2)

            input_ids = torch.stack(new_input_ids, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            sequence_output = outputs[0]
            pooled_output = outputs[1]
            hidden_states = outputs[2]
            attention = outputs[-1][-1]

            hidden_states = torch.cat([x.unsqueeze(0) for x in outputs[2]], dim=0)

            current_idx = 0
            new_sequence_output = []
            new_attention = []
            new_hidden_states = []
            for (n_s, l_i) in zip(num_seg, seq_len):
                if n_s == 1:
                    output = F.pad(sequence_output[current_idx], (0, 0, 0, c - 512))
                    att = F.pad(attention[current_idx], (0, c - 512, 0, c - 512))
                    hidden_state = F.pad(hidden_states[:, current_idx], (0, 0, 0, c - 512))#.unsqueeze(1)

                    new_sequence_output.append(output)
                    new_attention.append(att)
                    new_hidden_states.append(hidden_state)
                elif n_s == 2:
                    output1 = sequence_output[current_idx][:512 - len_end]
                    mask1 = attention_mask[current_idx][:512 - len_end]
                    att1 = attention[current_idx][:, :512 - len_end, :512 - len_end]
                    hidden_state1 = hidden_states[:, current_idx][:, :512 - len_end]

                    output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                    mask1 = F.pad(mask1, (0, c - 512 + len_end))
                    att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))
                    hidden_state1 = F.pad(hidden_state1, (0, 0, 0, c - 512 + len_end))

                    output2 = sequence_output[current_idx + 1][len_start:]
                    mask2 = attention_mask[current_idx + 1][len_start:]
                    att2 = attention[current_idx + 1][:, len_start:, len_start:]
                    hidden_state2 = hidden_states[:, current_idx + 1][:, len_start:]

                    output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                    mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                    att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                    hidden_state2 = F.pad(hidden_state2, (0, 0, l_i - 512 + len_start, c - l_i))

                    mask = mask1 + mask2 + 1e-10
                    output = (output1 + output2) / mask.unsqueeze(-1)
                    att = att1 + att2
                    att = att / (att.sum(-1, keepdim=True) + 1e-10)
                    hidden_state = ((hidden_state1 + hidden_state2) / mask.unsqueeze(-1))#.unsqueeze(1)

                    new_sequence_output.append(output)
                    new_attention.append(att)
                    new_hidden_states.append(hidden_state)

                current_idx += n_s

            sequence_output = torch.stack(new_sequence_output, dim=0)
            attention = torch.stack(new_attention, dim=0)

            hidden_states = torch.stack(new_hidden_states, dim=1)
            hidden_states = tuple(state for state in hidden_states)

            outputs = (sequence_output, pooled_output, hidden_states, attention)

        return outputs # (encoded_sequence, pooled_output, hidden_states, attention)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, entity_position_ids=None, head_tail_idxs=None, labels=None):
        start_tokens = [self.tokenizer.cls_token_id]
        end_tokens = [self.tokenizer.sep_token_id]

        outputs = self.encode_sequence(input_ids, attention_mask, start_tokens, end_tokens)

        # outputs = self.model(input_ids,
        #                      attention_mask=attention_mask,
        #                      token_type_ids=token_type_ids,
        #                      position_ids=position_ids,
        #                      head_mask=head_mask)

        return outputs

        # return outputs, attention  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_pretrained_model.bin")

        torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Saving model checkpoint to %s", save_directory)


class Adapter(nn.Module):
    def __init__(self, args,adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]

        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers

        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)

        up_projected = self.up_project(encoder_outputs[0])

        return hidden_states + up_projected


class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float= 0.1
            hidden_dropout_prob: float=0.1
            hidden_size: int=768
            initializer_range: float=0.02
            intermediate_size: int=3072
            layer_norm_eps: float=1e-05
            max_position_embeddings: int=514
            num_attention_heads: int=12
            num_hidden_layers: int=self.args.adapter_transformer_layers
            num_labels: int=2
            output_attentions: bool=False
            output_hidden_states: bool=False
            torchscript: bool=False
            type_vocab_size: int=1
            vocab_size: int=50265

        self.adapter_config = AdapterConfig

        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.adapter_list = args.adapter_list
        self.adapter_num = len(self.adapter_list)
        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

    def forward(self, pretrained_model_outputs):
        outputs = pretrained_model_outputs
        sequence_output = outputs[0]
        # pooler_output = outputs[1]
        hidden_states = outputs[2]
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to('cuda')

        adapter_hidden_states = []
        adapter_hidden_states_count = 0

        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1

            if self.adapter_skip_layers >= 1:
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[int(adapter_hidden_states_count/self.adapter_skip_layers)]

        outputs = (hidden_states_last,) + outputs[2:]

        return outputs  # (loss), logits, (hidden_states), (attentions)


def load_pretrained_adapter(adapter, adapter_path):
    new_adapter= adapter
    model_dict = new_adapter.state_dict()
    logger.info('Adapter model weight:')
    logger.info(new_adapter.state_dict().keys())
    logger.info('Load model state dict from {}'.format(adapter_path))
    adapter_meta_dict = torch.load(adapter_path, map_location=lambda storage, loc: storage)
    logger.info('Load pretraiend adapter model state dict ')
    logger.info(adapter_meta_dict.keys())

    for item in ['out_proj.bias', 'out_proj.weight', 'dense.weight', 'dense.bias']:
        if item in adapter_meta_dict:
            adapter_meta_dict.pop(item)

    changed_adapter_meta = {}
    for key in adapter_meta_dict.keys():
        changed_adapter_meta[key.replace('adapter.', 'adapter.')] = adapter_meta_dict[key]

    changed_adapter_meta = {k: v for k, v in changed_adapter_meta.items() if k in model_dict.keys()}
    model_dict.update(changed_adapter_meta)
    new_adapter.load_state_dict(model_dict)
    logger.info('Adapter-meta new model weight key')
    logger.info(new_adapter.state_dict().keys())

    return new_adapter


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--comment", default='', type=str,
                        help="The comment")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--restore", type=bool, default=False, help="Whether restore from the last checkpoint, is nochenckpoints, start from scartch")

    parser.add_argument("--freeze_bert", default=True, type=bool,
                        help="freeze the parameters of pretrained model.")
    parser.add_argument("--freeze_adapter", default=False, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument("--test_mode", default=0, type=int,
                        help="test freeze adapter")

    parser.add_argument('--fusion_mode', type=str, default='concat',help='the fusion mode for bert feautre and adapter feature |add|concat')
    parser.add_argument("--version", default='v1', type=str, help="the ablated model")
    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,22", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=3, type=int,
                        help="The skip_layers of adapter according to bert layers")

    parser.add_argument('--meta_fac_adaptermodel', default='',type=str, help='the pretrained factual adapter model')
    parser.add_argument('--meta_et_adaptermodel', default='',type=str, help='the pretrained entity typing adapter model')
    parser.add_argument('--meta_lin_adaptermodel', default='', type=str, help='the pretrained linguistic adapter model')

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--eval_steps', type=int, default=None,
                        help="eval every X updates steps.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=40, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true', default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--meta_bertmodel', default='', type=str, help='the pretrained bert model')
    parser.add_argument('--save_model_iteration', type=int, help='when to save the model..')
    parser.add_argument('--negative_sample', type=int, help='how many negative samples to select')

    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]

    name_prefix = 'batch-'+str(args.per_gpu_train_batch_size)+'_'+'lr-'+str(args.learning_rate)+'_'+'warmup-'+str(args.warmup_steps)+'_'+'epoch-'+str(args.num_train_epochs)+'_'+str(args.comment)
    args.my_model_name = args.task_name+'_'+name_prefix
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    if args.eval_steps is None:
        args.eval_steps = args.save_steps * 1

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device

    wandb.init(project='K-Adapter', name='DocRED_bilinear')

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    args.task_name = args.task_name.lower()

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(max_seq_length=args.max_seq_length)
    num_labels = len(label_list)
    args.num_labels = num_labels

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    tokenizer.add_special_tokens(dict(additional_special_tokens=[ENTITY_MARKER]))

    pretrained_model = PretrainedModel(args)
    pretrained_model.model.resize_token_embeddings(len(tokenizer))

    if args.meta_fac_adaptermodel:
        fac_adapter = AdapterModel(args, pretrained_model.config)
        fac_adapter = load_pretrained_adapter(fac_adapter,args.meta_fac_adaptermodel)
    else:
        fac_adapter = None

    if args.meta_et_adaptermodel:
        et_adapter = AdapterModel(args, pretrained_model.config)
        et_adapter = load_pretrained_adapter(et_adapter,args.meta_et_adaptermodel)
    else:
        et_adapter = None

    if args.meta_lin_adaptermodel:
        lin_adapter = AdapterModel(args, pretrained_model.config)
        lin_adapter = load_pretrained_adapter(lin_adapter,args.meta_lin_adaptermodel)
    else:
        lin_adapter = None

    docred_model = DocREDModel(args, pretrained_model.config, fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter)

    if args.meta_bertmodel:
        model_dict = pretrained_model.state_dict()
        bert_meta_dict = torch.load(args.meta_bertmodel, map_location=lambda storage, loc: storage)
        for item in ['out_proj.weight', 'out_proj.bias', 'dense.weight', 'dense.bias', 'lm_head.bias', 'lm_head.dense.weight', 'lm_head.dense.bias',
                     'lm_head.layer_norm.weight', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight']:
            if item in bert_meta_dict:
                bert_meta_dict.pop(item)
        changed_bert_meta = {}
        for key in bert_meta_dict.keys():
            changed_bert_meta[key.replace('model.','roberta.')] = bert_meta_dict[key]
        changed_bert_meta = {k: v for k, v in changed_bert_meta.items() if k in model_dict.keys()}
        model_dict.update(changed_bert_meta)
        pretrained_model.load_state_dict(model_dict)

    model = (pretrained_model, docred_model)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    pretrained_model.to(args.device)
    docred_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        _, train_dataloader = load_and_cache_examples(args, args.task_name, tokenizer, 'train', evaluate=False)
        global_step, tr_loss, normal_scores, ignore_scores = train(args, train_dataloader, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(f"Normal scores: {normal_scores}")
        logger.info(f"Ignore scores: {ignore_scores}")

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = docred_model.module if hasattr(docred_model, 'module') else docred_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # # Evaluation
    results = {}

    return results


if __name__ == '__main__':
    main()
