import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers import (AdamW,
                                  BertConfig, BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaModel, RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  WarmupLinearSchedule)
from pytorch_transformers.modeling_bert import BertEncoder
from pytorch_transformers.modeling_roberta import gelu
from pytorch_transformers.my_modeling_roberta import RobertaForTACRED


logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())


MODEL_CLASSES = {'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                 'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
                 'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
                 'roberta': (RobertaConfig, RobertaForTACRED, RobertaTokenizer)}


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
