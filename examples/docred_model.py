import logging
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss


logger = logging.getLogger(__name__)


class DocREDModel(nn.Module):
    def __init__(self, args, pretrained_model_config, fac_adapter, et_adapter, lin_adapter):
        super(DocREDModel, self).__init__()
        self.args = args
        self.config = pretrained_model_config

        self.loss_function = BCEWithLogitsLoss()

        self.block_size = 64
        self.num_labels = 97

        self.fac_adapter = fac_adapter
        self.et_adapter = et_adapter
        self.lin_adapter = lin_adapter

        self.head_extractor = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.tail_extractor = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier = nn.Linear(self.config.hidden_size * self.block_size, self.num_labels)

        if args.freeze_adapter and (self.fac_adapter is not None):
            for p in self.fac_adapter.parameters():
                p.requires_grad = False

        if args.freeze_adapter and (self.et_adapter is not None):
            for p in self.et_adapter.parameters():
                p.requires_grad = False

        if args.freeze_adapter and (self.lin_adapter is not None):
            for p in self.lin_adapter.parameters():
                p.requires_grad = False

        self.adapter_num = 0

        if self.fac_adapter is not None:
            self.adapter_num += 1

        if self.et_adapter is not None:
            self.adapter_num += 1

        if self.lin_adapter is not None:
            self.adapter_num += 1

        if self.args.fusion_mode == 'concat':
            self.task_dense_lin = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense_fac = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)

        self.dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.config.hidden_size, self.num_labels)

    def get_head_tail_representations(self, task_features, entity_position_ids, head_tail_idxs):
        """
        subj_special_start_id.shape = [batch_size, max_seq_length]
        """
        all_subj_outputs = []
        all_obj_outputs = []

        for batch_idx, _ in enumerate(head_tail_idxs):
            subj_outputs = []
            obj_outputs = []

            task_feature = task_features[batch_idx]
            seq_length = task_feature.shape[0]

            head_tail_pairs = head_tail_idxs[batch_idx]
            entities = entity_position_ids[batch_idx]

            for pair in head_tail_pairs:
                subj_special_start_id = torch.zeros(seq_length).to(task_feature)
                obj_special_start_id = torch.zeros(seq_length).to(task_feature)

                subj_idx = pair[0]
                obj_idx = pair[1]

                subj_positions = entities[subj_idx]
                obj_positions = entities[obj_idx]

                subj_start_idxs = [x[0] for x in subj_positions]
                obj_start_idxs = [x[0] for x in obj_positions]

                subj_special_start_id[subj_start_idxs] = 1
                obj_special_start_id[obj_start_idxs] = 1

                subj_output = torch.matmul(subj_special_start_id.unsqueeze(0), task_feature)
                obj_output = torch.matmul(obj_special_start_id.unsqueeze(0), task_feature)

                subj_outputs.append(subj_output)
                obj_outputs.append(obj_output)

            all_subj_outputs.append(subj_outputs)
            all_obj_outputs.append(obj_outputs)

        return all_subj_outputs, all_obj_outputs

    def get_labels(self, logits, k):
        threshold_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        logit_mask = (logits > threshold_logit)

        if k > 0:
            top_k, _ = torch.topk(input=logits, k=k, dim=1)
            top_k = top_k[:, -1]
            logit_mask = (logits >= top_k.unsqueeze(1)) & logit_mask

        output[logit_mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.0).to(logits)

        return output

    def forward(self, pretrained_model_outputs, input_ids, attention_mask=None, entity_position_ids=None, head_tail_idxs=None, labels=None):
        pretrained_model_last_hidden_states = pretrained_model_outputs[0]

        if self.fac_adapter is not None:
            fac_adapter_outputs, _, _ = self.fac_adapter(pretrained_model_outputs)

        if self.et_adapter is not None:
            et_adapter_outputs, _, _ = self.et_adapter(pretrained_model_outputs)

        if self.lin_adapter is not None:
            lin_adapter_outputs, _, _ = self.lin_adapter(pretrained_model_outputs)

        if self.args.fusion_mode == 'add':
            task_features = pretrained_model_last_hidden_states

            if self.fac_adapter is not None:
                task_features = task_features + fac_adapter_outputs

            if self.et_adapter is not None:
                task_features = task_features + et_adapter_outputs

            if self.lin_adapter is not None:
                task_features = task_features + lin_adapter_outputs
        elif self.args.fusion_mode == 'concat':
            combine_features = pretrained_model_last_hidden_states
            fac_features = self.task_dense_fac(torch.cat([combine_features, fac_adapter_outputs], dim=2))
            lin_features = self.task_dense_lin(torch.cat([combine_features, lin_adapter_outputs], dim=2))
            task_features = self.task_dense(torch.cat([fac_features, lin_features], dim=2))

        # Need to clean up padding values that were used for tensor conversion. ###################
        cleaned_head_tail_idxs = []
        cleaned_labels = []
        for batch_idx, head_tail_pairs in enumerate(head_tail_idxs):
            head_tail_idxs_temp = []
            labels_temp = []
            for idx, pair in enumerate(head_tail_pairs):
                if pair.detach().cpu().tolist() != [-1, -1]:
                    head_tail_idxs_temp.append(head_tail_pairs[idx].detach().cpu().tolist())
                    labels_temp.append(labels[batch_idx][idx].detach().cpu().tolist())

            cleaned_head_tail_idxs.append(head_tail_idxs_temp)
            cleaned_labels.append(labels_temp)

        cleaned_entity_positions = []
        for batch_idx, entities in enumerate(entity_position_ids):
            entities_temp = []
            for entity in entities:
                mentions_temp = []
                for idx, mention in enumerate(entity):
                    if mention.detach().cpu().tolist() != [-1, -1]:
                        mentions_temp.append(mention.detach().cpu().tolist())

                if mentions_temp != []:
                    entities_temp.append(mentions_temp)
            if entities_temp != []:
                cleaned_entity_positions.append(entities_temp)

        head_tail_idxs = cleaned_head_tail_idxs
        labels = cleaned_labels
        entity_position_ids = cleaned_entity_positions
        ###########################################################################################

        all_subj_outputs, all_obj_outputs = self.get_head_tail_representations(task_features, entity_position_ids, head_tail_idxs)

        subj_outputs = torch.cat(sum(all_subj_outputs, []), dim=0)
        obj_outputs = torch.cat(sum(all_obj_outputs, []), dim=0)
        relation_representations = torch.cat([subj_outputs, obj_outputs], dim=1)

        z_s = torch.tanh(self.head_extractor(subj_outputs))
        z_o = torch.tanh(self.tail_extractor(obj_outputs))

        b1 = z_s.view(-1, self.config.hidden_size // self.block_size, self.block_size)
        b2 = z_o.view(-1, self.config.hidden_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.config.hidden_size * self.block_size)

        logits = self.classifier(bl)
        outputs = (self.get_labels(logits, k=4),)

        if labels is not None:
            labels = torch.tensor(sum(labels, [])).to(logits)
            loss = (self.loss_function(logits, labels),)
            outputs = loss + outputs

        return outputs

    def save_pretrained(self, save_directory):
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

        logger.info("Saving model checkpoint to %s", save_directory)
