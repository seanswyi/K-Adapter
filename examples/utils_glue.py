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
# distributed under tconvert_examples_to_features_trexhe License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

from collections import Counter
import csv
from io import open
from itertools import permutations
import json
import logging
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm


logger = logging.getLogger(__name__)


ENTITY_MARKER = '[ENTITY]'


with open(file='/hdd1/seokwon/data/DocRED/rel_info.json') as f:
    docred_id2rel = json.load(fp=f)


with open(file='/hdd1/seokwon/data/DocRED/DocRED_baseline_metadata/rel2id.json') as f:
    docred_rel2id = json.load(fp=f)


def adjust_mention_positions(entities, text):
    for entity_idx, entity in enumerate(entities):
        entity = sorted(entity, key=lambda x: x['sent_id'])
        for mention_idx, mention in enumerate(entity):
            sent_id = mention['sent_id']

            # No need to adjust span for first sentence.
            if sent_id == 0:
                continue

            prev_sent_lengths = sum([len(x) for x in text[:sent_id]])
            adjusted_positions = [i + prev_sent_lengths for i in mention['pos']]
            entity[mention_idx]['pos'] = adjusted_positions

    return entities


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DocREDInputExample():
    def __init__(self, id_, text, entity_pos, head_tail_idxs, labels):
        self.id = id_
        self.text = text
        self.entity_pos = entity_pos
        self.head_tail_idxs = head_tail_idxs
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, start_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.start_id = start_id


class DocREDInputFeature():
    def __init__(self, title, word_ids, word_attention_mask, entity_position_ids, labels, head_tail_idxs):
        self.title = title
        self.word_ids = word_ids
        self.word_attention_mask = word_attention_mask
        self.entity_position_ids = entity_position_ids
        self.labels = labels
        self.head_tail_idxs = head_tail_idxs


class tacredInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, subj_special_start_id, obj_special_start_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.subj_special_start_id = subj_special_start_id
        self.obj_special_start_id = obj_special_start_id


class semevalInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, e1_start_id, e2_start_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.e1_start_id = e1_start_id
        self.e2_start_id = e2_start_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)

    @classmethod
    def _read_semeval_txt(clas, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            examples = []
            example = []
            for line in f:
                if line.strip() == '':
                    examples.append(example)
                    example = []
                else:
                    example.append(line.strip())
            return examples


class EntityTypeProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, dataset_type=None):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label_list = ['entity', 'location', 'time', 'organization', 'object', 'event', 'place', 'person', 'group']
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = [0 for item in range(len(label_list))]
            for item in line['labels']:
                label[label_list.index(item)] = 1

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


relations = ['per:siblings', 'per:parents', 'org:member_of', 'per:origin', 'per:alternate_names', 'per:date_of_death',
             'per:title', 'org:alternate_names', 'per:countries_of_residence', 'org:stateorprovince_of_headquarters',
             'per:city_of_death', 'per:schools_attended', 'per:employee_of', 'org:members', 'org:dissolved',
             'per:date_of_birth', 'org:number_of_employees/members', 'org:founded', 'org:founded_by',
             'org:political/religious_affiliation', 'org:website', 'org:top_members/employees', 'per:children',
             'per:cities_of_residence', 'per:cause_of_death', 'org:shareholders', 'per:age', 'per:religion',
             'no_relation',
             'org:parents', 'org:subsidiaries', 'per:country_of_birth', 'per:stateorprovince_of_death',
             'per:city_of_birth',
             'per:stateorprovinces_of_residence', 'org:country_of_headquarters', 'per:other_family',
             'per:stateorprovince_of_birth',
             'per:country_of_death', 'per:charges', 'org:city_of_headquarters', 'per:spouse']


class TACREDProcessor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type, negative_sample):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, negative_sample)

    def get_dev_examples(self, data_dir, dataset_type, negative_sample):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, negative_sample)

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return relations

    def _create_examples(self, lines, dataset_type, negative_sample):
        """Creates examples for the training and dev sets."""
        examples = []
        no_relation_number = negative_sample

        pbar = tqdm(iterable=lines, desc=f"Processing for {dataset_type}", total=len(lines))
        for i, line in enumerate(pbar):
            guid = i
            # text_a: tokenized words
            text_a = line['token']
            # text_b: other information
            text_b = (line['subj_start'], line['subj_end'], line['obj_start'], line['obj_end'])
            label = line['relation']

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

            # if label == 'no_relation' and dataset_type == 'train':
            #     no_relation_number -= 1
            #     if no_relation_number > 0:
            #         examples.append(
            #             InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            #     else:
            #         continue
            # else:
            #     examples.append(
            #         InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class DocREDProcessor(DataProcessor):
    def get_train_examples(self, data_dir, max_seq_length):
        """See base class."""
        train_filename = os.path.join(data_dir, 'train_annotated.json')
        with open(file=train_filename) as f:
            train_data = json.load(fp=f)

        return self._create_examples(train_data, dataset_type='train', max_seq_length=max_seq_length)

    def get_dev_examples(self, data_dir, max_seq_length):
        """See base class."""
        dev_filename = os.path.join(data_dir, 'dev.json')
        with open(file=dev_filename) as f:
            dev_data = json.load(fp=f)

        return self._create_examples(dev_data, dataset_type='dev', max_seq_length=max_seq_length)

    def get_labels(self, data_dir='/hdd1/seokwon/data/DocRED/original', examples=None, max_seq_length=None):
        if examples == None:
            examples = self.get_dev_examples(data_dir, max_seq_length=max_seq_length)

        labels = set()

        for example in examples:
            labels_ = example.labels

            for label in labels_:
                for idx, element in enumerate(label):
                    if element != 0:
                        labels.add(idx)

        return sorted(labels)

        # """See base class."""
        # # return ["0", "1"]
        # return relations

    def _create_examples(self, lines, dataset_type, max_seq_length):
        """Creates examples for the training and dev sets."""
        examples = []

        pbar = tqdm(iterable=lines, desc=f"Processing DocRED {dataset_type}", total=len(lines))
        for i, item in enumerate(pbar):
            title = item['title']
            sentences = item['sents']
            triplets = item['labels']
            entities = adjust_mention_positions(entities=item['vertexSet'], text=sentences)

            entity_pos = []
            for entity in entities:
                mention_positions = []
                for mention in entity:
                    mention_positions.append(mention['pos'])

                entity_pos.append(mention_positions)

            head_tail_pair2relation = {}
            for triplet in triplets:
                relation = int(docred_rel2id[triplet['r']])
                head_idx = triplet['h']
                tail_idx = triplet['t']
                head_tail_pair = (head_idx, tail_idx)

                if head_tail_pair in head_tail_pair2relation:
                    head_tail_pair2relation[head_tail_pair].append(relation)
                else:
                    head_tail_pair2relation[head_tail_pair] = [relation]

            whole_text = ' '.join([' '.join(sentence) for sentence in sentences]).split()
            entity_idxs = list(range(len(entities)))
            all_entity_pairs = list(permutations(iterable=entity_idxs, r=2))

            positive_pairs = list(head_tail_pair2relation.keys())
            negative_pairs = [pair for pair in all_entity_pairs if pair not in positive_pairs]

            relations = []
            head_tail_pairs = []
            for head_tail_pair in head_tail_pair2relation:
                relation = [0] * len(docred_rel2id)

                for sample in head_tail_pair2relation[head_tail_pair]:
                    relation[sample] = 1

                relations.append(relation)
                head_tail_pairs.append(head_tail_pair)

            for head_idx, _ in enumerate(entities):
                for tail_idx, _ in enumerate(entities):
                    head_tail_pair = (head_idx, tail_idx)

                    if (head_idx != tail_idx) and (head_tail_pair not in head_tail_pairs):
                        relation = [1] + ([0] * (len(docred_rel2id) - 1))
                        relations.append(relation)
                        head_tail_pairs.append(head_tail_pair)

            example = DocREDInputExample(f'{title}',
                                         ' '.join(whole_text),
                                         entity_pos,
                                         head_tail_pairs,
                                         relations)
            examples.append(example)

        return examples


semeval_relations = ['Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
                     'Content-Container(e1,e2)', 'Content-Container(e2,e1)',
                     'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
                     'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
                     'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
                     'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
                     'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
                     'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
                     'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
                     'Other'
                     ]

semeval_relations_no_direction = ['Content-Container', 'Cause-Effect', 'Entity-Origin', 'Member-Collection',
                                  'Component-Whole',
                                  'Entity-Destination', 'Instrument-Agency', 'Other', 'Message-Topic',
                                  'Product-Producer']


class SemEvalProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_semeval_txt(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_semeval_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return semeval_relations

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            sentence = line[0].split('\t')[1][1:-1]
            label = line[1]
            # I have checked @ and ^ do not appear in the corpus.
            sentence = sentence.replace('<e1>', '@ ').replace('</e1>', ' @').replace('<e2>', '^ ').replace('</e2>',
                                                                                                           ' ^')
            guid = i
            # text_a: raw text including @ and ^, after word piece, just tokens.index['@'] to get the first index
            text_a = sentence
            # text_b: None
            text_b = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def convert_examples_to_features_tacred(examples, label_list, max_seq_length,
                                        tokenizer, output_mode,
                                        cls_token_at_end=False,
                                        cls_token='[CLS]',
                                        cls_token_segment_id=1,
                                        sep_token='[SEP]',
                                        sep_token_extra=False,
                                        pad_on_left=False,
                                        pad_token=0,
                                        pad_token_segment_id=0,
                                        sequence_a_segment_id=0,
                                        sequence_b_segment_id=1,
                                        mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    pbar = tqdm(iterable=examples, desc="Creating examples for TACRED", total=len(examples))
    for ex_index, example in enumerate(pbar):
        text_a = example.text_a
        subj_start, subj_end, obj_start, obj_end = example.text_b
        relation = example.label

        if subj_start < obj_start:
            tokens = tokenizer.tokenize(' '.join(text_a[:subj_start]))
            subj_special_start = len(tokens)
            tokens += ['@']
            tokens += tokenizer.tokenize(' '.join(text_a[subj_start:subj_end + 1]))
            tokens += ['@']
            tokens += tokenizer.tokenize(' '.join(text_a[subj_end + 1:obj_start]))
            obj_special_start = len(tokens)
            tokens += ['#']
            tokens += tokenizer.tokenize(' '.join(text_a[obj_start:obj_end + 1]))
            tokens += ['#']
            tokens += tokenizer.tokenize(' '.join(text_a[obj_end + 1:]))
        else:
            tokens = tokenizer.tokenize(' '.join(text_a[:obj_start]))
            obj_special_start = len(tokens)
            tokens += ['#']
            tokens += tokenizer.tokenize(' '.join(text_a[obj_start:obj_end + 1]))
            tokens += ['#']
            tokens += tokenizer.tokenize(' '.join(text_a[obj_end + 1:subj_start]))
            subj_special_start = len(tokens)
            tokens += ['@']
            tokens += tokenizer.tokenize(' '.join(text_a[subj_start:subj_end + 1]))
            tokens += ['@']
            tokens += tokenizer.tokenize(' '.join(text_a[subj_end + 1:]))

        _truncate_seq_pair(tokens, [], max_seq_length - 2)
        tokens = ['<s>'] + tokens + ['</s>']
        subj_special_start += 1
        obj_special_start += 1
        relation = label_map[example.label]

        segment_ids = [sequence_a_segment_id] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(label_map[example.label])
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))

        if subj_special_start > max_seq_length:
            subj_special_start = max_seq_length - 10

        if obj_special_start > max_seq_length:
            obj_special_start = max_seq_length - 10

        # Subj and obj start special token positions.
        subj_special_start_id = np.zeros(max_seq_length)
        obj_special_start_id = np.zeros(max_seq_length)
        subj_special_start_id[subj_special_start] = 1
        obj_special_start_id[obj_special_start] = 1

        features.append(
            tacredInputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_id,
                                subj_special_start_id=subj_special_start_id,
                                obj_special_start_id=obj_special_start_id))
    return features


def convert_examples_to_features_docred(examples, label_list, tokenizer, max_seq_length):
    label_map = {label: idx for idx, label in enumerate(label_list)}

    features = []
    pbar = tqdm(iterable=examples, desc="Creating features for DocRED", total=len(examples))
    for idx, example in enumerate(pbar):
        title = example.id
        text = example.text
        entity_positions = example.entity_pos
        labels = example.labels
        head_tail_idxs = example.head_tail_idxs

        for idx, entity in enumerate(entity_positions):
            entity = sorted(entity, key=lambda x: x[0])
            entity_positions[idx] = entity

        words = text.split()
        text_ = ''
        current_idx = 0
        char_lvl_spans = []
        for entity in entity_positions:
            text_ = ''
            current_idx = 0
            mention_char_spans = []

            for mention_span in entity:
                text_ += ' '.join(words[current_idx:mention_span[0]])

                if text_:
                    text_ += ' '

                start = len(text_)
                text_ += ' '.join(words[mention_span[0]:mention_span[1]]) + ' '

                end = len(text_)
                current_idx = mention_span[1]

                mention_char_spans.append([start, end])

            char_lvl_spans.append(mention_char_spans)

        text_sliced_entities = []
        all_spans = sorted(sum(char_lvl_spans, []), key=lambda x: x[0])
        current_idx = 0
        for span in all_spans:
            start = span[0]
            end = span[1]

            text_slice = text[current_idx:start]
            text_sliced_entities.append(text_slice)
            text_sliced_entities.append(text[start:end])
            current_idx = end

        text_sliced_entities.append(text[current_idx:])
        for idx, text_slice in enumerate(text_sliced_entities):
            text_sliced_entities[idx] = text_slice.strip()

        entity_marked_text = (' ' + ENTITY_MARKER + ' ').join(text_sliced_entities)
        entity_marked_text = ' '.join(entity_marked_text.split())

        for idx, entity in enumerate(char_lvl_spans):
            marked_mentions = []
            for mention in entity:
                entity_id_marked = (idx, mention)
                marked_mentions.append(entity_id_marked)

            char_lvl_spans[idx] = marked_mentions

        all_spans = sorted(sum(char_lvl_spans, []), key=lambda x: x[1][0])
        current_idx = 0
        adjustment_length = (len(ENTITY_MARKER) * 2) + 2
        for idx, id_span in enumerate(all_spans):
            entity_id = id_span[0]
            span = id_span[1]

            original_start = span[0]
            original_end = span[1]

            new_start = original_start + (adjustment_length * current_idx)
            new_end = original_end + (adjustment_length * (current_idx + 1))
            new_span = [new_start, new_end]

            new_pair = (entity_id, new_span)
            all_spans[idx] = new_pair

            current_idx += 1

        tokens = [tokenizer.cls_token]
        current_idx = 0
        token_lvl_spans = []
        for idx, id_span in enumerate(all_spans):
            entity_id = id_span[0]
            span = id_span[1]

            start = span[0]
            end = span[1]

            text_chunk = entity_marked_text[current_idx:start]
            tokens += tokenizer.tokenize(text_chunk)

            token_start = len(tokens)

            entity_text_chunk = entity_marked_text[start:end]
            tokens += tokenizer.tokenize(entity_text_chunk)

            token_end = len(tokens)

            token_lvl_id_span = (entity_id, [token_start, token_end])
            token_lvl_spans.append(token_lvl_id_span)

            current_idx = end

        tokens += tokenizer.tokenize(entity_marked_text[current_idx:])
        # tokens = tokens[:max_seq_length - 1]
        tokens = tokens + [tokenizer.sep_token]

        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        # word_ids = word_ids[:max_seq_length]

        word_attention_mask = [1] * len(tokens)
        # word_attention_mask = word_attention_mask[:max_seq_length]

        num_entities = len(entity_positions)
        entity_position_ids = {idx: [] for idx in range(num_entities)}
        for id_span in token_lvl_spans:
            entity_id = id_span[0]
            span = id_span[1]
            entity_position_ids[entity_id].append(span)

        entity_position_ids = list(entity_position_ids.values())

        # # Since we're capping out max_seq_length to 512, we need to ensure entity positions are within that boundary.
        # len_capped_entity_ids = []
        # for entity in entity_position_ids:
        #     entities = []
        #     for mention in entity:
        #         if mention[0] < max_seq_length:
        #             entities.append(mention)
        #     len_capped_entity_ids.append(entities)

        # no_entity_idxs = [idx for idx, entity in enumerate(len_capped_entity_ids) if entity == []]
        # len_capped_head_tail_idxs = []
        # new_labels = []
        # for label, pair in zip(labels, head_tail_idxs):
        #     if not any(set(pair).intersection(set(no_entity_idxs))):
        #         len_capped_head_tail_idxs.append(pair)
        #         new_labels.append(label)

        # len_capped_entity_ids = [x for x in len_capped_entity_ids if x != []]
        # ###########################################################################################

        # assert len(len_capped_head_tail_idxs) == len(new_labels), "Number of (head, tail) pairs and labels don't match."

        feature = DocREDInputFeature(title=title,
                                     word_ids=word_ids,
                                     word_attention_mask=word_attention_mask,
                                     entity_position_ids=entity_position_ids,
                                     labels=labels,
                                     head_tail_idxs=head_tail_idxs)
        features.append(feature)

    return features


def convert_examples_to_features_semeval(examples, label_list, max_seq_length,
                                         tokenizer, output_mode,
                                         cls_token_at_end=False,
                                         cls_token='[CLS]',
                                         cls_token_segment_id=1,
                                         sep_token='[SEP]',
                                         sep_token_extra=False,
                                         pad_on_left=False,
                                         pad_token=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         sequence_b_segment_id=1,
                                         mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_a = example.text_a
        relation = example.label

        tokens = tokenizer.tokenize(text_a)

        _truncate_seq_pair(tokens, [], max_seq_length - 2)
        tokens = ['<s>'] + tokens + ['</s>']
        e1_start = tokens.index('Ġ@')
        e2_start = tokens.index('Ġ^')

        relation = label_map[example.label]

        segment_ids = [sequence_a_segment_id] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            # label_id = label_map[example.label]
            # label_id = [label_map[item] for item in example.label]
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(label_map[example.label])
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))

        e1_start_id = np.zeros(max_seq_length)
        e2_start_id = np.zeros(max_seq_length)
        e1_start_id[e1_start] = 1
        e2_start_id[e2_start] = 1

        features.append(
            semevalInputFeatures(input_ids=input_ids,
                                 input_mask=input_mask,
                                 segment_ids=segment_ids,
                                 label_id=label_id,
                                 e1_start_id=e1_start_id,
                                 e2_start_id=e2_start_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break

        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def micro_f1_tacred(preds, labels):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    NO_RELATION = 28
    for guess, gold in zip(preds, labels):
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))

    return {'precision': prec_micro, 'recall': recall_micro, 'f1': f1_micro}
    # return prec_micro, recall_micro, f1_micro


def micro_f1_docred(preds, labels):
    correct_by_relation = Counter() # TP
    guessed_by_relation = Counter() # TP + FP
    gold_by_relation = Counter() # TP + FN

    NO_RELATION = 0
    for guess, gold in zip(preds, labels):
        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))

    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))

    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))

    results = {'precision': prec_micro,
               'recall': recall_micro,
               'f1': f1_micro}

    return results


def micro_f1(preds, labels):
    return f1_score(y_true=labels, y_pred=preds, average='micro')


def macro_f1(preds, labels):
    return f1_score(y_true=labels, y_pred=preds, average='macro')


def entity_typing_accuracy(out, l):
    def f1(p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)

    def loose_macro(true, pred):
        num_entities = len(true)
        p = 0.
        r = 0.
        for true_labels, predicted_labels in zip(true, pred):
            if len(predicted_labels) > 0:
                p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            if len(true_labels):
                r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
        precision = p / num_entities
        recall = r / num_entities
        return precision, recall, f1(precision, recall)

    def loose_micro(true, pred):
        num_predicted_labels = 0.
        num_true_labels = 0.
        num_correct_labels = 0.
        for true_labels, predicted_labels in zip(true, pred):
            num_predicted_labels += len(predicted_labels)
            num_true_labels += len(true_labels)
            num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
        if num_predicted_labels > 0:
            precision = num_correct_labels / num_predicted_labels
        else:
            precision = 0.
        recall = num_correct_labels / num_true_labels
        return precision, recall, f1(precision, recall)

    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            # if x1[i] > 0 or x1[i] == top:
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, loose_micro(y2, y1), loose_macro(y2, y1)


def macro_f1_semeval(preds, labels):
    def f1_score_semeval(category, category_preds, category_labels):

        true_positive = 0
        false_positive = 0
        false_negative = 0
        for i in range(len(preds)):
            predict_category = semeval_relations[category_preds[i]]
            true_category = semeval_relations[category_labels[i]]

            if not (category in predict_category or category in true_category):
                continue

            # true_positive
            if predict_category == true_category:
                true_positive += 1
                continue

            # false positive
            if category in predict_category and predict_category != true_category:
                false_positive += 1
                continue

            # false negative
            if category in true_category and predict_category != true_category:
                false_negative += 1
                continue

        if true_positive + false_positive == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return f1

    f1_total_score = 0
    for i in range(len(semeval_relations_no_direction)):
        f1_total_score += f1_score_semeval(semeval_relations_no_direction[i], preds, labels)

    return f1_total_score / len(semeval_relations_no_direction)


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)

    if task_name == 'entity_type':
        return entity_typing_accuracy(preds, labels)
    elif task_name == 'tacred':
        return micro_f1_tacred(preds, labels)
    elif task_name == 'docred':
        return micro_f1_docred(preds, labels)
    elif task_name == 'semeval':
        return macro_f1_semeval(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "entity_type": EntityTypeProcessor,
    "tacred": TACREDProcessor,
    "semeval": SemEvalProcessor,
    'docred': DocREDProcessor
}

output_modes = {
    "entity_type": "classification",
    "tacred": "classification",
    "semeval": "classification",
    'docred': 'classification'
}

GLUE_TASKS_NUM_LABELS = {
    "entity_type": 9,
    "tacred": 42,
    "semeval": 19,
    'docred': 97
}
