import logging

import torch
from torch.utils.data import DataLoader

from utils_glue import compute_metrics, convert_examples_to_features_docred, ENTITY_MARKER, output_modes, processors


logger = logging.getLogger(__name__)


def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()

    logger.info("Creating features from dataset file at %s", args.data_dir)

    if evaluate:
        examples = processor.get_dev_examples(args.data_dir, max_seq_length=args.max_seq_length)
    elif not evaluate:
        examples = processor.get_train_examples(args.data_dir, max_seq_length=args.max_seq_length)

    label_list = processor.get_labels(examples=examples, max_seq_length=args.max_seq_length)

    features = convert_examples_to_features_docred(examples, label_list, tokenizer, args.max_seq_length)
    # features: word_ids, word_attention_mask, entity_position_ids, head_tail_idxs, labels

    if args.debug:
        features = features[:100]

    def docred_collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

            return padded_tensor

        entity_position_ids = [getattr(x, 'entity_position_ids') for x in batch]
        labels = [getattr(x, 'labels') for x in batch]
        head_tail_idxs = [getattr(x, 'head_tail_idxs') for x in batch]

        # Pad the entities and mentions so that we can convert them to tensors. ###################
        max_num_entities = max([len(entity) for entity in entity_position_ids])
        max_num_mentions = 0
        for entity in entity_position_ids:
            for mention in entity:
                if len(mention) > max_num_mentions:
                    max_num_mentions = len(mention)

        for entity_idx, entity in enumerate(entity_position_ids):
            num_entities = len(entity)
            necessary_entity_padding = max_num_entities - num_entities
            entity_position_ids[entity_idx].extend([[]] * necessary_entity_padding)

            for mention_idx, mention in enumerate(entity):
                num_mentions = len(mention)
                necessary_mention_padding = max_num_mentions - num_mentions
                entity_position_ids[entity_idx][mention_idx].extend([[-1, -1]] * necessary_mention_padding)
        ###########################################################################################

        # Pad the head_tail_idxs and labels so we can convert them to tensors. ####################
        max_num_pairs = max([len(pairs) for pairs in head_tail_idxs])
        for pair_idx, pair in enumerate(head_tail_idxs):
            num_pairs = len(pair)
            necessary_pair_padding = max_num_pairs - num_pairs
            head_tail_idxs[pair_idx].extend([(-1, -1)] * necessary_pair_padding)
            labels[pair_idx].extend([[-1] * 97] * necessary_pair_padding)
        ###########################################################################################

        entity_position_ids = torch.tensor(entity_position_ids)
        labels = torch.tensor(labels)
        head_tail_idxs = torch.tensor(head_tail_idxs)

        collated_data = {'word_ids': create_padded_sequence('word_ids', tokenizer.pad_token_id),
                         'word_attention_mask': create_padded_sequence('word_attention_mask', 0),
                         'entity_position_ids': entity_position_ids,
                         'labels': labels,
                         'head_tail_idxs': head_tail_idxs}

        return collated_data

    if not evaluate:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=docred_collate_fn)
    elif evaluate:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        dataloader = DataLoader(features, batch_size=args.eval_batch_size, shuffle=False, collate_fn=docred_collate_fn)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return dataloader
