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
