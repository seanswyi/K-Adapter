import json
import os

import numpy as np


with open(file='/hdd1/seokwon/data/DocRED/DocRED_baseline_metadata/rel2id.json') as f:
    rel2id = json.load(fp=f)
    id2rel = {id_: rel for rel, id_ in rel2id.items()}


def convert_to_official_format(predictions, features):
    head_idxs = []
    tail_idxs = []
    titles = []

    for feature in features:
        head_tail_idxs = feature.head_tail_idxs
        head_idxs += [pair[0] for pair in head_tail_idxs]
        tail_idxs += [pair[1] for pair in head_tail_idxs]
        titles += [feature.title for pair in head_tail_idxs]

    results = []
    for pred_idx in range(predictions.shape[0]):
        prediction = predictions[pred_idx]
        prediction = np.nonzero(prediction)[0].tolist()

        for p in prediction:
            if p != 0:
                template = {'title': titles[pred_idx],
                            'h_idx': head_idxs[pred_idx],
                            't_idx': tail_idxs[pred_idx],
                            'r': id2rel[p]}
                results.append(template)

    return results


def generate_train_facts(data_filename, ground_truth_dir):
    fact_filename = data_filename[data_filename.find('train_'):]
    fact_filename = os.path.join(ground_truth_dir, fact_filename.replace('.json', '.fact'))
    fact_in_train = set([])

    if os.path.exists(fact_filename):
        with open(file=fact_filename) as f:
            triples = json.load(fp=f)

        for triple in triples:
            fact_in_train.add(tuple(triple))

        return fact_in_train

    fact_in_train = set([])

    with open(file=data_filename) as f:
        original_data = json.load(fp=f)

    for sample in original_data:
        entities = sample['vertexSet']
        labels = sample['labels']

        for label in labels:
            relation = label['r']

            for head in entities[label['h']]:
                for tail in entities[label['t']]:
                    triple = (head['name'], tail['name'], relation)
                    fact_in_train.add(triple)

    with open(file=fact_filename, mode='w') as f:
        json.dump(obj=list(fact_in_train), fp=f)

    return fact_in_train


def docred_official_evaluate(tmp, path):
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = generate_train_facts(os.path.join(path, "train_annotated.json"), truth_dir)

    with open(file=os.path.join(path, 'dev.json')) as f:
        truth = json.load(fp=f)

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0

    correct_in_train_annotated = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']

        titleset2.add(title)

        if title not in title2vectexSet:
            continue

        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            in_train_annotated = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True

            if in_train_annotated:
                correct_in_train_annotated += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_r_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (tot_relations - correct_in_train_annotated + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r_ignore_train_annotated / (re_p_ignore_train_annotated + re_r_ignore_train_annotated)

    normal_scores = {'precision': re_p, 'recall': re_r, 'f1': re_f1}
    ignore_scores = {'ign_precision': re_p_ignore_train_annotated, 'ign_recall': re_r_ignore_train_annotated, 'ign_f1': re_f1_ignore_train_annotated}

    return normal_scores, ignore_scores
