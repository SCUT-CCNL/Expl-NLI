import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from tqdm import tqdm
import json
import random
import pickle
# import logging

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

# from model import myRationalizer
from model import myRationalizer_no_reader

from transformers import *

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(0)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', default='../datas/snli_data_dir/train.json', type=str)
parser.add_argument('--dev_file', default='../datas/snli_data_dir/dev.json', type=str)
parser.add_argument('--test_file', default='../datas/snli_data_dir/test.json', type=str)
parser.add_argument('--model_name', default='roberta-base', type=str)
parser.add_argument('--model_to_load', default='./myRationale_no_reader_models/epoch9.pk', type=str)
# parser.add_argument('--output_dir',default='./predict_rationales/', type=str)
parser.add_argument('--output_dir', default='./predict_rationales/no_reader_snli', type=str)
args = parser.parse_args()


def load_dataset(target='train', cased=False):
    data = []
    with open(target, 'r') as f:
        for line in f.readlines():
            data_item = []
            line = line.strip()
            d = json.loads(line)
            if cased:
                data_item.append((d['sentence1'], [x for item in d['marked_idx1'] for x in range(item[0], item[1])]))
                data_item.append((d['sentence2'], [x for item in d['marked_idx2'] for x in range(item[0], item[1])]))
                data_item.append(d['explanation'])
            else:
                data_item.append(
                    ([x.lower() for x in d['sentence1']],
                     [x for item in d['marked_idx1'] for x in range(item[0], item[1])])
                )
                data_item.append(
                    ([x.lower() for x in d['sentence2']],
                     [x for item in d['marked_idx2'] for x in range(item[0], item[1])])
                )

                data_item.append([x.lower() for x in d['explanation']])
            data_item.append(d['label'])
            data.append(data_item)
    return data


def load_all_dataset(cased=False):
    train_data = load_dataset(args.train_file, cased)
    dev_data = load_dataset(args.dev_file, cased)
    test_data = load_dataset(args.test_file, cased)
    return train_data, dev_data, test_data


def tokenize_sent(tokens, hints):
    sub_tokens_list = []
    for i in range(len(tokens)):
        sub_tokens = tokenizer.tokenize(tokens[i])
        sub_tokens_list.append(sub_tokens)
    assert len(sub_tokens_list) == len(tokens)

    normalized_tokens = []
    normalized_hints = []
    index_of_original_tokens = []
    for i in range(len(tokens)):
        sub_tokens = sub_tokens_list[i]

        for sub in sub_tokens:
            normalized_tokens.append(sub)

        n = len(sub_tokens)
        cur_sent_len = len(normalized_tokens)

        if i in hints:  # and tokens[i] not in ['are', 'a', 'is', 'in', 'the', 'of', '.', 'for', ',']:
            for j in range(cur_sent_len - n, cur_sent_len):
                normalized_hints.append(j)

        index_of_original_tokens.append(cur_sent_len - n)

    assert len(index_of_original_tokens) == len(tokens)
    return normalized_tokens, normalized_hints, index_of_original_tokens


def prepare_batch_with_lb(batch, batch2):
    batch_ids = []
    batch_mask = []
    batch_targets = []
    batch_index_of_original_tokens = []

    # sents = [' '.join(tp[0]) for _, (tp, lb) in enumerate(batch)]
    labels = [lb for _, (tp, lb) in enumerate(batch)]

    # ---------------------------------------
    gold_lb = labels[0]
    for lb in ['entailment', 'neutral', 'contradiction']:
        if not gold_lb == lb:
            labels.append(lb)
    # ---------------------------------------

    for lb_idx in range(3):
        for ix, (tokens, hints) in enumerate(batch2):
            tokens = [labels[lb_idx]] + tokens
            # print(tokens)
            assert len(tokens) >= len(hints)

            normalized_tokens, normalized_hints, index_of_original_tokens = tokenize_sent(tokens, hints)

            if 'roberta-' in model_name:
                normalized_tokens = ['<s>'] + normalized_tokens + ['</s>']
            else:
                normalized_tokens = ['[CLS]'] + normalized_tokens + ['[SEP]']
            input_attn_mask = [1] * len(normalized_tokens)
            input_ids = tokenizer.convert_tokens_to_ids(normalized_tokens)

            assert len(input_ids) == len(normalized_tokens)

            targets = torch.zeros(len(normalized_tokens)).long().to(device)
            normalized_hints = [x + 1 for x in normalized_hints]  # add 1 for <s>

            targets[normalized_hints] = 1

            index_of_original_tokens = [x + 1 for x in index_of_original_tokens]  # add 1 for <s>

            batch_ids.append(input_ids)
            batch_mask.append(input_attn_mask)
            batch_targets.append(targets)
            batch_index_of_original_tokens.append(index_of_original_tokens)

        max_len = max(len(item) for item in batch_ids)
        for i in range(len(batch_ids)):
            paddings = [1] * (max_len - len(batch_ids[i]))  # 1 for <pad> in roberta and 0 for [PAD] in bert
            if 'bert-' in model_name:
                paddings = [0] * (max_len - len(batch_ids[i]))
            batch_ids[i] += paddings
            batch_mask[i] += paddings
    return batch_ids, batch_mask, batch_targets, batch_index_of_original_tokens


def prepare_batch(batch):
    batch_ids = []
    batch_mask = []
    batch_targets = []
    batch_index_of_original_tokens = []
    for ix, (tokens, hints) in enumerate(batch):
        assert len(tokens) >= len(hints)

        normalized_tokens, normalized_hints, index_of_original_tokens = tokenize_sent(tokens, hints)

        if 'roberta-' in model_name:
            normalized_tokens = ['<s>'] + normalized_tokens + ['</s>']
        else:
            normalized_tokens = ['[CLS]'] + normalized_tokens + ['[SEP]']
        input_attn_mask = [1] * len(normalized_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(normalized_tokens)

        assert len(input_ids) == len(normalized_tokens)

        targets = torch.zeros(len(normalized_tokens)).long().to(device)
        normalized_hints = [x + 1 for x in normalized_hints]  # add 1 for <s>

        targets[normalized_hints] = 1

        index_of_original_tokens = [x + 1 for x in index_of_original_tokens]  # add 1 for <s>

        batch_ids.append(input_ids)
        batch_mask.append(input_attn_mask)
        batch_targets.append(targets)
        batch_index_of_original_tokens.append(index_of_original_tokens)

    max_len = max(len(item) for item in batch_ids)
    for i in range(len(batch_ids)):
        paddings = [1] * (max_len - len(batch_ids[i]))  # 1 for <pad> in roberta and 0 for [PAD] in bert
        if 'bert-' in model_name:
            paddings = [0] * (max_len - len(batch_ids[i]))
        batch_ids[i] += paddings
        batch_mask[i] += paddings

    # ---------------------------------------
    batch_ids = [batch_ids[0] for _ in range(3)]
    batch_mask = [batch_mask[0] for _ in range(3)]
    batch_targets = [batch_targets[0] for _ in range(3)]
    batch_index_of_original_tokens = [batch_index_of_original_tokens[0] for _ in range(3)]
    # ---------------------------------------

    return batch_ids, batch_mask, batch_targets, batch_index_of_original_tokens


def prepare_lb_batch(batch):
    lbs = []
    for ix, label in enumerate(batch):
        lbs.append(label2idx[label])

    # ---------------------------------------
    gold_lb = batch[0]
    for lb in ['entailment', 'neutral', 'contradiction']:
        if not gold_lb == lb:
            lbs.append(label2idx[lb])
    # ---------------------------------------

    return lbs


def prepare_batch_final(batch):
    u_ids, u_mask, u_targets, u_index_of_original_tokens = prepare_batch_with_lb([(x[0], x[3]) for x in batch],
                                                                                 [x[0] for x in batch])
    v_ids, v_mask, v_targets, v_index_of_original_tokens = prepare_batch([x[1] for x in batch])
    lbs = prepare_lb_batch([x[3] for x in batch])
    u_tuple = (u_ids, u_mask, u_targets, u_index_of_original_tokens)
    v_tuple = (v_ids, v_mask, v_targets, v_index_of_original_tokens)
    return u_tuple, v_tuple, lbs


def evaluate(data, output_file_name):
    gold_all_u, gold_all_v, pred_all_u, pred_all_v = [], [], [], []
    with torch.no_grad():
        with open(output_file_name, 'w') as fw:
            batches = [data[x:x + batch_size] for x in range(0, len(data), batch_size)]
            process_bar = tqdm(batches)
            for batch_no, batch in enumerate(process_bar):
                input1_tuple, input2_tuple, lbs = prepare_batch_final(batch)
                logits_u, logits_v = model(input1_tuple[:2], input2_tuple[:2])
                scores_u = F.softmax(logits_u, dim=-1)
                scores_v = F.softmax(logits_v, dim=-1)
                # process_bar.set_postfix(scores=scores)
                process_bar.update()
                probs_u = scores_u[:, :, 1]
                probs_v = scores_v[:, :, 1]
                idx_u = torch.zeros(probs_u.shape).long()
                idx_v = torch.zeros(probs_v.shape).long()
                idx_u[probs_u > 0.5] = 1
                idx_v[probs_v > 0.5] = 1
                len_v = len(input2_tuple[0])
                for i in range(len_v):
                    interested_indexes_v = input2_tuple[3][i]

                    update_interested_indexes_v = []
                    for j in interested_indexes_v:
                        update_interested_indexes_v.append(j)

                    gold_v = input2_tuple[2][i][update_interested_indexes_v].tolist()

                    pred_v = idx_v[i, update_interested_indexes_v].tolist()

                    gold_all_v.extend(gold_v)
                    pred_all_v.extend(pred_v)
                len_u = len(input1_tuple[0])
                for i in range(len_u):
                    interested_indexes_u = input1_tuple[3][i]

                    tokens = tokenizer.convert_ids_to_tokens(input1_tuple[0][i])
                    update_interested_indexes_u = []
                    for j in interested_indexes_u:
                        update_interested_indexes_u.append(j)

                    gold_u = input1_tuple[2][i][update_interested_indexes_u].tolist()
                    pred_u = idx_u[i, update_interested_indexes_u].tolist()

                    gold_all_u.extend(gold_u)
                    pred_all_u.extend(pred_u)

                ####################################
                data_item = batch[0]
                gold_sent1_tokens, gold_marked_idx1 = data_item[0]
                gold_sent2_tokens, gold_marked_idx2 = data_item[1]
                gold_expl_tokens = data_item[2]
                gold_label = data_item[3]

                pred_hints_u = []
                for i in range(3):
                    hints = []
                    pred = idx_u[i, interested_indexes_u].tolist()
                    for j in range(len(pred)):
                        if pred[j] == 1:
                            hints.append([j, j + 1])
                    pred_hints_u.append(hints)

                pred_hints_v = []
                for i in range(3):
                    hints = []
                    pred = idx_v[i, interested_indexes_v].tolist()
                    for j in range(len(pred)):
                        if pred[j] == 1:
                            hints.append([j, j + 1])
                    pred_hints_v.append(hints)

                d = {}
                d['sentence1'] = gold_sent1_tokens
                d['sentence2'] = gold_sent2_tokens
                #                 d['marked_idx1'] = [[j, j+1] for j in gold_marked_idx1]
                #                 d['marked_idx2'] = pred_marked_idx2
                d['gold_label'] = gold_label
                d['gold_explanation'] = gold_expl_tokens

                d['hints_u'] = {}

                # d['hints_u'][gold_label] = [[j, j+1] for j in gold_marked_idx1] # use this for train data
                d['hints_u'][gold_label] = pred_hints_u[0]  # use this for test data.

                d['hints_u'][idx2label[lbs[1]]] = pred_hints_u[1]
                d['hints_u'][idx2label[lbs[2]]] = pred_hints_u[2]

                d['hints_v'] = {}
                # d['hints_v'][gold_label] = [[j, j+1] for j in gold_marked_idx2] # use this for train data
                d['hints_v'][gold_label] = pred_hints_v[0]  # use this for test data.

                d['hints_v'][idx2label[lbs[1]]] = pred_hints_v[1]
                d['hints_v'][idx2label[lbs[2]]] = pred_hints_v[2]

                fw.writelines(json.dumps(d) + '\n')

    print(
        classification_report(
            gold_all_u, pred_all_u, target_names=['O_u', 'V_u'], digits=4
        )
    )
    print(
        classification_report(
            gold_all_v, pred_all_v, target_names=['O_v', 'V_v'], digits=4
        )
    )
    report_dict_u = classification_report(gold_all_u, pred_all_u, target_names=['O_u', 'V_u'], output_dict=True)
    report_dict_v = classification_report(gold_all_v, pred_all_v, target_names=['O_v', 'V_v'], output_dict=True)
    return report_dict_u['V_u']['recall'], report_dict_v['V_v']['recall']


if __name__ == '__main__':
    label2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    idx2label = {v: k for k, v in label2idx.items()}

    train_data, dev_data, test_data = load_all_dataset(cased=True)

    batch_size = 1
    model_name = args.model_name

    tokenizer = None
    if model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name)

    model = myRationalizer_no_reader(model_name).to(device)
    model.load_state_dict(torch.load(args.model_to_load))
    model.eval()
    evaluate(train_data, '{}/train-rationale.json'.format(args.output_dir))
    evaluate(dev_data, '{}/dev-rationale.json'.format(args.output_dir))
    evaluate(test_data, '{}/test-rationale.json'.format(args.output_dir))
