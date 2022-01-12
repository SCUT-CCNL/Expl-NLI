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

from model import myRationalizer2

from transformers import *
from torch.distributions.categorical import Categorical
from model_classify import BaseSNLI_roberta

# logging.getLogger().setLevel(print)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(1)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str)  # train.json
parser.add_argument('--dev_file', type=str)  # train.json
parser.add_argument('--test_file', type=str)  # train.json
parser.add_argument('--model_name', type=str)
parser.add_argument('--model_to_save', type=str)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--n_epoch', default=10, type=int)
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--selector_model', type=str) # the best model of selector

parser.add_argument('--train_data3', default='../datas/snli_data_dir/train.json', type=str)
parser.add_argument('--dev_data3', default='../datas/snli_data_dir/dev.json', type=str)
parser.add_argument('--test_data3', default='../datas/snli_data_dir/test.json', type=str)
# the premise and hypothesis with golden rationales

args = parser.parse_args()

model_name = 'roberta-base'
config = RobertaConfig.from_pretrained(model_name)
config.num_labels = 3
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model_selector = BaseSNLI_roberta(config).to(device)
model_selector.load_state_dict(torch.load(args.selector_model, map_location='cuda:0'))
model_selector.eval()


def repetition(s1_tokens, s2_tokens, expl_tokens):
    s1 = ' '.join(s1_tokens).lower()
    s2 = ' '.join(s2_tokens).lower()
    expl = ' '.join(expl_tokens).lower()
    if s1 in expl or s2 in expl:
        return True
    return False


def load_dataset(target='train', cased=False):
    data = []
    n_removed = 0
    with open(target, 'r') as f:
        for line in f.readlines():
            data_item = []
            line = line.strip()
            d = json.loads(line)

            if target == 'train' and repetition(d['sentence1'], d['sentence2'], d['explanation']):
                n_removed += 1
                continue

            if cased:
                data_item.append((d['sentence1'], [x for item in d['marked_idx1'] for x in range(item[0], item[1])]))
                data_item.append((d['sentence2'], [x for item in d['marked_idx2'] for x in range(item[0], item[1])]))
                data_item.append(d['explanation'])
            else:
                data_item.append(
                    ([x.lower() for x in d['sentence1']],
                     [x for item in d['marked_idx1'] for x in range(item[0],  item[1])])
                )
                data_item.append(
                    ([x.lower() for x in d['sentence2']],
                     [x for item in d['marked_idx2'] for x in range(item[0], item[1])])
                )

                data_item.append([x.lower() for x in d['explanation']])
            data_item.append(d['label'])
            data.append(data_item)

    print(n_removed)
    return data


def load_dataset2(target, data_set):
    data = []
    label_ids = get_label_ids_list(data_set)
    i = 0
    with open(target, 'r') as f:
        for line in f.readlines():
            data_item = []
            line = line.strip()
            d = json.loads(line)
            premise = d['sentence1']
            hypothesis = d['sentence2']
            golden_label = d['gold_label']
            hints_u = d['hints_u']
            hints_v = d['hints_v']

            for key_u, value_u in hints_u.items():
                j = label_ids[i]
                # print('j_u:',j)
                if key_u == idx2label[j]:
                    data_item.append((premise, [x for item in hints_u[key_u] for x in range(item[0], item[1])]))

            for key_v, value_v in hints_v.items():
                j = label_ids[i]
                # print('j_v:',j)
                if key_v == idx2label[j]:
                    data_item.append((hypothesis, [x for item in hints_v[key_v] for x in range(item[0], item[1])]))
            data_item.append(golden_label)
            data.append(data_item)
            i += 1
    return data


def load_dataset3(target, cased=False):
    data = []
    with open(target, 'r') as f:
        for line in f.readlines():
            line = line.strip().replace('\\', '')
            if not line == '':
                d = json.loads(line)
                data.append(d)
    return data


def load_all_dataset3(cased=False):
    train_data = load_dataset3(args.train_data3, cased)
    dev_data = load_dataset3(args.dev_data3, cased)
    test_data = load_dataset3(args.test_data3, cased)
    return train_data, dev_data, test_data


train_data3, dev_data3, test_data3 = load_all_dataset3(cased=True)

def packing(d):
    max_length = max([len(item) for item in d['input_ids']])
    for i in range(len(d['input_ids'])):
        diff = max_length - len(d['input_ids'][i])
        for _ in range(diff):
            d['input_ids'][i].append(1)  # Roberta: <s>: 0, </s>: 2, <pad>: # Bert: [CLS]: 101, [SEP]: 102, [PAD]: 0
            d['attention_mask'][i].append(0)
    return d


def get_label_id(batch, use_max):
    d_input = {'input_ids': [], 'attention_mask': []}
    for i in range(len(batch)):
        text = "{} </s> {}".format(batch[i]['Premise'].replace('[ ','').replace(' ]',''),
                                   batch[i]['Hypothesis'].replace('[ ','').replace(' ]',''))
        d_cur = tokenizer(text)
        d_input['input_ids'].append(d_cur['input_ids'])
        d_input['attention_mask'].append(d_cur['attention_mask'])
    d_input = packing(d_input)

    with torch.no_grad():
        probs = model_selector.predict_probs(d_input)
        sampled_expl_idx = None
        if not use_max:
            dist = Categorical(probs)
            sampled_expl_idx = dist.sample()
        else:
            _, sampled_expl_idx = torch.max(probs, dim=1)

        _, sampled_expl_idx = torch.max(probs, dim=1)
        label_index = []
        for i in range(len(batch)):
            label_index.append(sampled_expl_idx[i].item())
    return label_index


def get_label_ids_list(data):
    label_ids = []
    batch_size = 1
    batches = [data[x:x + batch_size] for x in range(0, len(data), batch_size)]
    process_bar = tqdm(batches)
    for batch_no, sent_batch in enumerate(process_bar):
        label_ids.append(get_label_id(sent_batch, use_max=False)[0])
        process_bar.update()
    print(len(label_ids))
    return label_ids


def write_to_myHints(file_path, data, include_hints=True, include_label_info=True):
    with open(file_path, 'w') as fw:
        for item in data:
            s1_tokens, hint1_idx = item[0]
            s2_tokens, hint2_idx = item[1]

            label = item[2]
            s1_tokens_with_hint = []
            s2_tokens_with_hint = []
            for ix in range(len(s2_tokens)):
                if include_hints and ix in hint2_idx:
                    s2_tokens_with_hint.append('[ ' + s2_tokens[ix] + ' ]')
                else:
                    s2_tokens_with_hint.append(s2_tokens[ix])
            for ix in range(len(s1_tokens)):
                if include_hints and ix in hint1_idx:
                    s1_tokens_with_hint.append('[ ' + s1_tokens[ix] + ' ]')
                else:
                    s1_tokens_with_hint.append(s1_tokens[ix])
            s1_with_hint = ' '.join(s1_tokens_with_hint)
            s2_with_hint = ' '.join(s2_tokens_with_hint)

            s1_with_hint = s1_with_hint.replace('"', '')
            s2_with_hint = s2_with_hint.replace('"', '')

            # s1 = "Premise : {}".format(s1)
            s1_with_hint = "{" + ('"Premise": "{}"'.format(s1_with_hint))
            s2_with_hint = ", " + ('"Hypothesis": "{}"'.format(s2_with_hint))
            label = ", " + ('"Label": "{}"'.format(label)) + "}"
            if include_label_info:
                fw.writelines(' '.join([s1_with_hint, s2_with_hint, label]) + '\n')
            else:
                fw.writelines(' '.join([s1_with_hint, s2_with_hint]) + '\n')

if __name__ == '__main__':
    label2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    idx2label = {v: k for k, v in label2idx.items()}
    print(idx2label[0])
    print(idx2label[1])
    print(idx2label[2])
    data_train = load_dataset2('./predict_rationales/train-rationale.json', train_data3)
    data_dev = load_dataset2('./predict_rationales/dev-rationale.json', dev_data3)
    data_test = load_dataset2('./predict_rationales/test-rationale.json', test_data3)

    write_to_myHints('../datas/snli_data_dir/ph-with-hints/train_with_our_hints.json', data_train,
                     include_hints=True, include_label_info=True)
    write_to_myHints('../datas/snli_data_dir/ph-with-hints/dev_with_our_hints.json', data_dev, include_hints=True,
                     include_label_info=True)
    write_to_myHints('../datas/snli_data_dir/ph-with-hints/test_with_our_hints.json', data_test,
                     include_hints=True, include_label_info=True)
