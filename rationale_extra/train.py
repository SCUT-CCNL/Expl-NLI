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

from model import myRationalizer

from transformers import *

# logging.getLogger().setLevel(print)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(0)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file',default='../datas/snli_data_dir/train.json', type=str)  # train.json
parser.add_argument('--dev_file',default='../datas/snli_data_dir/dev.json', type=str)  # train.json
parser.add_argument('--test_file', default='../datas/snli_data_dir/test.json', type=str)  # train.json
parser.add_argument('--model_name', default='roberta-base', type=str)
# parser.add_argument('--model_to_save', type=str)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--n_epoch', default=10, type=int)
args = parser.parse_args()

def load_dataset(target='train', cased=False):
    data = []
    with open(target, 'r') as f:
        for line in f.readlines():
            data_item = []
            line = line.strip()
            d = json.loads(line)
            if cased:
                data_item.append((d['label']+'<s>'+d['sentence1'], [x for item in d['marked_idx1'] for x in range(item[0], item[1])]))
                data_item.append((d['sentence2'], [x for item in d['marked_idx2'] for x in range(item[0], item[1])]))
                data_item.append(d['explanation'])
            else:
                data_item.append(
                    ([d['label']+' <s> '] + [x.lower() for x in d['sentence1']],
                     [x for item in d['marked_idx1'] for x in range(item[0], item[1])])
                )
                data_item.append(
                    ([x.lower() for x in d['sentence2']],
                     [x for item in d['marked_idx2'] for x in range(item[0], item[1])])
                )

                data_item.append([x.lower() for x in d['explanation']])
            data_item.append(d['label'])
            data.append(data_item)
    print(data[0:2])
    return data
    # [([premise list],[hints list]),([hypothesis list],[hints list]), [explain list], label]
    # [(['a', 'person', 'on', 'a', 'horse', 'jumps', 'over', 'a', 'broken', 'down', 'airplane', '.'], []),
    # (['a', 'person', 'is', 'training', 'his', 'horse', 'for', 'a', 'competition', '.'], [3, 4, 5]),
    # ['the', 'person', 'is', 'not', 'necessarily', 'training', 'his', 'horse'], 'neutral']


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


def prepare_batch_with_lb(batch):
    sents = [' '.join(tp[0]) for _, (tp, lb) in enumerate(batch)]
    labels = [lb for _, (tp, lb) in enumerate(batch)]

    # <s> label </s> sentence </s>
    d2 = tokenizer(sents)
    d1 = {'input_ids': [[] for _ in range(len(batch))], 'attention_mask': [[] for _ in range(len(batch))]}
    d1 = tokenizer(labels)

    for i in range(len(d1['input_ids'])):
        d1['input_ids'][i].extend(d2['input_ids'][i][1:])
        d1['attention_mask'][i].extend(d2['attention_mask'][i][1:])

    max_length = max([len(item) for item in d1['input_ids']])
    for i in range(len(d1['input_ids'])):
        diff = max_length - len(d1['input_ids'][i])
        for _ in range(diff):
            if 'roberta-' in model_name:
                d1['input_ids'][i].append(1)  # Roberta: <s>: 0, </s>: 2, <pad>: 1
            else:  # Bert: [CLS]: 101, [SEP]: 102, [PAD]: 0
                d1['input_ids'][i].append(0)
            d1['attention_mask'][i].append(0)
    return d1


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
    return batch_ids, batch_mask, batch_targets, batch_index_of_original_tokens


def prepare_lb_batch(batch):
    lbs = []
    for ix, label in enumerate(batch):
        lbs.append(label2idx[label])
    return lbs


def prepare_batch_final(batch):
    u_ids, u_mask, u_targets, u_index_of_original_tokens = prepare_batch([x[0] for x in batch])
    v_ids, v_mask, v_targets, v_index_of_original_tokens = prepare_batch([x[1] for x in batch])
    lbs = prepare_lb_batch([x[3] for x in batch])
    u_tuple = (u_ids, u_mask, u_targets, u_index_of_original_tokens)
    v_tuple = (v_ids, v_mask, v_targets, v_index_of_original_tokens)
    return u_tuple, v_tuple, lbs


def train(batch, epoch):
    optimizer.zero_grad()

    u_tuple, v_tuple, lbs = prepare_batch_final(batch)
    logits_u, logits_v = model(u_tuple[:2], v_tuple[:2])

    loss = 0.0
    for i in range(len(v_tuple[0])):
        sent_length_v = v_tuple[2][i].size(0)
        loss_v = F.cross_entropy(logits_v[i, :sent_length_v], v_tuple[2][i][:sent_length_v])
        loss += loss_v
    for i in range(len(u_tuple[0])):
        sent_length_u = u_tuple[2][i].size(0)
        loss_u = F.cross_entropy(logits_u[i, :sent_length_u], u_tuple[2][i][:sent_length_u])
        loss += loss_u

    loss_dict = {'loss': loss.item()}
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_dict


def evaluate(data):
    gold_all_u, gold_all_v, pred_all_u, pred_all_v = [], [], [], []
    with torch.no_grad():
        batches = [data[x:x + batch_size] for x in range(0, len(data), batch_size)]
        for batch_no, batch in enumerate(batches):
            input1_tuple, input2_tuple, lbs = prepare_batch_final(batch)
            logits_u, logits_v = model(input1_tuple[:2], input2_tuple[:2])

            scores_u = F.softmax(logits_u, dim=-1)
            scores_v = F.softmax(logits_v, dim=-1)
            probs_u, idx_u = torch.max(scores_u, dim=-1)
            probs_v, idx_v = torch.max(scores_v, dim=-1)

            for i in range(len(input1_tuple[0])):
                interested_indexes_u = input1_tuple[3][i]
                tokens = tokenizer.convert_ids_to_tokens(input1_tuple[0][i])
                update_interested_indexes_u = []
                for j in interested_indexes_u:
                    if tokens[j] not in ['are', 'a', 'is', 'in', 'the', 'of', '.', 'for', ',']:
                        update_interested_indexes_u.append(j)
                gold_u = input1_tuple[2][i][update_interested_indexes_u].tolist()
                pred_u = idx_u[i, update_interested_indexes_u].tolist()
                gold_all_u.extend(gold_u)
                pred_all_u.extend(pred_u)

            for i in range(len(input2_tuple[0])):
                interested_indexes_v = input2_tuple[3][i]
                tokens = tokenizer.convert_ids_to_tokens(input2_tuple[0][i])
                update_interested_indexes_v = []
                for j in interested_indexes_v:
                    if tokens[j] not in ['are', 'a', 'is', 'in', 'the', 'of', '.', 'for', ',']:
                        update_interested_indexes_v.append(j)
                gold_v = input2_tuple[2][i][update_interested_indexes_v].tolist()
                pred_v = idx_v[i, update_interested_indexes_v].tolist()
                gold_all_v.extend(gold_v)
                pred_all_v.extend(pred_v)

            if batch_no == 0:
                sample_idx_u = 0
                sample_idx_v = 0
                # print("Sentence1: {}".format(' '.join(batch[sample_idx][0][0])))
                print("Sentence1: {}".format(' '.join(tokenizer.convert_ids_to_tokens(input1_tuple[0][sample_idx_u]))))
                print("Sentence2: {}".format(' '.join(tokenizer.convert_ids_to_tokens(input2_tuple[0][sample_idx_v]))))
                print('Label: {}'.format(idx2label[lbs[sample_idx_u]]))
                interested_indexes_v = input2_tuple[3][sample_idx_v]
                interested_indexes_u = input1_tuple[3][sample_idx_u]
                gold_u = input1_tuple[2][sample_idx_u][interested_indexes_u].tolist()
                gold_v = input2_tuple[2][sample_idx_v][interested_indexes_v].tolist()
                pred_u = idx_u[sample_idx_u, interested_indexes_u].tolist()
                pred_v = idx_v[sample_idx_v, interested_indexes_v].tolist()
                print("Target hints_u: {}".format(gold_u))
                print("Pred hints_u: {}".format(pred_u))
                print("Target hints_v: {}".format(gold_v))
                print("Pred hints_v: {}".format(pred_v))
                print("Probs_u: {}".format(scores_u[sample_idx_u, interested_indexes_u, 1].tolist()))
                print("Probs_v: {}".format(scores_v[sample_idx_v, interested_indexes_v, 1].tolist()))
    print(
            classification_report(
                gold_all_u, pred_all_u, target_names=['O_u', 'V_u']
            )
        )
    print(
            classification_report(
                gold_all_v, pred_all_v, target_names=['O_v', 'V_v']
            )
        )
    report_dict_u = classification_report(gold_all_u, pred_all_u, target_names=['O_u', 'V_u'], output_dict=True)
    report_dict_v = classification_report(gold_all_v, pred_all_v, target_names=['O_v', 'V_v'], output_dict=True)
    return report_dict_u['V_u']['recall'], report_dict_v['V_v']['recall'], report_dict_u['accuracy'], report_dict_v[
        'accuracy']
    # return report_dict_u['V_u']['precision']+report_dict_v['V_v']['precision']


if __name__ == '__main__':
    label2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    idx2label = {v: k for k, v in label2idx.items()}

    train_data, dev_data, test_data = load_all_dataset(cased=False)

    n_epoch = args.n_epoch
    lr = args.lr
    batch_size = args.batch_size

    model_name = args.model_name
    # model = Rationalizer(model_name).to(device)
    model = myRationalizer(model_name).to(device)

    tokenizer = None
    if model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in list(model.named_parameters()) if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
        {"params": [p for n, p in list(model.named_parameters()) if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_epoch)

    best_f1_dev_u, best_f1_dev_v, best_epoch_dev = 0, 0, 0
    for epoch in range(n_epoch):
        prev_lr = lr
        random.shuffle(train_data)

        batches = [train_data[x:x + batch_size] for x in range(0, len(train_data), batch_size)]
        process_bar = tqdm(batches, desc='epoch:' + str(epoch))
        model.train()

        current_loss_dict = {'loss': 0}

        seen_sentences, modulo = 0, max(1, int(len(batches) / 10))
        for batch_no, sent_batch in enumerate(process_bar):
            batch_loss_dict = train(sent_batch, epoch)

            for k in current_loss_dict:
                current_loss_dict[k] += batch_loss_dict[k]

            seen_sentences += len(sent_batch)
            # if batch_no % modulo == 0:
            process_bar.set_postfix(loss=str(current_loss_dict['loss'] / seen_sentences))
            process_bar.update()
            iteration = epoch * len(batches) + batch_no

        for k in current_loss_dict:
            current_loss_dict[k] /= len(train_data)

        scheduler.step()

        model.eval()
        print('-' * 100)
        print('---------- dev data ---------')
        random.shuffle(dev_data)
        f1_dev_u, f1_dev_v, u_acc, v_acc = evaluate(dev_data)

        if f1_dev_v > best_f1_dev_v:
            best_f1_dev_v = f1_dev_v
            best_epoch_dev = epoch
        print('best acc_v: {}, best epoch: {}'.format(best_f1_dev_v, best_epoch_dev))
        print('v_acc:\t', v_acc)
        if f1_dev_u > best_f1_dev_u:
            best_f1_dev_u = f1_dev_u
        print('best acc_u: ', str(best_f1_dev_u))
        print('u_acc: ', u_acc)
        torch.save(model.state_dict(), './rationale_models/train_rationales_epoch'+str(epoch)+'.pk')
