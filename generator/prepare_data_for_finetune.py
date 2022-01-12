import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data',default='../datas/snli_data_dir/train.json', type=str)
parser.add_argument('--output',default='../datas/snli_data_dir/train-finetune.txt', type=str)

# parser.add_argument('--data',default='../datas/snli_data_dir/dev.json', type=str)
# parser.add_argument('--output',default='../datas/snli_data_dir/dev-finetune.txt', type=str)
#
# parser.add_argument('--data',default='../datas/snli_data_dir/test.json', type=str)
# parser.add_argument('--output',default='../datas/snli_data_dir/test-finetune.txt', type=str)

args = parser.parse_args()


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
                     [x for item in d['marked_idx1'] for x in range(item[0], item[1])])
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


data = load_dataset(args.data, cased=True)


def write_to_file(file_path, data, include_hints=True, include_label_info=True):
    with open(file_path, 'w') as fw:
        for item in data:
            s1_tokens, _ = item[0]
            s2_tokens, hint2_idx = item[1]
            #             print(s2_tokens)
            #             print(hint2_idx)

            hint_tokens = [s2_tokens[ix] for ix in hint2_idx]
            expl_tokens = item[2]
            label = item[3]

            s1 = ' '.join(s1_tokens)
            s2 = ' '.join(s2_tokens)

            s2_tokens_with_hint = []
            for ix in range(len(s2_tokens)):
                if include_hints and ix in hint2_idx:
                    s2_tokens_with_hint.append('[ ' + s2_tokens[ix] + ' ]')
                else:
                    s2_tokens_with_hint.append(s2_tokens[ix])
            s2_with_hint = ' '.join(s2_tokens_with_hint)

            expl = ' '.join(expl_tokens)

            s1 = "Premise : {}".format(s1)
            s2_with_hint = "Hypothesis : {}".format(s2_with_hint)
            label = "Label : {}".format(label)
            expl = "Explanation : {}".format(expl)

            if include_label_info:
                fw.writelines(' '.join([s1, s2_with_hint, label, expl]) + '\n')
            else:
                fw.writelines(' '.join([s1, s2_with_hint, expl]) + '\n')
            fw.writelines('\n')


def write_to_file2(file_path, data, include_hints=True, include_label_info=True):
    with open(file_path, 'w') as fw:
        for item in data:
            s1_tokens, hint1_idx = item[0]
            s2_tokens, hint2_idx = item[1]
            #             print(s2_tokens)
            #             print(hint2_idx)

            hint_tokens = [s2_tokens[ix] for ix in hint2_idx]
            expl_tokens = item[2]
            label = item[3]

            s1 = ' '.join(s1_tokens)
            s2 = ' '.join(s2_tokens)

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

            expl = ' '.join(expl_tokens)

            # s1 = "Premise : {}".format(s1)
            s1_with_hint = "Premise : {}".format(s1_with_hint)
            s2_with_hint = "Hypothesis : {}".format(s2_with_hint)
            print(s1_with_hint)
            print(s2_with_hint)
            label = "Label : {}".format(label)
            expl = "Explanation : {}".format(expl)

            if include_label_info:
                fw.writelines(' '.join([s1_with_hint, s2_with_hint, label, expl]) + '\n')
            else:
                fw.writelines(' '.join([s1_with_hint, s2_with_hint, expl]) + '\n')
            fw.writelines('\n')


def write_to_file3(file_path, data, include_hints=True, include_label_info=True):
    with open(file_path, 'w') as fw:
        for item in data:
            s1_tokens, hint1_idx = item[0]
            s2_tokens, hint2_idx = item[1]
            #             print(s2_tokens)
            #             print(hint2_idx)

            hint_tokens = [s2_tokens[ix] for ix in hint2_idx]
            expl_tokens = item[2]
            label = item[3]

            s1 = ' '.join(s1_tokens)
            s2 = ' '.join(s2_tokens)

            s1_tokens_with_hint = []
            s2_tokens_with_hint = []
            for ix in range(len(s2_tokens)):
                if include_hints and ix in hint2_idx:
                    s2_tokens_with_hint.append('"explanation: ' + s2_tokens[ix] + ' "')
                else:
                    s2_tokens_with_hint.append(s2_tokens[ix])
            for ix in range(len(s1_tokens)):
                if include_hints and ix in hint1_idx:
                    s1_tokens_with_hint.append('"explanation: ' + s2_tokens[ix] + ' "')
                else:
                    s1_tokens_with_hint.append(s1_tokens[ix])
            s1_with_hint = ' '.join(s1_tokens_with_hint)
            s2_with_hint = ' '.join(s2_tokens_with_hint)

            expl = ' '.join(expl_tokens)

            # s1 = "Premise : {}".format(s1)
            s1_with_hint = "Premise : {}".format(s1_with_hint)
            s2_with_hint = "Hypothesis : {}".format(s2_with_hint)
            print(s1_with_hint)
            print(s2_with_hint)
            label = "Label : {}".format(label)
            expl = "Explanation : {}".format(expl)

            if include_label_info:
                fw.writelines(' '.join([s1_with_hint, s2_with_hint, label, expl]) + '\n')
            else:
                fw.writelines(' '.join([s1_with_hint, s2_with_hint, expl]) + '\n')
            fw.writelines('\n')


write_to_file3(args.output, data, include_hints=True, include_label_info=False)
