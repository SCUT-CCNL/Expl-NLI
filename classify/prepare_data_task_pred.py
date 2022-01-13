import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../datas/snli_data_dir/train-gen.txt', type=str)
parser.add_argument('--output',default='../datas/snli_data_dir/train-r-with-e.json', type=str)
# parser.add_argument('--data', default='../datas/snli_data_dir/dev-gen.txt', type=str)
# parser.add_argument('--output',default='../datas/snli_data_dir/dev-r-with-e.json', type=str)
# parser.add_argument('--data', default='../datas/snli_data_dir/test-gen.txt', type=str)
# parser.add_argument('--output',default='../datas/snli_data_dir/test-r-with-e.json', type=str)
args = parser.parse_args()

with open(args.data, 'r') as f:
    data = []
    d = {}
    for line in f.readlines():
        line = line.strip()
        if not line == '':
            items = line.split('\t')

            idx = items[0]
            gold_label = items[1]
            x = items[-1]
            premise = x[:x.index('Hypothesis :')][10:]

            hypothesis = x[x.index('Hypothesis :') + 13: x.index('Explanation :')].strip()

            expl = x[x.index('Explanation :'):][13:].strip()

            d['premise'] = premise
            d['hypothesis'] = hypothesis
            d['gold_label'] = gold_label

            if not 'expl' in d:
                d['expl'] = {}

            cur_label = ''
            if len(d['expl']) == 0:
                cur_label = 'entailment'
            elif len(d['expl']) == 1:
                cur_label = 'neutral'
            elif len(d['expl']) == 2:
                cur_label = 'contradiction'

            d['expl'][cur_label] = expl

            if len(d['expl']) == 3:
                assert all([lb in d['expl'] for lb in ['entailment', 'neutral', 'contradiction']])
                data.append(d)
                d = {}

with open(args.output, 'w') as fw:
    for item in data:
        fw.writelines(json.dumps(item) + '\n')
