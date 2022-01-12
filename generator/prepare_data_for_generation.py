import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data',default='../datas/snli_data_dir/train_with_our_hints.json', type=str)  #train-rationale.json
parser.add_argument('--output',default='./data-with-our-exp/trian-prompts.json', type=str) #train-prompts.txt

# parser.add_argument('--data',default='../datas/snli_data_dir_new/test_with_our_hints.json', type=str)
# parser.add_argument('--output',default='./data-with-our-exp/ablate/test-prompts.json', type=str)

# parser.add_argument('--data',default='../datas/snli_data_dir_new/dev_with_our_hints.json', type=str)
# parser.add_argument('--output',default='./data-with-our-exp/ablate/dev-prompts.json', type=str)

args = parser.parse_args()

def repetition(s1_tokens, s2_tokens, expl_tokens):
    s1 = ' '.join(s1_tokens).lower()
    s2 = ' '.join(s2_tokens).lower()
    expl = ' '.join(expl_tokens).lower()
    if s1 in expl or s2 in expl:
        return True
    return False


def load_dataset(target, cased=False):
    data = []
    with open(target, 'r') as f:
        for line in f.readlines():
            line = line.strip().replace('\\','')
            if not line == '':
                data.append(json.loads(line.strip()))
    return data

data = load_dataset(args.data, cased=True)

def write_to_file(file_path, data):
    with open(file_path, 'w') as fw:
        for instance_id, d in enumerate(data):
            gold_label = d['Label']
            hypo_tokens = d['Hypothesis']
            prem_tokens = d['Premise']
            prem_tokens_no_hints = prem_tokens.replace('[ ','').replace(' ]','').replace(" 's"," is")
            s1_with_hints = "Premise : {}".format(prem_tokens_no_hints)
            s2_with_hints = "Hypothesis : {}".format(hypo_tokens)
            label_info = "Label : {}".format(gold_label)
            expl = "Explanation :"

            prompt = ' '.join([s1_with_hints, s2_with_hints, expl])
            line = '\t'.join([str(instance_id), gold_label, prompt])
            fw.writelines(line + '\n')

write_to_file(args.output, data)
