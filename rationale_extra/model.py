import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from transformers import *

if torch.cuda.is_available():
    device = 'cuda'
    # torch.cuda.set_device(1)

class Rationale_extra(torch.nn.Module):
    def __init__(self, model_name):
        super(myRationalizer, self).__init__()
        self.lstm_size = 450

        if 'roberta' in model_name:
            self.encoder = RobertaModel.from_pretrained(model_name)
        else:
            self.encoder = BertModel.from_pretrained(model_name)

        self.phReadBILSTM = nn.LSTM(input_size=768,hidden_size=self.lstm_size,num_layers=1,
                               bidirectional=True,batch_first=True)
        self.lstmEncoder = nn.LSTM(input_size=768,hidden_size=self.lstm_size,num_layers=1,
                               bidirectional=True,batch_first=True)

        self.w1 = torch.nn.Parameter(torch.randn(2*self.lstm_size, 2*self.lstm_size))
        self.w2 = torch.nn.Parameter(torch.randn(2*self.lstm_size, 2*self.lstm_size))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.lstm_size * 8, 2)
        )

    def forward(self, u_tuple, v_tuple):
        u_ids, u_attn_mask = u_tuple
        v_ids, v_attn_mask = v_tuple

        u_ids = torch.LongTensor(u_ids).to(device)
        u_attn_mask = torch.Tensor(u_attn_mask).to(device)
        v_ids = torch.LongTensor(v_ids).to(device)
        v_attn_mask = torch.Tensor(v_attn_mask).to(device)

        # u_ids = u_ids.to(device)
        # u_attn_mask = u_attn_mask.to(device)
        # v_ids = v_ids.to(device)
        # v_attn_mask = v_attn_mask.to(device)

        u_out = self.encoder(u_ids, attention_mask=u_attn_mask)
        v_out = self.encoder(v_ids, attention_mask=v_attn_mask)

        u_states = u_out.last_hidden_state                         # last_hidden_states---- shape: batch*seq*emb_size
        v_states = v_out.last_hidden_state

        out_phBiLSTM_v, (hn_phBiLSTM_v, cn_phBiLSTM_v) = self.phReadBILSTM(input=v_states)
        out_lstmEncoder_u, (hn_lstmencoder_u, cn_lstmencoder_u) = self.lstmEncoder(u_states, [hn_phBiLSTM_v, cn_phBiLSTM_v]) # out_lstmEncoder.shpae = bs * seq_len * 2hidden_size
        out_phBiLSTM_u, (hn_phBiLSTM_u, cn_phBiLSTM_u) = self.phReadBILSTM(input=u_states)
        out_lstmEncoder_v, (hn_lstmencoder_v, cn_lstmencoder_v) = self.lstmEncoder(v_states, [hn_phBiLSTM_u, cn_phBiLSTM_u]) # out_lstmEncoder.shpae = bs * seq_len * 2hidden_size

        # cat_ph = torch.cat((out_lstmEncoder_u, out_lstmEncoder_v),dim=-1)
        bsize = v_ids.size(0)
        stepsize = out_lstmEncoder_v.size(1)
        scores_ph = torch.bmm(torch.tanh(torch.bmm(out_lstmEncoder_u, self.w1.unsqueeze(0).expand(bsize, -1, -1))),
                           out_lstmEncoder_v.transpose(1, 2))
        # print(scores_ph.size(),'\n.....', out_lstmEncoder_u.size())
        scores_ph = F.softmax(scores_ph.transpose(1, 2), dim=-1)
        # print(scores_ph.size(),'\n.....', out_lstmEncoder_u.size())
        attention_hidden_v = torch.bmm(scores_ph, out_lstmEncoder_u)

        scores_hp = torch.bmm(torch.tanh(torch.bmm(out_lstmEncoder_v, self.w2.unsqueeze(0).expand(bsize, -1, -1))),
                           out_lstmEncoder_u.transpose(1, 2))
        scores_hp = F.softmax(scores_hp.transpose(1, 2), dim=-1)
        attention_hidden_u = torch.bmm(scores_hp, out_lstmEncoder_v)

        # logits_u = self.classifier(attention_hidden_u)
        # logits_v = self.classifier(attention_hidden_v)
        # return logits_u, logits_v

        cat_u = torch.cat((out_lstmEncoder_u, attention_hidden_u, out_lstmEncoder_u-attention_hidden_u, out_lstmEncoder_u*attention_hidden_u), dim=-1)
        cat_v = torch.cat((out_lstmEncoder_v, attention_hidden_v, out_lstmEncoder_v-attention_hidden_v, out_lstmEncoder_v*attention_hidden_v), dim=-1)
        logits_u = self.classifier(cat_u)
        logits_v = self.classifier(cat_v)

        return logits_u, logits_v


