"""
Time:2023.11.21
Author: Xiaokun Li
Organization: Harbin Institute of Technology
"""
#We built a simple demo for users to get the attention score of their own predictors

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            # print("V_num:", v.shape)
            # print("V_num:", q.shape)
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num) # get the attention score map
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits) # get the features
        return logits, att_maps


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

#if you have already got your own attention score
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_ticks = ['=O', 'HN', 'N-N', 'H2N', "Br"]
y_ticks = ['1','2','3','4','5','6','7','8','9','10']

GAT = pd.read_csv('GAT.csv')
CNN = pd.read_csv('CNN.csv')
MT = pd.read_csv('MT.csv')

GAT = np.asarray(GAT)
CNN = np.asarray(CNN)
MT = np.asarray(MT)


# cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

#BuGn_r,YlGnBu, BuPu, Dark2_r, winter_r，Accent
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
ax1 = sns.heatmap(GAT,cmap="PuBu",xticklabels=x_ticks, yticklabels=y_ticks, linewidths=.5, cbar=False)
ax1.set_title('GAT')
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)


plt.subplot(1,3,2)
ax2 = sns.heatmap(CNN,cmap="PuBu",xticklabels=x_ticks, yticklabels=y_ticks, linewidths=.5, cbar=False)
ax2.set_title('CNN')
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

plt.subplot(1,3,3)

ax3 = sns.heatmap(MT,cmap="PuBu",xticklabels=x_ticks, yticklabels=y_ticks, linewidths=.5)
ax3.set_title('MT')
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)

plt.savefig('attention.png', dpi=600)
plt.show()
