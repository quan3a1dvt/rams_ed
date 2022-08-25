import torch
from torch.nn.modules.module import Module
import numpy as np
import torch.nn as nn



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
    def forward(self, inputs, adjs):
        """
        inputs: B,L,D
        adj: B,L,L
        """

        support = self.fc(inputs)  # B,L,D
        output = torch.bmm(adjs, support)  # B,L,L x B,L,D -> B, L, D
        denom = torch.sum(adjs, dim=2, keepdim=True) + 1
        output = output / denom
        output = self.relu(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    