import torch
from torch.nn.modules.module import Module
import numpy as np

def gcn_layer(A_hat, D_hat, X, W):
    adj = torch.inverse(D_hat) * A_hat * X
    F = torch.mm(adj, W)
    return F

# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fc = torch.nn.Linear(in_features, out_features)
#
#     def forward(self, inputs, A_hat):
#         D_hat = torch.tensor(torch.sum(A_hat.clone().detach(), dim=0))
#         D_hat = torch.tensor(torch.diag(D_hat.clone().detach()))
#         I = torch.eye(A_hat.shape[0], dtype=torch.float32).to('cuda')
#         support = self.fc(inputs)  # B,L,D
#         H_1 = gcn_layer(A_hat, D_hat, I, support)
#         return H_1
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs, A_hat):
        """
        inputs: B,L,D
        adj: B,L,L
        """
        A_hat = A_hat.cpu().numpy()
        D_hat = np.array(np.sum(A_hat, axis=0))
        D_hat = np.matrix(np.diag(D_hat))
        D_hat = torch.from_numpy(D_hat).to('cuda')
        A_hat = torch.from_numpy(A_hat).to('cuda')
        I = np.eye(A_hat.shape[0], dtype=np.float32)
        I = torch.from_numpy(I).to('cuda')
        support = self.fc(inputs)  # L,D
        H_1 = gcn_layer(A_hat, D_hat, I, support)          #L, D


        return H_1

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'