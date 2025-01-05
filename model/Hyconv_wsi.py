import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import Parameter
from model.utils.utils import degree_hyedge, degree_node
from model.utils.utils import neighbor_distance, get_full_H, get_full_select_H, get_fusion_H
from einops import rearrange, repeat
from torch import einsum
from torch_kmeans import KMeans


class SelfAttention(nn.Module):
    def __init__(self, in_ch, hidden_dim) -> None:
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.to_q = Parameter(torch.Tensor(in_ch, hidden_dim))
        self.to_k = Parameter(torch.Tensor(in_ch, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q)
        nn.init.xavier_uniform_(self.to_k)

    def forward(self, x):
        q = einsum('nvc,co->nvo', x, self.to_q)
        k = einsum('nvc,co->nvo', x, self.to_k)
        H_attn = einsum('nvc,nkc->nkv', q, k) * self.scale

        return torch.sigmoid(H_attn)


class HyConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_threshold, k_nearest, dropout, bias=True) -> None:
        super().__init__()
        self.k_threshold = k_threshold
        self.k_nearest = k_nearest
        self.drop_out = dropout

        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.relu = nn.ReLU(inplace=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, H, train, coors=None, hyedge_weight=None):
        assert len(x.shape) == 3, 'the input of HyperConv should be N * V * C'

        y = einsum('nvc,co->nvo', x, self.theta)  # x.matmul(self.theta)

        if hyedge_weight is not None:
            Dv = torch.diag_embed(1.0 / (H * hyedge_weight.unsqueeze(1)).sum(-1))
        else:
            Dv = torch.diag_embed(1.0 / H.sum(-1))
        HDv = einsum('nkv,nve->nke', Dv, H)

        De = torch.diag_embed(1.0 / H.sum(1))
        HDe = einsum('nve,nek->nvk', H, De)
        if hyedge_weight is not None:
            HDe = einsum('nve,ne->nve', HDe, hyedge_weight)
        y = einsum('nvc,nve->nec', y, HDe)  # HDe
        y = einsum('nec,nve->nvc', y, HDv)
        y = y + self.bias.unsqueeze(0).unsqueeze(0)

        # y = self.pooling(y)

        return self.relu(y)


class Model(nn.Module):
    def __init__(self, in_channels, n_target, hiddens, k_threshold=None, k_nearest=None, dropout=0.3):
        super().__init__()
        self.drop_out = nn.Dropout(dropout)
        _in = in_channels
        base_channel = hiddens[0]
        self.ft_l1 = HyConv(in_channels, hiddens[0], k_threshold=k_threshold, k_nearest=k_nearest, dropout=dropout)
        self.ft_l2 = HyConv(hiddens[0], hiddens[1], k_threshold=k_threshold, k_nearest=k_nearest, dropout=dropout)
        self.ft_l3 = HyConv(hiddens[1], hiddens[2], k_threshold=k_threshold, k_nearest=k_nearest, dropout=dropout)

        self.last_fc = nn.Linear(hiddens[-1], n_target)

        self.k_nearest = k_nearest
        self.k_threshold = k_threshold

    def forward(self, x, coors=None, train=False):  #
        H_ft, H_ft_edge_weight = self.get_H(x, full=True)
        # H_coor, H_coor_edge_weight = self.get_H(coors, full=True)

        x_ft = self.ft_l1(x, H_ft, train)
        x_ft = self.drop_out(x_ft)
        x_ft = self.ft_l2(x_ft, H_ft, train)
        x_ft = self.drop_out(x_ft)
        x_ft = self.ft_l3(x_ft, H_ft, train)
        x_ft = x_ft.max(dim=1)[0]

        x = self.drop_out(x_ft)
        x = self.last_fc(x)

        return x, x_ft  #

    def patch_contribution(self, x, coors=None, train=False):  #
        H_ft, H_ft_edge_weight = self.get_H(x, full=True)

        x_ft = self.ft_l1(x, H_ft, train)
        x_ft = self.drop_out(x_ft)
        x_ft = self.ft_l2(x_ft, H_ft, train)
        x_ft = self.drop_out(x_ft)
        x_ft = self.ft_l3(x_ft, H_ft, train)

        x_ft = self.drop_out(x_ft)
        x_ft = self.last_fc(x_ft)

        return x_ft  #

    def get_H(self, fts, full=False):
        if full:
            H, edge_weight = get_full_H(fts, k_threshold=self.k_threshold, k_nearest=self.k_nearest)
            return H, edge_weight
        else:
            return neighbor_distance(fts, k_threshold=self.k_threshold, k_nearest=self.k_nearest)

    def set_k_nearst(self, k_nearest):
        self.k_nearest = k_nearest
