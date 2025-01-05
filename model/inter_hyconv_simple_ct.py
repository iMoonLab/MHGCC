import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import Parameter
from model.utils.utils import degree_hyedge, degree_node
from model.utils.utils import neighbor_distance, get_full_H, weight_init, pairwise_euclidean_distance_2d
from einops import rearrange, repeat
from torch import einsum


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class HyConv(nn.Module):
    def __init__(self, in_ch, out_ch, drop_out=0.3, bias=True) -> None:
        super().__init__()
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        self.drop_out_ratio = drop_out

        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, H: torch.Tensor, mask=None, hyedge_weight=None):
        assert len(x.shape) == 2, 'the input of HyperConv should be N * C'

        y = einsum('nc,co->no', x, self.theta)
        y = y + self.bias.unsqueeze(0)

        if hyedge_weight is not None:
            Dv = torch.diag_embed(1.0 / (H * hyedge_weight).sum(1))
        else:
            Dv = torch.diag_embed(1.0 / H.sum(1))
        De = torch.diag_embed(1.0 / H.sum(0))
        if mask is not None:
            H = H * mask

        HDv = einsum('kv,ve->ke', Dv, H)
        HDe = einsum('ve,ek->vk', H, De)
        if hyedge_weight is not None:
            HDe *= hyedge_weight
        y = einsum('vc,ve->ec', y, HDe)  # HDe

        y = einsum('ec,ve->vc', y, HDv)

        return y


class Model(nn.Module):
    def __init__(self, in_channels, linear_hiddens, n_target, k_threshold=None, k_nearest=None,
                k_nearest_wsi=None, dropout=0.3):
        super().__init__()
        self.drop_out = nn.Dropout(dropout)

        _in = in_channels
        self.ct_hyconvs = []

        self.fc = nn.Linear(_in + linear_hiddens[-1] + 2, n_target)

        for _h in linear_hiddens:
            self.ct_hyconvs.append(HyConv(_in, _h))
            _in = _h
        self.ct_hyconvs = nn.ModuleList(self.ct_hyconvs)


        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.k_nearest = k_nearest
        self.k_threshold = k_threshold
        self.k_nearest_wsi = k_nearest_wsi
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(_init_vit_weights)

    def get_features_H(self, features, K):
        dis = pairwise_euclidean_distance_2d(features)
        norm_dis = dis / torch.max(dis)
        min_topk, _ = torch.topk(norm_dis, K * 2, dim=1, largest=False)
        median = torch.median(min_topk)

        ones = torch.ones_like(dis)
        zeros = torch.zeros_like(dis)
        H = torch.where(norm_dis < median, ones, zeros)

        return H

    def forward(self, x_ct, x_wsi, x_cli=None, train=False):  # , stage=None, stage_t=None, stage_m=None ,stage_n=None,
        res = x_ct

        H_wsi = self.get_features_H(x_wsi, self.k_nearest_wsi)
        H_ct_1 = self.get_features_H(x_ct[:, :512], self.k_nearest)
        H_ct_2 = self.get_features_H(x_ct[:, 512:], self.k_nearest)
        H_ct = torch.logical_or(H_ct_1, H_ct_2).float()
        H_ct = torch.logical_or(H_ct, H_wsi).float()


        x_ = x_ct
        for i in range(len(self.ct_hyconvs)):
            x_ = self.ct_hyconvs[i](x_, H_ct)  # , mask
            if i != len(self.ct_hyconvs) - 1:
                x_ = self.drop_out(x_)

        x = self.drop_out(x_)
        x = self.fc(torch.cat((x, res, x_cli), dim=-1))  # +res

        return x, x

    def train_H(self, x_ct, x_wsi, x_cli=None, train=False):
        H_wsi = self.get_features_H(x_wsi, self.k_nearest_wsi)
        H_ct_1 = self.get_features_H(x_ct[:, :512], self.k_nearest)
        H_ct_2 = self.get_features_H(x_ct[:, 512:], self.k_nearest)
        H_ct = torch.logical_or(H_ct_1, H_ct_2).float()
        H_ct = torch.logical_or(H_ct, H_wsi).float()


        return H_ct



    def test_forward(self, x_ct, train_ct_fts, train_wsi_fts, x_cli=None, train_cli_fts=None,
                     train_risk=None):  # , stage=None, stage_t=None, stage_m=None ,stage_n=None,
        N, C = x_ct.size()

        x_ct = torch.concat((x_ct, train_ct_fts), dim=0)

        res = x_ct  # [:N]

        H_ct_1 = self.get_features_H(x_ct[:, :512], self.k_nearest)
        H_ct_2 = self.get_features_H(x_ct[:, 512:], self.k_nearest)
        H_ct = torch.logical_or(H_ct_1, H_ct_2).float()

        H_wsi_train = self.get_features_H(train_wsi_fts, self.k_nearest_wsi)
        H_wsi = torch.zeros_like(H_ct)
        H_wsi[N:, N:] = H_wsi_train

        H_ct = torch.logical_or(H_ct, H_wsi).float()


        x_ = x_ct

        for i in range(len(self.ct_hyconvs)):
            x_ = self.ct_hyconvs[i](x_, H_ct)  # , mask
            if i != len(self.ct_hyconvs) - 1:
                x_ = self.drop_out(x_)

        x_ = x_[:N]
        x_ft = x_

        x = self.drop_out(x_)
        x = self.fc(torch.cat((x, res[:N], x_cli), dim=-1))  # +res

        return x, x_ft  # self.scale_tensor(x)+self.scale_tensor(risk)

    def get_intra_embedding(self, x_ct):
        for i in range(len(self.ct_linears)):
            x_ct = self.ct_linears[i](x_ct)  # , mask
            x_ct = self.bns[i](x_ct)
        return x_ct

    def get_H(self, score):
        dis_score = torch.abs(score.T - score)
        min_topk, _ = torch.topk(dis_score, self.k_nearest, dim=1, largest=False)
        min_threshold = min_topk[:, -1].unsqueeze(0).repeat(score.shape[0], 1)
        ones = torch.ones_like(dis_score)
        zeros = torch.zeros_like(dis_score)
        H = torch.where(dis_score <= min_threshold, ones, zeros)
        return H

    def get_mask(self, x, risk):
        mask = torch.ones(risk.shape[0], risk.shape[0]).to(x.device)
        mask[(risk.T - risk) == 0] = 0
        return mask.T

    def scale_tensor(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        scaled_tensor = (tensor - min_val) / (max_val - min_val)
        return scaled_tensor

    def get_ft_H(self, fts, full=False):
        if full:
            H, edge_weight = get_full_H(fts, k_threshold=self.k_threshold, k_nearest=self.k_nearest)
            return H, edge_weight
        else:
            return neighbor_distance(fts, self.k_nearest)

    def set_k_nearst(self, k_nearest):
        self.k_nearest = k_nearest
