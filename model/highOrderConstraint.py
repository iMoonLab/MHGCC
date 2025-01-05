import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import einsum


class Model(nn.Module):
    def __init__(self, noise_level=1.0, tau=0.5):
        super().__init__()
        self.tau = tau
        self.noise_level = noise_level
        self.delta_e_ = None

    def preprocess(self, model, X, H, X_cli=None):
        model.eval()
        pred = model.forward_w_H(X, H, X_cli).softmax(dim=-1).detach()
        entropy_x = -(pred * pred.log()).sum(1, keepdim=True)
        entropy_x[entropy_x.isnan()] = 0
        entropy_e = self.v2e(entropy_x, H, aggr="mean")

        X_noise = X.clone() * (torch.randn_like(X) + 1) * self.noise_level
        pred_ = model.forward_w_H(X_noise, H, X_cli).softmax(dim=-1).detach()
        entropy_x_ = -(pred_ * pred_.log()).sum(1, keepdim=True)
        entropy_x_[entropy_x_.isnan()] = 0
        entropy_e_ = self.v2e(entropy_x_, H, aggr="mean")

        self.delta_e_ = (entropy_e_ - entropy_e).abs()
        self.delta_e_ = 1 - self.delta_e_ / self.delta_e_.max()
        self.delta_e_ = self.delta_e_.squeeze()

    def v2e(self, entropy, H, aggr="mean"):
        De = torch.diag_embed(1.0 / H.sum(0))
        if aggr == "mean":
            HDe = einsum('ve,ek->vk', H, De)
            res = einsum('vc,ve->ec', entropy, HDe)
            return res
        else:
            raise NotImplementedError

    def forward(self, pred_s, pred_t, H):
        pred_s, pred_t = F.softmax(pred_s, dim=1), F.softmax(pred_t, dim=1)
        e_mask = torch.bernoulli(self.delta_e_).bool()
        pred_s_e = self.v2e(pred_s, H, aggr="mean")
        pred_s_e = pred_s_e[e_mask]
        pred_t_e = self.v2e(pred_t, H, aggr="mean")
        pred_t_e = pred_t_e[e_mask]
        loss = F.kl_div(torch.log(pred_s_e / self.tau + 1e-8), torch.log(pred_t_e / self.tau + 1e-8), reduction="batchmean", log_target=True)
        return loss