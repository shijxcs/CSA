import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight.T, dim=0))
        return out

class BCLHead(nn.Module):
    def __init__(self, dim_in, num_classes=1000, head='mlp', use_norm=False, hidden_dim=512, out_dim=128):
        super(BCLHead, self).__init__()
        if head == 'mlp':
            self.head = nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
                                      nn.Linear(hidden_dim, out_dim))
        else:
            raise NotImplementedError(
                'head not supported'
            )
        if use_norm:
            self.fc = NormedLinear(dim_in, num_classes)
        else:
            self.fc = nn.Linear(dim_in, num_classes)
        self.head_fc = nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
                                   nn.Linear(hidden_dim, out_dim))

    def forward(self, feat):
        feat_mlp = F.normalize(self.head(feat), dim=1)
        logits = self.fc(feat)
        centers_logits = F.normalize(self.head_fc(self.fc.weight), dim=1)
        return feat_mlp, logits, centers_logits

