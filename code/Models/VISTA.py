import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.model_utils import TemporalFilter, BrainDisentanglementMachine
from einops.layers.torch import Rearrange
from torch import Tensor


class VISTA(nn.Module):
    def __init__(
            self,
            time_length=250,
            n_channel=63,
            hidden_dim=250,
            timePatch_size=10,
            filter_time_length=5,
            n_filters=1,
            att_layers=1,
            dropout_rate=0.5,
            device='cuda:0'
    ):
        super(VISTA, self).__init__()
        self.T = time_length
        self.C = n_channel

        self.filter = TemporalFilter(num_input=1,
                                     n_filters=n_filters,
                                     filter_time_length=filter_time_length,
                                     dropout_rate=dropout_rate,
                                     device=device, )  # 小感受野的短波滤波器

        self.disentanglementer = BrainDisentanglementMachine(n_channel=n_channel,
                                                             time_length=time_length,
                                                             hidden_dim=hidden_dim,
                                                             timePatch_size=timePatch_size,
                                                             att_layers=att_layers,
                                                             dropout_rate=dropout_rate,
                                                             device=device)

        self.visual_STCConv = STC(n_channel, dropout_rate)
        self.semantic_STCConv = STC(n_channel, dropout_rate)

        proj_dim = 768
        f_size = 1440
        self.visual_project = nn.Sequential(nn.Linear(f_size, proj_dim),
                                            ResidualAdd(nn.Sequential(
                                                nn.GELU(),
                                                nn.Linear(proj_dim, proj_dim),
                                                nn.Dropout(dropout_rate),
                                            )),
                                            nn.LayerNorm(proj_dim),
                                            )
        self.semantic_project = nn.Sequential(nn.Linear(f_size, proj_dim),
                                              ResidualAdd(nn.Sequential(
                                                  nn.GELU(),
                                                  nn.Linear(proj_dim, proj_dim),
                                                  nn.Dropout(dropout_rate),
                                              )),
                                              nn.LayerNorm(proj_dim),
                                              )

    def forward(self, x):
        # embedding = self.filter(x)
        embedding = x

        v_x, s_x, [visual_binary, semantic_binary], [visualBrain, semanticBrain], loss_time = self.disentanglementer(
            embedding)

        v_x = v_x.unsqueeze(1)
        v_x = self.visual_STCConv(v_x)
        v_x = v_x.contiguous().view(v_x.size(0), -1)
        v_x = self.visual_project(v_x)

        s_x = s_x.unsqueeze(1)
        s_x = self.semantic_STCConv(s_x)
        s_x = s_x.contiguous().view(s_x.size(0), -1)
        s_x = self.semantic_project(s_x)

        return [v_x, s_x], [loss_time], [visual_binary, semantic_binary], [visualBrain, semanticBrain]


class STC(nn.Module):
    def __init__(self, n_channel=63, dropout_rate=0.5):
        super(STC, self).__init__()
        con_size = 40
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, con_size, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(con_size),
            nn.ELU(),
            nn.Conv2d(con_size, con_size, (n_channel, 1), (1, 1)),
            nn.BatchNorm2d(con_size),
            nn.ELU(),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(con_size, con_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, input_dim=768):
        super(AdaptiveFeatureFusion, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim * input_dim, 2)
    def forward(self, v_x, s_x):
        batch_size, a = v_x.size()
        M = torch.matmul(v_x.unsqueeze(2), s_x.unsqueeze(1))  # (batch_size, a, a)

        M_flat = M.view(batch_size, -1)  # (batch_size, a * a)
        weights = F.softmax(self.fc(M_flat), dim=-1)  # (batch_size, 2)
        a, b = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)  # (batch_size, 1)
        fused_x = a * v_x + b * s_x  # (batch_size, a)

        return fused_x, (a, b)


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=512, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
