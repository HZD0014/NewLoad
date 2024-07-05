import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.MPatch_layers import PatchBlock

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embed_size = 128 #embed_size
        self.hidden_size = 256 #hidden_size

        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        self.top_k = configs.top_k

        self.model = nn.ModuleList([PatchBlock(configs, patch_len=patch) for patch in configs.patch_list])

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)      # [B,L,C] → [B,C,L]
        res = []
        for layer in self.model:
            out = layer(x)
            res.append(out)


        return x
