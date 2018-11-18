import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.init import orthogonal_
import torch

class EncoderVecAVE(nn.Module):

    def __init__(self, idim, cdim, drop):
        super(EncoderVecAVE, self).__init__()
        self.idim = idim
        self.odim = cdim
        self.cdim = cdim
        self.dropout = nn.Dropout(drop)
        self.emb2inp = nn.Linear(idim, cdim)

    def forward(self, **input):
        embs = input['embs']
        lens = input['lens']
        mask = input['mask']
        embs = embs * mask.float().unsqueeze(-1)
        embs = self.dropout(embs)
        # inps = self.emb2inp(embs)
        inps = embs
        seq_len, bsz, edim = inps.shape
        res = torch.sum(inps, dim=0) / lens.float().unsqueeze(-1)
        # res = torch.max(embs, dim=0)[0]
        os = res.unsqueeze(0).expand(seq_len, bsz, edim)

        return os
