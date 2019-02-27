import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.init import orthogonal_
import torch

class EncoderTOPNN(nn.Module):

    def __init__(self, idim, cdim, drop):
        super(EncoderTOPNN, self).__init__()
        self.idim = idim
        self.odim = cdim
        self.cdim = cdim
        self.pos_embedding = nn.Embedding(1000, idim)
        # self.pos_embedding.weight = orthogonal_(self.pos_embedding.weight)
        self.pos_embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(drop)

        # inp: (bsz, 1 , edim, edim)
        # self.trans = nn.Sequential(nn.Dropout(drop),
        #                            nn.Conv2d(1, 10, 5, padding=2),
        #                            nn.ReLU(),
        #                            nn.Dropout(drop),
        #                            nn.Conv2d(10, 1, 5, padding=2))
        # self.trans = nn.Sequential(nn.Dropout(drop),
        #                            nn.Conv1d(idim, idim, 21, padding=10),
        #                            nn.ReLU(),
        #                            nn.Dropout(drop),
        #                            nn.Conv1d(idim, idim, 21, padding=10))

        self.trans = nn.Sequential(nn.Dropout(drop),
                                   nn.Linear(idim * idim, idim * idim),
                                   nn.ReLU(),
                                   nn.Dropout(drop),
                                   nn.Linear(idim * idim, idim * idim))

    def forward(self, **input):
        embs = input['embs']
        embs = self.dropout(embs)
        seq_len, bsz, edim = embs.shape
        # pos = embs.new_tensor(range(seq_len)).unsqueeze(-1).expand(seq_len, bsz).long()
        pos = embs.new_tensor(range(seq_len)).unsqueeze(-1).long()
        pemb = self.pos_embedding(pos) # 1, bsz, edim
        pembs = pemb
        # pembs = pemb.expand(seq_len, bsz, edim)
        # pembs = self.pos_embedding(pos) # seq_len, bsz, edim
        # e * p^T, columns are about scattered words
        ops = embs.unsqueeze(-1).matmul(pembs.unsqueeze(-2)) # seq_len, bsz, edim, edim
        S = ops.sum(0) # bsz, edim, edim
        # S = S.transpose(1, 2)
        S = self.trans(S.view(bsz, -1)).view(bsz, edim, edim)
        # S = self.trans(S) # bsz, edim, edim
        # S = self.trans(S.unsqueeze(1)).squeeze(1)  # bsz, edim, edim
        # S = S.transpose(1, 2)
        os = S.matmul(pembs.unsqueeze(-1)).squeeze(-1)

        return os
