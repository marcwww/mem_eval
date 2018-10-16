import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class MANNBaseEncoder(nn.Module):

    def __init__(self, idim, cdim, N, M, dropout):
        super(MANNBaseEncoder, self).__init__()
        self.idim = idim
        self.odim = cdim + M
        self.cdim = cdim
        self.N = N
        self.M = M
        self.controller = nn.LSTM(idim + M, cdim)
        self.dropout = nn.Dropout(dropout)
        self._reset_controller()

        self.h0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)
        self.r0 = nn.Parameter(torch.randn(1, M) * 0.02, requires_grad=False)

    def _reset_controller(self):
        for p in self.controller.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.idim + self.M + self.cdim))
                nn.init.uniform(p, -stdev, stdev)

    def reset_read(self, bsz):
        raise NotImplementedError

    def reset_write(self, bsz):
        raise NotImplementedError

    def reset_mem(self, bsz):
        raise NotImplementedError

    def read(self, controller_outp):
        raise NotImplementedError

    def write(self, controller_outp, input):
        raise NotImplementedError

    def forward(self, **input):
        embs = input['embs']
        embs = self.dropout(embs)
        bsz = embs.shape[1]

        self.reset_read(bsz)
        self.reset_write(bsz)
        self.reset_mem(bsz)

        h = self.h0.expand(1, bsz, self.cdim).contiguous()
        c = self.c0.expand(1, bsz, self.cdim).contiguous()
        r = self.r0.expand(bsz, self.M).contiguous()

        hs = []
        cs = []
        os = []
        for emb in embs:
            controller_inp = torch.cat([emb, r], dim=1).unsqueeze(0)
            controller_outp, (h, c) = self.controller(controller_inp, (h, c))
            controller_outp = controller_outp.squeeze(0)

            self.write(controller_outp, emb)
            r = self.read(controller_outp)
            o = torch.cat([controller_outp, r], dim=1)
            o = self.dropout(o)

            hs.append(h)
            cs.append(c)
            os.append(o.unsqueeze(0))

        os = torch.cat(os, dim=0)

        return os




