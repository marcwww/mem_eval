import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from .MANN import MANNBaseEncoder
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderALSTM(MANNBaseEncoder):
    def __init__(self,
                 idim,
                 cdim,
                 N,
                 M,
                 idrop,
                 odrop):
        super(EncoderALSTM, self).__init__(idim, cdim, N, M, idrop, odrop)
        self.atten = utils.Attention(cdim, M)
        self.zero = nn.Parameter(torch.zeros(M), requires_grad=False)

    def read(self, controller_outp):
        bsz = controller_outp.shape[0]
        if len(self.mem) > 0:
        # mem: (seq_len, bsz, cdim)
            mem = torch.cat(self.mem, dim=1)
            c = self.atten(controller_outp, mem)
        else:
            c = self.zero.expand(bsz, self.M)

        return c

    def write(self, controller_outp, r):
        self.mem.append(controller_outp.unsqueeze(1))

    def reset_read(self, bsz):
        pass

    def reset_write(self, bsz):
        pass

    def reset_mem(self, bsz):
        self.mem = []
