import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from .MANN import MANNBaseEncoder
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderLSTM(MANNBaseEncoder):
    def __init__(self,
                 idim,
                 cdim,
                 N,
                 M,
                 dropout):
        super(EncoderLSTM, self).__init__(idim, cdim, N, M, dropout)
        self.zero = nn.Parameter(torch.zeros(M), requires_grad=False)

    def read(self, controller_outp):
        bsz = controller_outp.shape[0]
        return self.zero.expand(bsz, self.M)

    def write(self, controller_outp, r):
        pass

    def reset_read(self, bsz):
        pass

    def reset_write(self, bsz):
        pass

    def reset_mem(self, bsz):
        pass
