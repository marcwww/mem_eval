import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from .MANN import MANNBaseEncoder
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack

class EncoderSARNN(MANNBaseEncoder):
    def __init__(self,
                 idim,
                 cdim,
                 N,
                 M,
                 dropout):
        super(EncoderSARNN, self).__init__(idim, cdim, N, M, dropout)

        self.mem_bias = nn.Parameter(torch.zeros(M),
                                     requires_grad=False)
        self.update_kernel = nn.Parameter(torch.eye(N + 1).
            view(N + 1, 1, N + 1, 1),
            requires_grad=False)

        self.policy_stack = nn.Conv1d(M, 2, kernel_size=2)
        # 2 for push and stay
        self.policy_input = nn.Linear(idim, 2 * (N + 1))
        self.hid2pushed = nn.Linear(cdim, M)

    def policy(self, input):
        bsz = input.shape[0]
        mem_padded = F.pad(self.mem.transpose(1, 2),
                           (0, 2), 'constant', 0)
        policy_stack = self.policy_stack(mem_padded).view(bsz, -1)
        policy_input = self.policy_input(input)

        return F.softmax(policy_stack + policy_input, dim=1)

    def update_stack(self,
                     p_push, p_stay, hid):
        bsz = hid.shape[0]

        p_stay = p_stay.unsqueeze(-1).unsqueeze(-1)
        p_push = p_push.unsqueeze(-1).unsqueeze(-1)

        mem_padded = F.pad(self.mem.unsqueeze(1),
                           (0, 0, 0, self.N),
                           'constant', 0)

        # m_stay: (bsz, N+1, N, M)
        m_stay = F.conv2d(mem_padded, self.update_kernel)

        # pushed: (bsz, M)
        pushed = self.hid2pushed(hid)
        # pushed: (bsz, N+1, 1, M)
        pushed = pushed.unsqueeze(1).unsqueeze(1).expand(bsz, self.N + 1, 1, self.M)
        m_push = torch.cat([pushed, m_stay[:, :, :-1]], dim=2)

        mem_new = (m_stay * p_stay).sum(dim=1) + (m_push * p_push).sum(dim=1)
        self.mem = mem_new

    def read(self, controller_outp):
        r = self.mem[:, 0]
        return r

    def write(self, controller_outp, input):
        # r: (bsz, M)
        # controller_outp: (bsz, cdim)
        # ctrl_info: (bsz, 3 + nstack * M)
        hid = controller_outp
        policy = self.policy(input)
        p_stay, p_push = torch.chunk(policy, 2, dim=1)
        self.update_stack(p_stay, p_push, hid)

    def reset_read(self, bsz):
        pass

    def reset_write(self, bsz):
        pass

    def reset_mem(self, bsz):
        self.mem = self.mem_bias.expand(bsz, self.N, self.M)
