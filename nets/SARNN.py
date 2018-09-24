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
                 nstack,
                 N,
                 M,
                 T,
                 depth):
        super(EncoderSARNN, self).__init__(idim, cdim, N, M, T)

        self.nstack = nstack
        self.depth = depth
        self.mem_bias = nn.Parameter(torch.Tensor(nstack, M),
                                     requires_grad=False)
        stdev = 1 / (np.sqrt(nstack * N + M))
        nn.init.uniform(self.mem_bias, -stdev, stdev)

        self.stack2r = nn.Linear(nstack * M * depth, M)

        # shift matrix for stack
        W_up, W_down = utils.shift_matrix(N)
        self.W_up = nn.Parameter(torch.Tensor(W_up), requires_grad=False)
        self.W_pop = self.W_up

        self.W_down = nn.Parameter(torch.Tensor(W_down), requires_grad=False)
        self.W_push = self.W_down

        self.wlstm = nn.LSTM(cdim + M, cdim)
        self.wstate_init = nn.Parameter(torch.zeros(cdim),
                                       requires_grad=False)
        self.analysis_ctrl = nn.Linear(cdim, 3 + nstack * M)
        self.write_ctrl = nn.LSTM(M, cdim)

    def update_stack(self, stack,
                     p_push, p_pop,
                     p_noop, push_vals):

        # stack: (bsz, nstack, ssz, sdim)
        # p_push, p_pop, p_noop: (bsz, nstack, 1, 1)
        # push_vals: (bsz, nstack, sdim)
        p_push = p_push.unsqueeze(-1).unsqueeze(-1)
        p_pop = p_pop.unsqueeze(-1).unsqueeze(-1)
        p_noop = p_noop.unsqueeze(-1).unsqueeze(-1)

        stack_push = self.W_push.matmul(stack)
        stack_push[:, :, 0, :] += push_vals

        stack_pop = self.W_pop.matmul(stack)
        # fill the stack with empty elements
        stack_pop[:, :, self.N - 1:, :] += \
            self.mem_bias.unsqueeze(1).unsqueeze(0)

        stack  = p_push * stack_push + p_pop * stack_pop + p_noop * stack
        return stack

    def read(self, controller_outp):
        bsz = controller_outp.shape[0]
        tops = self.stack[:, :, :self.depth, :].\
            contiguous(). \
            view(bsz, -1)
        return self.stack2r(tops)

    def write(self, controller_outp, r):
        # r: (bsz, M)
        # controller_outp: (bsz, cdim)
        # ctrl_info: (bsz, 3 + nstack * M)
        o = torch.cat([controller_outp, r], dim=1)
        wout, self.whid = self.wlstm(o.unsqueeze(0), self.whid)

        ctrl_info = self.analysis_ctrl(wout.squeeze(0))
        p_push, p_pop, p_noop = \
            F.softmax(ctrl_info[:, :3], dim=-1)\
            .chunk(3, dim=-1)
        push_vals = ctrl_info[:, 3:].view(-1, self.nstack, self.M)
        self.stack = \
            self.update_stack(self.stack, p_push, p_pop, p_noop, push_vals)

    def reset_read(self, bsz):
        pass

    def reset_write(self, bsz):
        self.whid = (self.wstate_init.expand(1, bsz, self.cdim).contiguous(),
                     self.wstate_init.expand(1, bsz, self.cdim).contiguous())

    def reset_mem(self, bsz):
        self.stack = self.mem_bias.unsqueeze(1). \
            unsqueeze(0). \
            expand(bsz,
                   self.nstack,
                   self.N,
                   self.M)
