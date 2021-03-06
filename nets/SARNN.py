import numpy as np
from torch import nn
import torch
from macros import *
from torch.nn import functional as F
import utils
from .MANN import MANNBaseEncoder
from torch.nn.utils.rnn import pack_padded_sequence as pack, \
    pad_packed_sequence as unpack
import json


class EncoderSARNN(MANNBaseEncoder):
    def __init__(self,
                 idim,
                 cdim,
                 N,
                 M,
                 drop,
                 read_first):
        super(EncoderSARNN, self).__init__(idim, cdim, N, M, drop, read_first=read_first)

        self.mem_bias = nn.Parameter(torch.zeros(M),
                                     requires_grad=False)
        self.pop_kernel = nn.Parameter(torch.eye(N + 1).
                                       view(N + 1, 1, N + 1, 1),
                                       requires_grad=False)
        self.zero = nn.Parameter(torch.zeros(1, 1, 1),
                                 requires_grad=False)
        self.policy = nn.Sequential(nn.Linear(idim + M * 2, 3), nn.LogSoftmax(dim=-1))
        self.hid2pushed = nn.Linear(cdim, M) if cdim != M else lambda x: x

    def _inf_bias_policy(self, inp, weight, bias):
        bias = bias.clone()
        bias[0] = -1e20
        logits = F.linear(inp, weight, bias)
        return F.log_softmax(logits, dim=-1).chunk(dim=-1, chunks=3)

    def update_stack(self, inp, hid):
        # inp: (bsz, edim)
        bsz, edim = inp.shape
        # self.mem: (bsz, N, M)
        mem_padded = F.pad(self.mem.unsqueeze(1), [0, 0, 0, self.N], 'constant', 0)

        # m_pop: (bsz, N+1, N, M)
        m_pop = F.conv2d(mem_padded, self.pop_kernel)
        pin_stack = torch.cat([m_pop[:, :, 0], m_pop[:, :, 1]], dim=2)  # pin_stack: (bsz, N+1, M*2)
        pin_inp = inp.unsqueeze(1).expand(bsz, self.N+1, edim)  # pin_inp: (bsz, N+1, edim)
        pin = torch.cat([pin_inp, pin_stack], dim=-1)  # pin: (bsz, N+1, edim + M*2)
        lp_pop, lp_stay, lp_push = self.policy(pin[:, :-1]).chunk(dim=-1, chunks=3)  # lp_xxx: (bsz, N, 1)
        _, lp_stay_tail, lp_push_tail = self._inf_bias_policy(pin[:, -1:], self.policy[0].weight, self.policy[0].bias)
        # lp_xxx_tail: (bsz, 1, 1)
        lp_stay = torch.cat([lp_stay, lp_stay_tail], dim=1)  # to (bsz, N+1, 1)
        lp_push = torch.cat([lp_push, lp_push_tail], dim=1)  # to (bsz, N+1, 1)
        lp_pop_revised = torch.cat([self.zero.expand(bsz, 1, 1),
                                    lp_pop],
                                   dim=1)  # lp_pop_revised: (bsz, N+1, 1)
        #  this 'revised' corresponding to the original paper for the base cases
        lp_pop = lp_pop_revised.cumsum(dim=1)
        p_stay = (lp_pop + lp_stay).exp().unsqueeze(-1)  # p_stay: (bsz, N+1, 1, 1)
        p_push = (lp_pop + lp_push).exp().unsqueeze(-1)  # p_push: (bsz, N+1, 1, 1)

        # assert ((p_stay + p_push).sum(dim=1).sum() - bsz) < 1e-4

        # pushed: (bsz, M)
        pushed = self.hid2pushed(hid)
        # pushed: (bsz, N+1, 1, M)
        pushed_expanded = pushed.unsqueeze(1).unsqueeze(1).expand(bsz, self.N + 1, 1, self.M)
        m_push = torch.cat([pushed_expanded, m_pop[:, :, :-1]], dim=2)

        mem_new_stay = (m_pop * p_stay).sum(dim=1)
        mem_new_push = (m_push * p_push).sum(dim=1)

        # mem_new = (m_stay * p_stay).sum(dim=1) + (m_push * p_push).sum(dim=1)
        mem_new = mem_new_stay + mem_new_push
        self.mem = mem_new
        return mem_new_stay, mem_new_push, m_pop, m_push, pushed

    def read(self, controller_outp):
        # r = torch.cat([self.mem[:, 0], self.mem[:, 1]], dim=1)
        r = self.mem[:, 0]
        return r

    def write(self, controller_outp, input):
        # r: (bsz, M)
        # controller_outp: (bsz, cdim)
        # ctrl_info: (bsz, 3 + nstack * M)

        hid = controller_outp
        mem_stay, mem_push, m_stay, m_push, pushed = \
            self.update_stack(input, hid)

        if 'analysis_mode' in dir(self) and self.analysis_mode:
            assert 'fsarnn' in dir(self)
            assert policy.shape[0] == 1

            val, pos = torch.topk(policy[0], k=1)
            pos = pos.item()
            val = val.item()
            line = {'type': 'actions',
                    'all': policy[0].cpu().numpy().tolist(),
                    'max_pos': pos,
                    'max_val': val,
                    'mem': self.mem[0].cpu().numpy().tolist()}

            # line['mem_stay'] = mem_stay[0].cpu().numpy().tolist()
            # line['mem_push'] = mem_push[0].cpu().numpy().tolist()
            # line['hid'] = hid[0].cpu().numpy().tolist()
            # line['pushed'] = pushed[0].cpu().numpy().tolist()
            # for i, m_push_i in enumerate(m_push[0]):
            #     line['mem_push_%d' % i] = m_push_i.cpu().numpy().tolist()
            # for i, m_stay_i in enumerate(m_stay[0]):
            #     line['mem_stay_%d' % i] = m_stay_i.cpu().numpy().tolist()

            line = json.dumps(line)
            if pos <= 5:
                # print(line)
                print(line, file=self.fsarnn)
                # print('stay after pop %d times with confidence %.3f' % (pos, val))
                # print('stay after pop %d times with confidence %.3f' % (pos, val), file=self.fanalysis)
            else:
                # print(line)
                print(line, file=self.fsarnn)
                # print('push after pop %d times with confidence %.3f' % (pos - 6, val))
                # print('push after pop %d times with confidence %.3f' % (pos-6, val), file=self.fanalysis)

    def reset_read(self, bsz):
        pass

    def reset_write(self, bsz):
        pass

    def reset_mem(self, bsz):
        self.mem = self.mem_bias.expand(bsz, self.N, self.M)
