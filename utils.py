import numpy as np
from torch.nn.init import xavier_uniform_
from torch.nn.init import kaiming_normal_
import torch
from torch import nn
from torch.nn import functional as F
import logging
import random
import time
from torch.autograd import Variable
from macros import *

LOGGER = logging.getLogger(__name__)

def modulo_convolve(w, s):
    # w: (bsz, N)
    # s: (bsz, 3)
    bsz, ksz = s.shape
    assert ksz == 3

    # t: (1, bsz, 1+N+1)
    t = torch.cat([w[:,-1:], w, w[:,:1]], dim=-1).\
        unsqueeze(0)
    device = s.device
    kernel = torch.zeros(bsz, bsz, ksz).to(device)
    kernel[range(bsz), range(bsz), :] += s
    # c: (bsz, N)
    c = F.conv1d(t, kernel).squeeze(0)
    return c

def split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

def one_hot_matrix(stoi, device, edim):

    assert len(stoi) <= edim, \
        'embedding dimension must be larger than voc_size'

    voc_size = len(stoi)
    res = torch.zeros(voc_size,
                      edim,
                      requires_grad=False)
    for i in range(voc_size):
        res[i][i] = 1

    return res.to(device)

def shift_matrix(n):
    W_up = np.eye(n)
    for i in range(n-1):
        W_up[i,:] = W_up[i+1,:]
    W_up[n-1,:] *= 0
    W_down = np.eye(n)
    for i in range(n-1,0,-1):
        W_down[i,:] = W_down[i-1,:]
    W_down[0,:] *= 0
    return W_up,W_down

def avg_vector(i, n):
    V = np.zeros(n)
    V[:i+1] = 1/(i+1)
    return V

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            # xavier_uniform_(p)
            kaiming_normal_(p)

class LayerNormalization(nn.Module):
    # From: https://discuss.pytorch.org/t/lstm-with-layer-normalization/2150

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z)
        sigma = torch.std(z)

        ln_out = (z - mu) / (sigma + self.eps)

        ln_out = ln_out * self.a2 + self.b2
        return ln_out

def progress_bar(percent, loss, epoch):
    """Prints the progress until the next report."""

    fill = int(percent * 40)
    str_disp = "\r[%s%s]: %.2f/epoch %d" % ('=' * fill,
                                         ' ' * (40 - fill),
                                         percent,
                                         epoch)
    for k, v in loss.items():
        str_disp += ' (%s:%.4f)' % (k, v)

    print(str_disp, end='')

def seq_lens(seq, padding_idx):
    mask = seq.data.eq(padding_idx)
    len_total, bsz = seq.shape
    lens = len_total - mask.sum(dim=0)
    return lens

def to_tree_sd(sd_lst, node_lst):
    if len(sd_lst) == 0:
        node = node_lst[0]
    else:
        i = np.argmax(sd_lst)
        child_l = to_tree_sd(sd_lst[:i], node_lst[:i+1])
        child_r = to_tree_sd(sd_lst[i+1:], node_lst[i+1:])
        node = (child_l, child_r)

    return node

class Attention(nn.Module):
    def __init__(self, cdim, odim):
        super(Attention, self).__init__()
        self.c2r = nn.Linear(cdim, odim)

    def forward(self, h, mem):
        # h: (bsz, hdim)
        # h_current: (bsz, 1, 1, hdim)
        h_current = h.unsqueeze(1).unsqueeze(1)
        # mem: (bsz, len_total, hdim, 1)
        mem = mem.unsqueeze(-1)
        # a: (bsz, len_total, 1, 1)
        a = h_current.matmul(mem)
        a = F.softmax(a, dim=1)
        # c: (bsz, len_total, hdim, 1)
        c = a * mem
        # c: (bsz, hdim)
        c = c.sum(1).squeeze(-1)
        r = self.c2r(c)
        return r, a[:,:,0,0]

def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

def gumbel_softmax_sample(logits, tau, hard, eps=1e-10):

    shape = logits.size()
    assert len(shape) == 2
    y_soft = F._gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        bsz, N = y_soft.shape
        k = []
        for b in range(bsz):
            idx = np.random.choice(N, p=y_soft[b].data.cpu().numpy())
            k.append(idx)
        k = np.array(k).reshape(-1, 1)
        k = y_soft.new_tensor(k, dtype=torch.int64)

        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y

def gumbel_sigmoid_sample(logit, tau, hard):

    shape = logit.shape
    assert (len(shape) == 2 and shape[-1] == 1) \
            or len(shape) == 1

    if len(shape) == 1:
        logit = logit.unsqueeze(-1)
        shape = logit.shape

    zero = logit.new_zeros(*shape)
    res = torch.cat([logit, zero], dim=-1)
    res = gumbel_softmax_sample(res, tau=tau, hard=hard)

    return res[:, 0]

def gumbel_sigmoid_max(logit, tau, hard):

    shape = logit.shape
    assert (len(shape) == 2 and shape[-1] == 1) \
            or len(shape) == 1

    if len(shape) == 1:
        logit = logit.unsqueeze(-1)
        shape = logit.shape

    zero = logit.new_zeros(*shape)
    res = torch.cat([logit, zero], dim=-1)
    res = F.gumbel_softmax(res, tau=tau, hard=hard)

    return res[:, 0]

def param_str(opt):
    res_str = {}
    for attr in dir(opt):
        if attr[0] != '_':
            res_str[attr] = getattr(opt, attr)
    return res_str

def time_int():
    return int(time.time())


class fixMaskDropout(nn.Module):
    def __init__(self, dropout=0.5):
        super(fixMaskDropout, self).__init__()
        self.dropout = dropout
        self.mask = None

    def forward(self, draw_mask, input):
        if self.training == False:
            return input
        if self.mask is None or draw_mask == True:
            self.mask = input.data.new().resize_(input.size()).bernoulli_(1 - self.dropout) / (1 - self.dropout)
        mask = Variable(self.mask)
        masked_input = mask * input
        return masked_input

class fixMaskEmbeddedDropout(nn.Module):
    def __init__(self, embed, dropout=0.5):
        super(fixMaskEmbeddedDropout, self).__init__()
        self.dropout = dropout
        self.e = embed
        w = getattr(self.e, 'weight')
        del self.e._parameters['weight']
        self.e.register_parameter('weight_raw', nn.Parameter(w.data))

    def _setweights(self):
        raw_w = getattr(self.e, 'weight_raw')
        if self.training:
            mask = raw_w.data.new().resize_((raw_w.size(0), 1)).bernoulli_(1 - self.dropout).expand_as(raw_w) / (
                        1 - self.dropout)
            w = Variable(mask) * raw_w
            setattr(self.e, 'weight', w)
        else:
            setattr(self.e, 'weight', Variable(raw_w.data))

    def forward(self, draw_mask, *args):
        if draw_mask or self.training == False:
            self._setweights()
        return self.e.forward(*args)

class analy(object):

    def __init__(self, model, fnames_dict):
        self.model = model
        self.fnames_dict = fnames_dict

    def __enter__(self):
        self.model.analysis_mode = True
        for name in self.fnames_dict:
            setattr(self.model, name, open(self.fnames_dict[name], 'w'))

    def __exit__(self, *args):
        self.model.analysis_mode = False
        for name in self.fnames_dict:
            f = getattr(self.model, name)
            f.close()

if __name__ == '__main__':
    up, down = shift_matrix(3)
    x = np.array([[0,1,2]]).transpose()
    print(x)
    print(up.dot(x))
    print(down)