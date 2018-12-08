import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils
from collections import defaultdict
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score
import json

class Example(object):

    def __init__(self, expr, ds, h):
        self.expr = self.tokenizer_expr(expr)
        self.ds = self.tokenizer_ds(ds)
        self.h = int(h)

    def tokenizer_expr(self, seq):
        return seq.split()

    def tokenizer_ds(self, ds):
        return [int(d) for d in ds.split()]

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            expr, ds, h = \
                line.strip().split('\t')
            examples.append(Example(expr, ds, h))

    return examples

def build_iters(**param):

    ftrain = param['ftrain']
    fvalid = param['fvalid']
    if 'ftest' in param:
        ftest = param['ftest']

    if 'fanaly' in param:
        fanaly = param['fanaly']

    bsz = param['bsz']
    device = param['device']

    examples_train = load_examples(ftrain)

    EXPR = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=None)
    DS = torchtext.data.Field(sequential=True, use_vocab=False, pad_token=PAD_DS)
    H = torchtext.data.Field(sequential=False, use_vocab=False)

    train = Dataset(examples_train, fields=[('expr', EXPR),
                                            ('ds', DS),
                                            ('h', H)])
    EXPR.build_vocab(train)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('expr', EXPR),
                                            ('ds', DS),
                                            ('h', H)])
    if 'ftest' in param:
        examples_test = load_examples(ftest)
        test = Dataset(examples_test, fields=[('expr', EXPR),
                                                ('ds', DS),
                                                ('h', H)])

    analy = None
    if 'fanaly' in param:
        examples_analy = load_examples(fanaly)
        analy = Dataset(examples_analy, fields=[('expr', EXPR),
                                                ('ds', DS),
                                                ('h', H)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.expr),
                                         sort_within_batch=True,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.expr),
                                         sort_within_batch=True,
                                         device=device)
    analy_iter = None
    if 'fanaly' in param:
        analy_iter = torchtext.data.Iterator(analy, batch_size=bsz,
                                             sort=False, repeat=False,
                                             sort_key=lambda x: len(x.expr),
                                             sort_within_batch=True,
                                             device=device)
    test_iter = None
    if 'ftest' in param:
        test_iter = torchtext.data.Iterator(test, batch_size=bsz,
                                             sort=False, repeat=False,
                                             sort_key=lambda x: len(x.expr),
                                             sort_within_batch=True,
                                             device=device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'test_iter': test_iter,
            'analy_iter': analy_iter,
            'SEQ': EXPR,
            'DS': DS,
            'H': H}

def valid_detail(model, itos, valid_iter):

    def num_extrem_vals(ds):
        res = 0
        for i, d in enumerate(ds):
            if i == 0:
                if d > ds[1]:
                    res += 1
            elif i == len(ds) - 1:
                if d > ds[-2]:
                    res += 1
            elif d > ds[i - 1] and d > ds[i + 1]:
                res += 1

        return res

    nt = defaultdict(int)
    nc = defaultdict(int)
    acc = defaultdict(float)
    padding_idx = model.padding_idx
    incorrect_predicts = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            expr = batch.expr
            mask_expr = expr.ne(padding_idx)
            len_expr = mask_expr.sum(0)
            ds_tar = batch.ds
            h = batch.h

            out = model(expr)
            ds_pred = out['ds'].squeeze(-1)
            dstmask = (ds_tar > PAD_DS)
            len_ds = dstmask.sum(0)

            for e, d_pred, d_tar, l_e, l_d, h_b in zip(expr.transpose(0, 1),
                                                  ds_pred.transpose(0, 1),
                                                  ds_tar.transpose(0, 1),
                                                  len_expr, len_ds, h):
                ne = num_extrem_vals(d_tar[:l_d])

                h_b = h_b.item()
                e = list(e[:l_e].cpu().numpy())
                d_pred = list(d_pred[:l_d].cpu().numpy())
                d_tar = list(d_tar[:l_d].cpu().numpy())
                try:
                    tree_pred = utils.to_tree_sd(d_pred, e)
                except:
                    tree_pred = None
                tree_tar = utils.to_tree_sd(d_tar, e)
                if str(tree_pred) == str(tree_tar):
                    nc[h_b] += 1
                else:
                    expr = ' '.join([itos[ch.item()] for ch in e if itos[ch] != PAD])
                    ds_b = ' '.join([str(d) for d in d_tar if d != PAD_DS])
                    incorrect_predicts.append((expr, ds_b, str(h_b)))

                nt[h_b] += 1

    for h_b in nc.keys():
        acc[h_b] = nc[h_b]/nt[h_b]

    acc_total = sum([nc[h] for h in nc.keys()])/sum([nt[h] for h in nt.keys()])

    return acc, nt, incorrect_predicts, acc_total

def valid(model, valid_iter):
    nt = 0
    nc = 0
    padding_idx = model.padding_idx
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            expr = batch.expr
            mask_expr = expr.ne(padding_idx)
            len_expr = mask_expr.sum(0)
            ds_tar = batch.ds

            out = model(expr)
            ds_pred = out['ds'].squeeze(-1)
            dstmask = (ds_tar > PAD_DS)
            len_ds = dstmask.sum(0)

            for e, d_pred, d_tar, l_e, l_d in zip(expr.transpose(0, 1),
                                                  ds_pred.transpose(0, 1),
                                                  ds_tar.transpose(0, 1),
                                                  len_expr, len_ds):
                e = list(e[:l_e].cpu().numpy())
                d_pred = list(d_pred[:l_d].cpu().numpy())
                d_tar = list(d_tar[:l_d].cpu().numpy())
                try:
                    tree_pred = utils.to_tree_sd(d_pred, e)
                except:
                    tree_pred = None
                tree_tar = utils.to_tree_sd(d_tar, e)
                if str(tree_pred) == str(tree_tar):
                    nc += 1
                nt += 1

    return nc / nt

def test_analy(model, itos, analy_iter, enc):
    nt = 0
    nc = 0
    padding_idx = model.padding_idx
    fanalysis = getattr(model.encoder, 'f' + enc)

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(analy_iter):
            expr = batch.expr
            mask_expr = expr.ne(padding_idx)
            len_expr = mask_expr.sum(0)
            ds_tar = batch.ds

            out = model(expr)
            ds_pred = out['ds'].squeeze(-1)
            dstmask = (ds_tar > PAD_DS)
            len_ds = dstmask.sum(0)

            for e, d_pred, d_tar, l_e, l_d in zip(expr.transpose(0, 1),
                                                  ds_pred.transpose(0, 1),
                                                  ds_tar.transpose(0, 1),
                                                  len_expr, len_ds):
                e = list(e[:l_e].cpu().numpy())
                d_pred = list(d_pred[:l_d].cpu().numpy())
                d_tar = list(d_tar[:l_d].cpu().numpy())
                try:
                    tree_pred = utils.to_tree_sd(d_pred, e)
                except:
                    tree_pred = None
                tree_tar = utils.to_tree_sd(d_tar, e)
                if str(tree_pred) == str(tree_tar):
                    nc += 1
                nt += 1

                is_correct = 1 if str(tree_pred) == str(tree_tar) else 0
                expr = [itos[ch.item()] for ch in expr[:, 0]]
                line = {'type': 'input',
                        'idx': i,
                        'expr': expr,
                        'is_correct': is_correct}
                line = json.dumps(line)
                print(line)
                print(line, file=fanalysis)

    return nc / nt

def rankloss(input, target, mask, exp=False):
    # input: (seq_len, bsz)
    # target: (seq_len, bsz)
    # mask: (seq_len, bsz)
    input = input.transpose(0, 1)
    target = target.transpose(0, 1)
    mask = mask.transpose(0, 1)
    diff = input[:, :, None] - input[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = mask[:, :, None] * mask[:, None, :] * target_diff

    if exp:
        loss = torch.exp(F.relu(target_diff - diff)) - 1
    else:
        loss = F.relu(target_diff - diff)
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)

    return loss

def train(model, iters, opt, optim, scheduler):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']

    basename = "{}-{}-{}-{}".format(opt.task,
                                       opt.sub_task,
                                       opt.enc_type,
                                       utils.time_int())
    log_fname = basename + ".json"
    log_path = os.path.join(RES, log_fname)
    with open(log_path, 'w') as f:
        f.write(str(utils.param_str(opt)) + '\n')
    # print(valid(model, valid_iter))

    best_performance = 0
    losses = []
    gnorms = []
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            expr = batch.expr
            ds_tar = batch.ds

            model.train()
            model.zero_grad()
            out = model(expr)
            ds_pred = out['ds'].squeeze(-1)
            dstmask = (ds_tar > PAD_DS).float()

            loss = rankloss(ds_pred, ds_tar, dstmask)
            losses.append(loss.item())

            loss.backward()
            gnorm = clip_grad_norm(model.parameters(), opt.gclip)
            gnorms.append(gnorm)

            optim.step()
            loss = {'loss': loss.item(), 'gnorm': gnorm}

            utils.progress_bar(i / len(train_iter), loss, epoch)

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                loss_ave = np.array(losses).sum() / len(losses)
                gnorm_ave = np.array(gnorms).sum() / len(gnorms)
                losses = []
                gnorms = []
                accurracy = \
                    valid(model, valid_iter)
                log_str = '{\'Epoch\':%d, \'Format\':\'a/l/g\', \'Metrics\':[%.4f, %.4f, %.4f]}' % \
                          (epoch, accurracy, loss_ave, gnorm_ave)
                print(log_str)
                with open(log_path, 'a+') as f:
                    f.write(log_str + '\n')

                scheduler.step(loss_ave)
                for param_group in optim.param_groups:
                    print('learning rate:', param_group['lr'])

                if accurracy > best_performance:
                    best_performance = accurracy
                    model_fname = basename + ".model"
                    save_path = os.path.join(RES, model_fname)
                    print('Saving to ' + save_path)
                    torch.save(model.state_dict(), save_path)

class Model(nn.Module):

    def __init__(self, encoder, embedding, drop):
        super(Model, self).__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.hdim = self.encoder.odim
        self.padding_idx = embedding.padding_idx
        self.num_words = embedding.num_embeddings
        self.edrop = nn.Dropout(drop)
        self.conv1d = nn.Sequential(nn.Dropout(drop),
                                 nn.Conv1d(self.hdim, self.hdim, 2),
                                 nn.ReLU())
        self.to_ds = nn.Sequential(nn.Dropout(drop),
                                 nn.Linear(self.hdim, self.hdim),
                                 nn.ReLU(),
                                 nn.Dropout(drop),
                                 nn.Linear(self.hdim, 1))

    def enc(self, seq):
        mask = seq.data.eq(self.padding_idx)
        len_total, bsz = seq.shape
        lens = len_total - mask.sum(dim=0)

        # inp = self.embedding_drop(True, seq)
        inp = self.embedding(seq)
        os = self.encoder(embs=inp, mask=1-mask, lens = lens)
        # rep = os[lens - 1, range(bsz)]
        return os

    def forward(self, seq):
        os = self.enc(seq)
        os = os.permute(1, 2, 0) # (bsz, hdim, seq_len)
        conv_os = self.conv1d(os)
        conv_os = conv_os.permute(2, 0, 1) # (seq_len, bsz, hdim)
        ds = self.to_ds(conv_os)

        return {'ds': ds}



