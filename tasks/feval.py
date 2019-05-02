import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import utils
from collections import defaultdict
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score
import json
import tqdm


class Example(object):

    def __init__(self, expr, ds, h, val):
        self.expr = self.tokenizer_expr(expr)
        self.ds = self.tokenizer_ds(ds)
        self.h = int(h)
        self.val = int(val)

    def tokenizer_expr(self, seq):
        return seq.split()

    def tokenizer_ds(self, ds):
        return [int(d) for d in ds.split()]


def load_examples(fname, seq_len_max=None):
    examples = []
    ndiscard = 0
    lens = []

    with open(fname, 'r') as f:
        for line in f:
            expr, ds, h, val = \
                line.strip().split('\t')
            # examples.append(Example(expr, ds, h, val))

            if seq_len_max == None or len(expr.split()) <= seq_len_max:
                examples.append(Example(expr, ds, h, val))
                lens.append(len(expr))
            else:
                ndiscard += 1
        print('Discarding %d samples' % ndiscard)

    len_ave = np.mean(lens)
    return examples, len_ave


def build_iters(**param):
    ftrain = param['ftrain']
    fvalid = param['fvalid']
    seq_len_max = param['seq_len_max']
    ftest = None
    fanaly = None
    if 'ftest' in param:
        ftest = param['ftest']

    if 'fanaly' in param:
        fanaly = param['fanaly']

    bsz = param['bsz']
    device = param['device']
    device = torch.device(device if device != -1 else 'cpu')

    examples_train, len_ave = load_examples(ftrain, seq_len_max)

    EXPR = torchtext.data.Field(sequential=True, use_vocab=True,
                                pad_token=PAD,
                                unk_token=UNK,
                                eos_token=None)
    DS = torchtext.data.Field(sequential=True, use_vocab=False, pad_token=PAD_DS)
    H = torchtext.data.Field(sequential=False, use_vocab=False)
    VAL = torchtext.data.Field(sequential=False, use_vocab=False)
    train = Dataset(examples_train, fields=[('expr', EXPR),
                                            ('ds', DS),
                                            ('h', H),
                                            ('val', VAL)])
    EXPR.build_vocab(train)
    examples_valid, _ = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('expr', EXPR),
                                            ('ds', DS),
                                            ('h', H),
                                            ('val', VAL)])

    def batch_size_fn(new_example, current_count, ebsz):
        return ebsz + (len(new_example.expr) / len_ave) ** 0.5
        # return ebsz + (len(new_example.expr) / len_ave)
        # return ebsz + len(new_example.expr)
        # return current_count

    test = None
    if 'ftest' in param:
        examples_test, _ = load_examples(ftest)
        test = Dataset(examples_test, fields=[('expr', EXPR),
                                              ('ds', DS),
                                              ('h', H),
                                              ('val', VAL)])

    analy = None
    if 'fanaly' in param:
        examples_analy, _ = load_examples(fanaly)
        analy = Dataset(examples_analy, fields=[('expr', EXPR),
                                                ('ds', DS),
                                                ('h', H),
                                                ('val', VAL)])

    train_iter = utils.BucketIterator(train, batch_size=bsz,
                                      sort=True,
                                      shuffle=True,
                                      repeat=False,
                                      sort_key=lambda x: len(x.expr),
                                      batch_size_fn=batch_size_fn,
                                      device=device)
    valid_iter = utils.BucketIterator(valid, batch_size=bsz,
                                      sort=True,
                                      shuffle=True,
                                      repeat=False,
                                      sort_key=lambda x: len(x.expr),
                                      batch_size_fn=batch_size_fn,
                                      device=device)
    test_iter = None
    if 'ftest' in param:
        test_iter = utils.BucketIterator(test, batch_size=bsz,
                                         sort=True,
                                         shuffle=True,
                                         repeat=False,
                                         sort_key=lambda x: len(x.expr),
                                         batch_size_fn=batch_size_fn,
                                         device=device)
    analy_iter = None
    if 'fanaly' in param:
        analy_iter = utils.BucketIterator(analy, batch_size=bsz,
                                          sort=True,
                                          shuffle=True,
                                          repeat=False,
                                          sort_key=lambda x: len(x.expr),
                                          batch_size_fn=batch_size_fn,
                                          device=device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'test_iter': test_iter,
            'analy_iter': analy_iter,
            'SEQ': EXPR,
            'DS': DS,
            'H': H,
            'VAL': VAL}


def valid(model, valid_iter):
    pred_lst = []
    true_lst = []
    itos = ['<unk>', '<pad>', '/', '*', '-', '+', '3', '4', '9', '5', '8', '6', '7', '2', '1']

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            seq, lbl = batch.expr, batch.val
            # print('expr:', ' '.join([itos[ch.item()] for ch in seq[:, 0]]))
            out = model(seq)

            pred = out.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            # assert pred == lbl
            # if i>1:
            #     exit()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accuracy = accuracy_score(true_lst, pred_lst)

    return accuracy


def test_analy(model, itos, analy_iter, enc):
    pred_lst = []
    true_lst = []
    fanalysis = getattr(model.encoder, 'f' + enc)
    # itos = ['<unk>', '<pad>', '/', '*', '-', '+', '3', '4', '9', '5', '8', '6', '7', '2', '1']

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(analy_iter):
            seq, lbl = batch.expr, batch.val
            out = model(seq)

            pred = out.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()

            is_correct = 1 if (pred[0] == lbl[0]) else 0
            # if i == 0:
            #     exit()
            expr = [itos[ch.item()] for ch in seq[:, 0]]
            line = {'type': 'input',
                    'idx': i,
                    'expr': expr,
                    'is_correct': is_correct}
            line = json.dumps(line)
            print(line)
            print(line, file=fanalysis)
            # print('expr:', ' '.join([itos[ch.item()] for ch in seq[:, 0]]))
            # print('expr:', ' '.join([itos[ch.item()] for ch in seq[:, 0]]), file=fanalysis)

            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accuracy = accuracy_score(true_lst, pred_lst)
    return accuracy


def MSU(nlst, modd=True):
    VALUES = range(1, 10)
    NUMERALS = list(map(str, VALUES)) + ['0']
    OP_MAP = ['+', '-', '*', '/']
    OPS = OP_MAP

    def m10eval(op, a0, a1):
        if op == '/':
            # res = int(a0) // int(a1) if not modd else int(a0) % int(a1)
            res = int(a0) + int(a1)
        else:
            res = eval(a0 + op + a1)
        res = res % 10
        return str(res)

    def reducible(mem, ninp):
        if len(mem) < 2:
            return False

        top, sec = mem[0], mem[1]
        if top in OPS and sec in NUMERALS:
            return True
        elif top in NUMERALS and sec[0] in NUMERALS and sec[1] in OPS:
            if sec[1] in ['+', '-'] and ninp not in ['*', '/']:
                return True
            if sec[1] in ['*', '/']:
                return True
            return False
        elif top == ')' and sec[0] == '(' and sec[1] in NUMERALS:
            return True
        elif top in NUMERALS and sec == '(' and ninp == ')':
            return True

        return False

    def reduce(mem):
        top = mem.pop(0)
        sec = mem.pop(0)

        if top in OPS and sec in NUMERALS:
            reduced = (sec, top)
        elif top in NUMERALS and sec[0] in NUMERALS and sec[1] in OPS:
            reduced = m10eval(sec[1], sec[0], top)
        elif top == ')' and sec[0] == '(' and sec[1] in NUMERALS:
            reduced = sec[1]
        elif top in NUMERALS and sec == '(':
            reduced = (sec, top)
        else:
            raise NotImplementedError

        return reduced

    stack = []
    reduce_lst = []
    msu = 0
    for t, n in enumerate(nlst):
        stack.insert(0, n)
        if len(stack) > msu:
            msu = len(stack)
        # reduce_lst.append(0)
        if t != len(nlst) - 1:
            ninp = nlst[t + 1]
        else:
            ninp = None

        r = 0
        while reducible(stack, ninp):
            stack.insert(0, reduce(stack))
            r += 1
            # reduce_lst.append(1)
        reduce_lst.append(r)

    return msu


def valid_mmc(model, itos, valid_iter):
    def MMC(nlst, sdlst):
        def combine(a0, a1):
            def unfinised(n):
                return isinstance(n, list) or isinstance(n, tuple)

            if unfinised(a0):
                if a1 == ')':
                    lp, num = a0
                    assert lp == '('
                    return num
                else:
                    num, op = a0
                    try:
                        return eval(''.join([str(num), op, str(a1)]))
                    except:
                        return -1
            else:
                return (a0, a1)

        N = 0
        mmc = 0
        mem = []
        mem.append((nlst[0], sdlst[0]))

        for n, sd in zip(nlst[1:], sdlst[1:] + [10000]):
            while sd > mem[-1][1]:
                a0, sd0 = mem.pop()
                n = combine(a0, n)
                if len(mem) == 0:
                    break
            mem.append((n, sd))
            if len(mem) > mmc:
                mmc = len(mem)

        assert len(mem) == 1
        return mmc

    pred_dict = {}
    true_dict = {}
    nsamples = defaultdict(int)
    acc = defaultdict(float)
    incorrect_predicts = []

    PAD_EXPR = None
    for i, word in enumerate(itos):
        if word == PAD:
            PAD_EXPR = i

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            seq, lbl, depth, ds, h = batch.expr, batch.val, batch.h, batch.ds, batch.h
            res_clf = model(seq)

            pred = res_clf.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            seq = seq.transpose(0, 1)
            bsz, len_total = seq.shape
            mask_seq = seq.eq(PAD_EXPR)
            lens_seq = len_total - mask_seq.sum(1)

            ds = ds.transpose(0, 1)
            bsz, len_total = ds.shape
            mask_ds = ds.eq(PAD_DS)
            lens_ds = len_total - mask_ds.sum(1)

            for seq_b, pred_b, lbl_b, depth_b, ds_b, h_b, ld_b, ls_b in \
                    zip(seq, pred, lbl, depth, ds, h, lens_ds, lens_seq):
                nlst = list(map(lambda x: itos[x], seq_b[:ls_b]))
                sdlst = list(ds_b[:ld_b].cpu().data.numpy())
                # mmc = MMC(nlst, sdlst)
                mmc = MSU(nlst)

                # depth_b = depth_b.item()
                pred_b = pred_b.item()
                h_b = h_b.item()
                # if depth_b > 49:
                #     continue
                if mmc not in pred_dict:
                    pred_dict[mmc] = []
                    true_dict[mmc] = []
                pred_dict[mmc].append(pred_b)
                true_dict[mmc].append(lbl_b)
                nsamples[mmc] += 1

                if pred_b != lbl_b:
                    expr = ' '.join([itos[ch.item()] for ch in seq_b if itos[ch] != PAD])
                    ds_b = ' '.join([str(d) for d in list(ds_b.cpu().numpy()) if d != PAD_DS])
                    lbl_b = str(lbl_b)
                    h_b = str(h_b)
                    incorrect_predicts.append((expr, ds_b, lbl_b, h_b))

    for mmc in pred_dict.keys():
        acc[mmc] = accuracy_score(true_dict[mmc], pred_dict[mmc])
    true_whole = [lbl for mmc in true_dict.keys() for lbl in true_dict[mmc]]
    pred_whole = [pred for mmc in pred_dict.keys() for pred in pred_dict[mmc]]
    acc_total = accuracy_score(true_whole, pred_whole)

    return acc, nsamples, incorrect_predicts, acc_total


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

    pred_dict = {}
    true_dict = {}
    nsamples = defaultdict(int)
    acc = defaultdict(float)
    incorrect_predicts = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            seq, lbl, depth, ds, h = batch.expr, batch.val, batch.h, batch.ds, batch.h
            res_clf = model(seq)

            pred = res_clf.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            seq = seq.transpose(0, 1)
            ds = ds.transpose(0, 1)
            bsz, len_total = ds.shape
            mask_ds = ds.eq(PAD_DS)
            lens_ds = len_total - mask_ds.sum(1)

            for seq_b, pred_b, lbl_b, depth_b, ds_b, h_b, lens_b in zip(seq, pred, lbl, depth, ds, h, lens_ds):
                ne = num_extrem_vals(ds_b[:lens_b])

                # depth_b = depth_b.item()
                pred_b = pred_b.item()
                h_b = h_b.item()
                # if depth_b > 49:
                #     continue
                if ne not in pred_dict:
                    pred_dict[ne] = []
                    true_dict[ne] = []
                pred_dict[ne].append(pred_b)
                true_dict[ne].append(lbl_b)
                nsamples[ne] += 1

                if pred_b != lbl_b:
                    expr = ' '.join([itos[ch.item()] for ch in seq_b if itos[ch] != PAD])
                    ds_b = ' '.join([str(d) for d in list(ds_b.cpu().numpy()) if d != PAD_DS])
                    lbl_b = str(lbl_b)
                    h_b = str(h_b)
                    incorrect_predicts.append((expr, ds_b, lbl_b, h_b))

    for ne in pred_dict.keys():
        acc[ne] = accuracy_score(true_dict[ne], pred_dict[ne])
    true_whole = [lbl for ne in true_dict.keys() for lbl in true_dict[ne]]
    pred_whole = [pred for ne in pred_dict.keys() for pred in pred_dict[ne]]
    acc_total = accuracy_score(true_whole, pred_whole)

    return acc, nsamples, incorrect_predicts, acc_total


def train(model, iters, opt, optim, scheduler):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']
    criterion_clf = nn.CrossEntropyLoss()

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
        train_iter_tqdm = tqdm.tqdm(train_iter)
        for i, batch in enumerate(train_iter_tqdm):
            expr = batch.expr
            val = batch.val

            model.train()
            model.zero_grad()
            out = model(expr)
            loss = criterion_clf(out, val)
            losses.append(loss.item())

            loss.backward()
            gnorm = clip_grad_norm_(model.parameters(), opt.gclip)
            gnorms.append(gnorm)

            optim.step()
            train_iter_tqdm.set_description(f'Epoch {epoch} loss {loss.item():.4f} gnorm {gnorm:.4f}')
            # valid:
            # if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
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
        self.rep2val = nn.Sequential(utils.LayerNormalization(self.hdim), nn.Dropout(drop),
                                     nn.Linear(self.hdim, 16), nn.ReLU(),
                                     utils.LayerNormalization(16), nn.Dropout(drop),
                                     nn.Linear(16, 16), nn.ReLU(),
                                     utils.LayerNormalization(16), nn.Dropout(drop),
                                     nn.Linear(16, 10))

    def enc(self, seq):
        mask = seq.data.eq(self.padding_idx)
        len_total, bsz = seq.shape
        lens = len_total - mask.sum(dim=0)

        # inp = self.embedding_drop(True, seq)
        inp = self.embedding(seq)
        os = self.encoder(embs=inp, mask=1 - mask, lens=lens)
        rep = os[lens - 1, range(bsz)]
        return rep

    def forward(self, seq):
        rep = self.enc(seq)
        val = self.rep2val(rep)

        return val
