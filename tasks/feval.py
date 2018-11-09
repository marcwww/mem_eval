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

    def __init__(self, expr, ds, h, val):
        self.expr = self.tokenizer_expr(expr)
        self.ds = self.tokenizer_ds(ds)
        self.h = int(h)
        self.val = int(val)

    def tokenizer_expr(self, seq):
        return seq.split()

    def tokenizer_ds(self, ds):
        return [int(d) for d in ds.split()]

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            expr, ds, h, val = \
                line.strip().split('\t')
            examples.append(Example(expr, ds, h, val))

    return examples

def build_iters(**param):

    ftrain = param['ftrain']
    fvalid = param['fvalid']
    ftest = None
    fanaly = None
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
    VAL = torchtext.data.Field(sequential=False, use_vocab=False)

    train = Dataset(examples_train, fields=[('expr', EXPR),
                                            ('ds', DS),
                                            ('h', H),
                                            ('val', VAL)])
    EXPR.build_vocab(train)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('expr', EXPR),
                                            ('ds', DS),
                                            ('h', H),
                                            ('val', VAL)])

    test = None
    if 'ftest' in param:
        examples_test = load_examples(ftest)
        test = Dataset(examples_test, fields=[('expr', EXPR),
                                            ('ds', DS),
                                            ('h', H),
                                            ('val', VAL)])

    analy = None
    if 'fanaly' in param:
        examples_analy  = load_examples(fanaly)
        analy = Dataset(examples_analy, fields=[('expr', EXPR),
                                            ('ds', DS),
                                            ('h', H),
                                            ('val', VAL)])

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
    test_iter = None
    if 'ftest' in param:
        test_iter = torchtext.data.Iterator(test, batch_size=bsz,
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

def test_analy(model, analy_iter, enc):
    pred_lst = []
    true_lst = []
    fanalysis = getattr(model.encoder, 'f' + enc)
    itos = ['<unk>', '<pad>', '/', '*', '-', '+', '3', '4', '9', '5', '8', '6', '7', '2', '1']

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

def valid_detail(model, itos, valid_iter):
    pred_dict= {}
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
            for seq_b, pred_b, lbl_b, depth_b, ds_b, h_b in zip(seq, pred, lbl, depth, ds, h):
                depth_b = depth_b.item()
                pred_b = pred_b.item()
                h_b = h_b.item()
                # if depth_b > 49:
                #     continue
                if depth_b not in pred_dict:
                    pred_dict[depth_b] = []
                    true_dict[depth_b] = []
                pred_dict[depth_b].append(pred_b)
                true_dict[depth_b].append(lbl_b)
                nsamples[depth_b] += 1

                if pred_b != lbl_b:
                    expr = ' '.join([itos[ch.item()] for ch in seq_b if itos[ch] != PAD])
                    ds_b = ' '.join([str(d) for d in list(ds_b.cpu().numpy()) if d != PAD_DS])
                    lbl_b = str(lbl_b)
                    h_b = str(h_b)
                    incorrect_predicts.append((expr, ds_b, lbl_b, h_b))

    for depth in pred_dict.keys():
        acc[depth] = accuracy_score(true_dict[depth], pred_dict[depth])

    return acc, nsamples, incorrect_predicts

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
        for i, batch in enumerate(train_iter):
            expr = batch.expr
            val = batch.val

            model.train()
            model.zero_grad()
            out = model(expr)
            loss = criterion_clf(out, val)
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
        os = self.encoder(embs=inp, mask=1-mask, lens = lens)
        rep = os[lens - 1, range(bsz)]
        return rep

    def forward(self, seq):
        rep = self.enc(seq)
        val = self.rep2val(rep)

        return val



