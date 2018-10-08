import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score

class Example(object):

    def __init__(self, seq, lbl, d, etype):
        self.seq = self.tokenizer(seq)
        self.lbl = int(lbl)
        self.d = int(d)
        self.etype = int(etype)

    def tokenizer(self, seq):
        return list(seq)

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            seq, lbl, d, t = \
                line.strip().split('\t')
            examples.append(Example(seq, lbl, d, t))

    return examples

def build_iters(**param):

    ftrain = param['ftrain']
    fvalid = param['fvalid']

    bsz = param['bsz']
    device = param['device']

    examples_train = load_examples(ftrain)

    SEQ = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS)
    LBL = torchtext.data.Field(sequential=False, use_vocab=False)
    D = torchtext.data.Field(sequential=False, use_vocab=False)
    ETYPE = torchtext.data.Field(sequential=False, use_vocab=False)

    train = Dataset(examples_train, fields=[('seq', SEQ),
                                            ('lbl', LBL),
                                            ('d', D),
                                            ('etype', ETYPE)])
    SEQ.build_vocab(train)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('seq', SEQ),
                                            ('lbl', LBL),
                                            ('d', D),
                                            ('etype', ETYPE)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq),
                                         sort_within_batch=True,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq),
                                         sort_within_batch=True,
                                         device=device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'SEQ': SEQ,
            'LBL': LBL,
            'ETYPE': ETYPE}

def build_iters_test(**param):

    ftrain = param['ftrain']
    fvalid = param['fvalid']
    ftest = param['ftest']

    bsz = param['bsz']
    device = param['device']

    examples_train = load_examples(ftrain)

    SEQ = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS)
    LBL = torchtext.data.Field(sequential=False, use_vocab=False)
    D = torchtext.data.Field(sequential=False, use_vocab=False)
    ETYPE = torchtext.data.Field(sequential=False, use_vocab=False)

    train = Dataset(examples_train, fields=[('seq', SEQ),
                                            ('lbl', LBL),
                                            ('d', D),
                                            ('etype', ETYPE)])
    SEQ.build_vocab(train)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('seq', SEQ),
                                            ('lbl', LBL),
                                            ('d', D),
                                            ('etype', ETYPE)])

    examples_test = load_examples(ftest)
    test = Dataset(examples_test, fields=[('seq', SEQ),
                                            ('lbl', LBL),
                                            ('d', D),
                                            ('etype', ETYPE)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq),
                                         sort_within_batch=True,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq),
                                         sort_within_batch=True,
                                         device=device)
    test_iter = torchtext.data.Iterator(test, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.seq),
                                         sort_within_batch=True,
                                         device=device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'test_iter': test_iter,
            'SEQ': SEQ,
            'LBL': LBL,
            'ETYPE': ETYPE}

def valid(model, valid_iter):
    pred_lst = []
    true_lst = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            seq, lbl = batch.seq, batch.lbl
            res= model(seq)
            res_clf = res['res_clf']

            pred = res_clf.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()
            pred_lst.extend(pred)
            true_lst.extend(lbl)

    accuracy = accuracy_score(true_lst, pred_lst)
    precision = precision_score(true_lst, pred_lst)
    recall = recall_score(true_lst, pred_lst)
    f1 = f1_score(true_lst, pred_lst)

    # return accuracy, precision, recall, f1
    return accuracy

def valid_detail(model, valid_iter):
    pred_lsts = {}
    true_lsts = {}

    etypes = ['exchange', 'omit', 'redun']

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            seq, lbl, d, etype = batch.seq, batch.lbl, batch.d, batch.etype
            res= model(seq)
            res_clf = res['res_clf']

            pred = res_clf.max(dim=1)[1].cpu().numpy()
            lbl = lbl.cpu().numpy()

            for b, (d, e) in enumerate(zip(d, etype)):
                d = d.item()
                if d not in pred_lsts:
                    pred_lsts[d] = {'exchange':[], 'omit':[], 'redun':[], 'overall':[]}
                    true_lsts[d] = {'exchange':[], 'omit':[], 'redun':[], 'overall':[]}
                pred_lsts[d][etypes[e]].append(pred[b])
                true_lsts[d][etypes[e]].append(lbl[b])
                pred_lsts[d]['overall'].append(pred[b])
                true_lsts[d]['overall'].append(lbl[b])

    accuracy = {}
    for d in pred_lsts.keys():
        accuracy[d] = {}
        for k in pred_lsts[d].keys():
            accuracy[d][k] = accuracy_score(true_lsts[d][k], pred_lsts[d][k])

    return accuracy

def train(model, iters, opt, optim, scheduler):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']
    criterion_clf = nn.CrossEntropyLoss()
    criterion_lm = nn.CrossEntropyLoss(ignore_index=model.padding_idx)

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
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            seq = batch.seq
            lbl = batch.lbl

            model.train()
            model.zero_grad()
            out = model(seq[:-1])
            pred_lbl = out['res_clf']
            next_words = out['next_words']

            loss_clf = criterion_clf(pred_lbl, lbl)
            # mask: (bsz)
            mask = lbl.eq(0).expand_as(seq)
            # seq: (seq_len, bsz)
            seq_lm = seq.clone().masked_fill_(mask, model.padding_idx)
            loss_lm = criterion_lm(next_words.view(-1, model.num_words),
                                   seq_lm[1:].view(-1))

            loss = (loss_clf + opt.lm_coef * loss_lm)/(1 + opt.lm_coef)
            losses.append(loss.item())

            loss.backward()
            clip_grad_norm(model.parameters(), 5)
            optim.step()
            loss = {'clf_loss': loss.item()}

            utils.progress_bar(i / len(train_iter), loss, epoch)

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                loss_ave = np.array(losses).sum() / len(losses)
                losses = []
                accurracy = \
                    valid(model, valid_iter)
                log_str = '{\'Epoch\':%d, \'Format\':\'a/l\', \'Metrics\':[%.4f, %.4f]}' % \
                          (epoch, accurracy, loss_ave)
                print(log_str)
                with open(log_path, 'a+') as f:
                    f.write(log_str + '\n')

                scheduler.step(accurracy)
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
        # self.embedding_drop = \
        #     utils.fixMaskEmbeddedDropout(self.embedding, drop)
        self.hdim = self.encoder.odim
        self.clf = nn.Linear(self.hdim, 2)
        self.padding_idx = embedding.padding_idx
        self.num_words = embedding.num_embeddings
        self.out2esz = nn.Linear(self.hdim, self.embedding.embedding_dim)
        self.edrop = nn.Dropout(drop)

    def enc(self, seq):
        mask = seq.data.eq(self.padding_idx)
        len_total, bsz = seq.shape
        lens = len_total - mask.sum(dim=0)

        # inp = self.embedding_drop(True, seq)
        inp = self.embedding(seq)
        os = self.encoder(embs=inp, mask=1-mask, lens = lens)
        rep = torch.cat([os[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                         dim=0)
        return rep, os

    def forward(self, seq):
        rep, os = self.enc(seq)
        w_t = self.embedding.weight.transpose(0, 1)
        next_words = self.out2esz(os).matmul(w_t)

        res_clf = self.clf(rep)
        return {'res_clf':res_clf,
                'next_words': next_words}



