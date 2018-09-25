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

    def __init__(self, seq, lbl):
        self.seq = self.tokenizer(seq)
        self.lbl = lbl

    def tokenizer(self, seq):
        return list(seq)

def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            lbl, seq = \
                line.split('\t')
            examples.append(Example(seq, lbl))

    return examples

def build_iters(param_iter):

    ftrain = param_iter['ftrain']
    fvalid = param_iter['fvalid']
    bsz = param_iter['bsz']
    device = param_iter['device']

    examples_train = load_examples(ftrain)

    SEQ = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=EOS)
    LBL = torchtext.data.Field(sequential=False, use_vocab=False)

    train = Dataset(examples_train, fields=[('seq', SEQ),
                                            ('lbl', LBL)])
    SEQ.build_vocab(train)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('seq', SEQ),
                                            ('lbl', LBL)])

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
            'LBL': LBL}

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

def train(model, iters, opt, criterion, optim, scheduler):
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
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            seq = batch.seq
            lbl = batch.lbl

            seq_len = seq.shape[0]
            model.train()
            model.zero_grad()
            out = model(seq[:-1])
            pred_lbl = out['res_clf']
            next_words = out['next_words']

            loss_clf = criterion(pred_lbl, lbl)
            # mask: (bsz)
            mask = lbl.eq(0).expand_as(seq)
            # seq: (seq_len, bsz)
            seq_lm = seq.clone().masked_fill_(mask, model.padding_idx)
            loss_lm = F.cross_entropy(next_words.view(-1, model.num_words), seq_lm[1:].view(-1),
                            ignore_index=model.padding_idx)

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

    def __init__(self, encoder, embedding):
        super(Model, self).__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.hdim = self.encoder.odim
        self.clf = nn.Linear(self.hdim, 2)
        self.padding_idx = embedding.padding_idx
        self.num_words = embedding.num_embeddings
        self.out = nn.Linear(self.hdim,
                             embedding.num_embeddings)

    def enc(self, seq):
        mask = seq.data.eq(self.padding_idx)
        len_total, bsz = seq.shape
        lens = len_total - mask.sum(dim=0)

        inp = self.embedding(seq)
        res = self.encoder(embs=inp, lens=lens)
        output = res['output']
        reps = torch.cat([output[lens[b] - 1, b, :].unsqueeze(0) for b in range(bsz)],
                         dim=0)
        res['reps'] = reps
        res['output'] = output
        return res

    def forward(self, seq):
        res = self.enc(seq)
        rep = res['reps']
        output = res['output']
        next_words = self.out(output)

        res_clf = self.clf(rep)
        return {'res_clf':res_clf,
                'next_words': next_words}



