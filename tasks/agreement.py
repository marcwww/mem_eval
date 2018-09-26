import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils


class Example(object):

    def __init__(self, sen, sgold, sdiff):
        self.sen = self.tokenizer(sen)
        self.sgold = self.tokenizer(sgold)
        self.sdiff = self.tokenizer(sdiff)

    def tokenizer(self, seq):
        return seq.split()


def load_examples(fname):
    examples = []

    with open(fname, 'r') as f:
        for line in f:
            sen, sgold, sdiff = \
                line.strip().split('\t')
            examples.append(Example(sen, sgold, sdiff))

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
                               eos_token=None)

    train = Dataset(examples_train, fields=[('sen', SEQ),
                                            ('sgold', SEQ),
                                            ('sdiff', SEQ)])
    SEQ.build_vocab(train)
    examples_valid = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=[('sen', SEQ),
                                            ('sgold', SEQ),
                                            ('sdiff', SEQ)])

    train_iter = torchtext.data.Iterator(train, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.sgold),
                                         sort_within_batch=True,
                                         device=device)
    valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
                                         sort=False, repeat=False,
                                         sort_key=lambda x: len(x.sgold),
                                         sort_within_batch=True,
                                         device=device)

    return {'train_iter': train_iter,
            'valid_iter': valid_iter,
            'SEQ': SEQ}

def build_iters_test(ftests, SEQ, bsz, device):
    iters = []
    for ftest in ftests:
        examples = load_examples(ftest)
        test = Dataset(examples, fields=[('sen', SEQ),
                                    ('sgold', SEQ),
                                    ('sdiff', SEQ)])
        iter = torchtext.data.Iterator(test, batch_size=bsz,
                                             sort=False, repeat=False,
                                             sort_key=lambda x: len(x.sgold),
                                             sort_within_batch=True,
                                             device=device)
        iters.append(iter)

    return iters

def valid(model, valid_iter):
    nc = 0
    nt = 0
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            sgold = batch.sgold
            sdiff = batch.sdiff
            next_words = model(sgold[:-1])
            bsz = next_words.shape[1]

            lens = sgold.ne(model.padding_idx).sum(0)
            final_word_logits = next_words[lens-2, range(bsz)]
            gold = sgold[lens-1, range(bsz)]
            diff = sdiff[lens-1, range(bsz)]

            prob_diff = final_word_logits[range(bsz), gold] - \
                    final_word_logits[range(bsz), diff]

            nc += prob_diff.gt(0).sum().item()
            nt += bsz
            # loss_gold = F.cross_entropy(next_words.view(-1, model.num_words),
            #                     sgold[1:].view(-1), reduce=False, ignore_index=model.padding_idx)
            # loss_diff = F.cross_entropy(next_words.view(-1, model.num_words),
            #                     sdiff[1:].view(-1), reduce=False, ignore_index=model.padding_idx)
            #
            # nc += (loss_diff - loss_gold).gt(0).sum().item()
            # nt += sgold.shape[0]

    accuracy = nc / nt
    return accuracy

def train(model, iters, opt, optim, scheduler):
    train_iter = iters['train_iter']
    valid_iter = iters['valid_iter']

    criterion_lm = nn.CrossEntropyLoss(ignore_index=model.padding_idx)

    basename = "{}-{}-{}-{}".format(opt.task,
                                       opt.sub_task,
                                       opt.enc_type,
                                       utils.time_int())
    log_fname = basename + ".json"
    log_path = os.path.join(RES, log_fname)
    with open(log_path, 'w') as f:
        f.write(str(utils.param_str(opt)) + '\n')

    best_performance = 0
    losses = []
    for epoch in range(opt.nepoch):
        for i, batch in enumerate(train_iter):
            sen = batch.sen

            model.train()
            model.zero_grad()
            next_words = model(sen[:-1])

            loss = criterion_lm(next_words.view(-1, model.num_words), sen[1:].view(-1))
            losses.append(loss.item())
            loss.backward()
            clip_grad_norm(model.parameters(), 15)
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
        res['output'] = output
        return res

    def forward(self, seq):
        res = self.enc(seq)
        output = res['output']
        next_words = self.out(output)

        return next_words



