import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import utils
import tqdm
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')


class Example(object):

    def __init__(self, sen, sgold, sdiff, nint):
        self.sen = self.tokenizer(sen)
        self.sgold = self.tokenizer(sgold)
        self.sdiff = self.tokenizer(sdiff)
        self.nint = int(nint)

    def tokenizer(self, seq):
        return seq.split()


def load_examples(fname):
    examples = []
    lens = []

    with open(fname, 'r') as f:
        for line in f:
            sen, sgold, sdiff, nint = \
                line.strip().split('\t')
            examples.append(Example(sen, sgold, sdiff, nint))
            lens.append(len(sgold.split()))

    return examples, np.mean(lens)


def build_iters(**param_iter):
    ftrain = param_iter['ftrain']
    fvalid = param_iter['fvalid']
    bsz = param_iter['bsz']
    device = param_iter['device']
    device = torch.device(device if device != -1 else 'cpu')

    examples_train, len_ave = load_examples(ftrain)

    SEQ = torchtext.data.Field(sequential=True, use_vocab=True,
                               pad_token=PAD,
                               unk_token=UNK,
                               eos_token=None)
    NUM = torchtext.data.Field(sequential=False,
                               use_vocab=False)
    fields = [('sen', SEQ), ('sgold', SEQ), ('sdiff', SEQ), ('nint', NUM)]

    train = Dataset(examples_train, fields=fields)
    SEQ.build_vocab(train)
    examples_valid, _ = load_examples(fvalid)
    valid = Dataset(examples_valid, fields=fields)

    def batch_size_fn(new_example, current_count, ebsz):
        return ebsz + (len(new_example.sgold) / len_ave) ** 0.3

    train_iter = utils.BucketIterator(train, batch_size=bsz,
                                      sort=True,
                                      shuffle=True,
                                      repeat=False,
                                      sort_key=lambda x: len(x.sgold),
                                      batch_size_fn=batch_size_fn,
                                      device=device)
    valid_iter = utils.BucketIterator(valid, batch_size=bsz,
                                      sort=True,
                                      shuffle=True,
                                      repeat=False,
                                      sort_key=lambda x: len(x.sgold),
                                      batch_size_fn=batch_size_fn,
                                      device=device)
    # train_iter = torchtext.data.Iterator(train, batch_size=bsz,
    #                                      sort=False, repeat=False,
    #                                      sort_key=lambda x: len(x.sgold),
    #                                      sort_within_batch=True,
    #                                      device=device)
    # valid_iter = torchtext.data.Iterator(valid, batch_size=bsz,
    #                                      sort=False, repeat=False,
    #                                      sort_key=lambda x: len(x.sgold),
    #                                      sort_within_batch=True,
    #                                      device=device)

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
    nc_nint = defaultdict(int)
    nt_nint = defaultdict(int)
    acc_nint = defaultdict(tuple)
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(valid_iter):
            sgold = batch.sgold
            sdiff = batch.sdiff
            nint = batch.nint
            next_words = model(sgold[:-1])
            bsz = next_words.shape[1]

            lens = sgold.ne(model.padding_idx).sum(0)
            final_word_logits = next_words[lens - 2, range(bsz)]
            gold = sgold[lens - 1, range(bsz)]
            diff = sdiff[lens - 1, range(bsz)]

            prob_diff = final_word_logits[range(bsz), gold] - \
                        final_word_logits[range(bsz), diff]

            correct = prob_diff.gt(0).cpu().numpy()
            nint = nint.cpu().numpy()
            for nc, ni in zip(correct, nint):
                nc_nint[ni] += nc
                nt_nint[ni] += 1

    for ni in nc_nint.keys():
        if ni > 5:
            continue
        acc_nint[ni] = (round(float(nc_nint[ni]) / float(nt_nint[ni]), 4), nt_nint[ni])
    accuracy = sum([value for key, value in nc_nint.items()]) / \
               sum([value for key, value in nt_nint.items()])
    accuracy = round(accuracy, 4)
    logging.info(f'acc_nint: {acc_nint}')
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

    acc = valid(model, valid_iter)
    print(acc)
    best_performance = 0
    losses = []
    for epoch in range(opt.nepoch):
        train_iter_tqdm = tqdm.tqdm(train_iter)
        for i, batch in enumerate(train_iter_tqdm):
            sen = batch.sen

            model.train()
            model.zero_grad()
            next_words = model(sen[:-1])

            loss = criterion_lm(next_words.view(-1, model.num_words), sen[1:].view(-1))
            losses.append(loss.item())
            loss.backward()
            gnorm = clip_grad_norm_(model.parameters(), opt.gclip)
            optim.step()
            train_iter_tqdm.set_description(f'Epoch {epoch} loss {loss.item():.4f} gnorm {gnorm:.4f}')

            if (i + 1) % int(1 / 4 * len(train_iter)) == 0:
                # print('\r')
                loss_ave = np.array(losses).sum() / len(losses)
                losses = []
                accurracy = \
                    valid(model, valid_iter)
                log_str = '{\'Epoch\':%d, \'Format\':\'a/l\', \'Metrics\':[%.4f, %.4f]}' % \
                          (epoch, accurracy, loss_ave)
                print(log_str + '\n')
                with open(log_path, 'a+') as f:
                    f.write(log_str + '\n')

                scheduler.step(loss_ave)
                for param_group in optim.param_groups:
                    print('learning rate:', param_group['lr'])

                if accurracy > best_performance:
                    best_performance = accurracy
                    model_fname = basename + ".model"
                    save_path = os.path.join(RES, model_fname)
                    print('Saving to ' + save_path + '\n')
                    torch.save(model.state_dict(), save_path)


class Model(nn.Module):

    def __init__(self, encoder, embedding, edrop):
        super(Model, self).__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.embedding_drop = \
            utils.fixMaskEmbeddedDropout(self.embedding, edrop)
        self.hdim = self.encoder.odim
        self.clf = nn.Linear(self.hdim, 2)
        self.padding_idx = embedding.padding_idx
        self.num_words = embedding.num_embeddings
        self.out2esz = nn.Linear(self.hdim, self.embedding.embedding_dim)
        # self.out = nn.Linear(self.hdim,
        #                      embedding.num_embeddings)

    def enc(self, seq):
        mask = seq.data.eq(self.padding_idx)
        len_total, bsz = seq.shape
        lens = len_total - mask.sum(dim=0)

        inp = self.embedding_drop(True, seq)
        res = self.encoder(embs=inp, lens=lens)
        output = res
        return output

    def forward(self, seq):
        output = self.enc(seq)
        w_t = self.embedding.weight.transpose(0, 1)
        next_words = self.out2esz(output).matmul(w_t)
        # next_words = self.out(output)

        return next_words
