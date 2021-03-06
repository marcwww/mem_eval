import nets
from macros import *
import torch
import utils
from params import opts
import argparse
from torch import nn
from tasks import feval
import crash_on_ipy
import sys

if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='feval_test_ne.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)
    opts.general_opts(parser)
    if '-task' in sys.argv:
        task = sys.argv[sys.argv.index('-task') + 1]
    else:
        task = parser._option_string_actions['-task'].default

    if '-enc_type' in sys.argv:
        enc_type = sys.argv[sys.argv.index('-enc_type') + 1]
    else:
        enc_type = parser._option_string_actions['-enc_type'].default

    parser = opts.select_opt(task, enc_type, parser)
    opt = parser.parse_args()

    assert opt.task == 'feval'

    train = feval.train
    valid = feval.valid
    build_iters = feval.build_iters
    valid_detail = feval.valid_detail
    Model = feval.Model

    res_iters = build_iters(ftrain=os.path.join('..', opt.ftrain),
                            fvalid=os.path.join('..', opt.fvalid),
                            ftest=os.path.join('..', opt.ftest),
                            bsz=opt.bsz,
                            device=opt.gpu,
                            sub_task=opt.sub_task,
                            seq_len_max=opt.seq_len_max)

    embedding = None
    embedding_enc = None
    embedding_dec = None
    SEQ = None
    SRC = None
    TAR = None
    SGOLD = None
    if 'SEQ' in res_iters.keys():
        SEQ = res_iters['SEQ']
        embedding = nn.Embedding(num_embeddings=len(SEQ.vocab.itos),
                                 embedding_dim=opt.edim,
                                 padding_idx=SEQ.vocab.stoi[PAD])
        embedding.weight.requires_grad = not opt.fix_emb

    if 'SRC' in res_iters.keys() and 'TAR' in res_iters.keys():
        SRC = res_iters['SRC']
        embedding_enc = nn.Embedding(num_embeddings=len(SRC.vocab.itos),
                                     embedding_dim=opt.edim,
                                     padding_idx=SRC.vocab.stoi[PAD])
        embedding_enc.weight.requires_grad = not opt.fix_emb

        TAR = res_iters['TAR']
        embedding_dec = nn.Embedding(num_embeddings=len(TAR.vocab.itos),
                                 embedding_dim=opt.edim,
                                 padding_idx=TAR.vocab.stoi[PAD])
        embedding_dec.weight.requires_grad = not opt.fix_emb

    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    if opt.emb_type == 'one-hot':
        if embedding is not None:
            one_hot_mtrx = utils.one_hot_matrix(SEQ.vocab.stoi, device, opt.edim)
            embedding.weight.data.copy_(one_hot_mtrx)
            embedding.weight.requires_grad = False

        if embedding_enc is not None:
            one_hot_mtrx = utils.one_hot_matrix(SRC.vocab.stoi, device, opt.edim)
            embedding_enc.weight.data.copy_(one_hot_mtrx)
            embedding_enc.weight.requires_grad = False

        if embedding_dec is not None:
            one_hot_mtrx = utils.one_hot_matrix(TAR.vocab.stoi, device, opt.edim)
            embedding_dec.weight.data.copy_(one_hot_mtrx)
            embedding_dec.weight.requires_grad = False

    encoder = None
    decoder = None
    if opt.enc_type == 'srnn':
        encoder = nets.EncoderSRNN(idim=opt.edim,
                                   cdim=opt.hdim,
                                   dropout=opt.dropout)

    if opt.enc_type == 'ntm':
        encoder = nets.EncoderNTM(idim=opt.edim,
                                    cdim=opt.hdim,
                                    N=opt.N,
                                    M=opt.M,
                                    drop=opt.dropout,
                                    read_first=opt.read_first)
    if opt.enc_type == 'sarnn':
        encoder = nets.EncoderSARNN(idim=opt.edim,
                                    cdim=opt.hdim,
                                    N=opt.N,
                                    M=opt.M,
                                    drop=opt.dropout,
                                    read_first=opt.read_first)
    if opt.enc_type == 'lstm':
        encoder = nets.EncoderLSTM(idim=opt.edim,
                                    cdim=opt.hdim,
                                    drop=opt.dropout)

    if opt.enc_type == 'alstm':
        encoder = nets.EncoderALSTM(idim=opt.edim,
                                    cdim=opt.hdim,
                                    N=opt.N,
                                    M=opt.M,
                                    drop=opt.dropout,
                                    read_first=opt.read_first)

    model = None
    if embedding is None:
        model = Model(encoder, opt.odim, opt.dropout).to(device)
    else:
        model = Model(encoder, embedding, opt.dropout).to(device)
    utils.init_model(model)

    if opt.fload is not None:
        model_fname = opt.fload
        location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
        model_path = os.path.join(RES, model_fname)
        model_path = os.path.join('..', model_path)
        model_dict = torch.load(model_path, map_location=location)
        model.load_state_dict(model_dict)
        print('Loaded from ' + model_path)

    param_str = utils.param_str(opt)
    for key, val in param_str.items():
        print(str(key) + ': ' + str(val))

    print('Valid result: \n', valid(model, res_iters['valid_iter']))
    acc, nt, incorrect_predicts, acc_total = valid_detail(model, SEQ.vocab.itos, res_iters['test_iter'])
    print('Test result total: \n', acc_total)
    print('Test result: \n', sorted(acc.items()))
    print('# samples under different ne\'s:', sorted(nt.items()))

    fincorrect = os.path.join(os.path.join('..', RES),
                              'incor-%s-%s-%d.txt' % ('feval', opt.enc_type, utils.time_int()))
    with open(fincorrect, 'w') as f:
        for idx, line in enumerate(incorrect_predicts):
            line = '\t'.join([str(idx)] + list(line)) + '\n'
            f.write(line)

