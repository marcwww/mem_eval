import nets
from macros import *
import torch
import utils
from params import opts
import argparse
from torch import nn
from tasks import feval
from utils import analy
import crash_on_ipy


if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='flang_test.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)
    opts.general_opts(parser)
    opt = parser.parse_args()

    parser = opts.select_opt(opt, parser)
    opt = parser.parse_args()

    utils.init_seed(opt.seed)

    assert opt.task == 'feval'

    train = feval.train
    valid = feval.valid
    test_analy = feval.test_analy

    build_iters = feval.build_iters
    valid_detail = feval.valid_detail
    Model = feval.Model

    res_iters = build_iters(ftrain=os.path.join('..', opt.ftrain),
                            fvalid=os.path.join('..', opt.fvalid),
                            fanaly=os.path.join('..', opt.fanaly),
                            bsz=1,
                            device=opt.gpu,
                            sub_task=opt.sub_task)

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
                                    drop=opt.dropout)
    if opt.enc_type == 'sarnn':
        encoder = nets.EncoderSARNN(idim=opt.edim,
                                    cdim=opt.hdim,
                                    N=opt.N,
                                    M=opt.M,
                                    drop=opt.dropout)
    if opt.enc_type == 'lstm':
        encoder = nets.EncoderLSTM(idim=opt.edim,
                                    cdim=opt.hdim,
                                    drop=opt.dropout)

    if opt.enc_type == 'alstm':
        encoder = nets.EncoderALSTM(idim=opt.edim,
                                    cdim=opt.hdim,
                                    N=opt.N,
                                    M=opt.M,
                                    drop=opt.dropout)

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

    fname_dict = {}
    fanaly_name = opt.fanaly.split('/')[-1].split('.')[0]
    if opt.enc_type in MANNS:
        fenc = os.path.join('..', os.path.join(ANALYSIS, '%s-%s-%s.txt' % (opt.task, fanaly_name, opt.enc_type)))
        fname_dict['f' + opt.enc_type] = fenc
    flstm = os.path.join('..', os.path.join(ANALYSIS, '%s-%s-%s_lstm.txt' % (opt.task, fanaly_name, opt.enc_type)))
    fname_dict['flstm'] = flstm

    with analy(model.encoder, fname_dict):
        test_analy(model, res_iters['analy_iter'], opt.enc_type)