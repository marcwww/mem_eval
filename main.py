import nets
from macros import *
import torch
import utils
from params import opts
import argparse
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tasks import polysemy, flang
import crash_on_ipy

if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)
    opts.general_opts(parser)
    opt = parser.parse_args()

    parser = opts.select_opt(opt, parser)
    opt = parser.parse_args()

    utils.init_seed(opt.seed)

    build_iters = None
    train = None
    Model = None
    criterion = None

    if opt.task == 'polysemy':
        build_iters = polysemy.build_iters
        train = polysemy.train
        Model = polysemy.Model

    if opt.task == 'flang':
        build_iters = flang.build_iters
        train = flang.train
        Model = flang.Model

    res_iters = build_iters(ftrain=opt.ftrain,
                            fvalid=opt.fvalid,
                            bsz=opt.bsz,
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
                                N=opt.N,
                                M=opt.M,
                                drop=opt.dropout)

    if opt.enc_type == 'alstm':
        encoder = nets.EncoderALSTM(idim=opt.edim,
                                cdim=opt.hdim,
                                N=opt.N,
                                M=opt.M,
                                drop=opt.dropout)

    model = None
    if embedding is None:
        model = Model(encoder, opt.odim).to(device)
    else:
        model = Model(encoder, embedding, opt.dropout).to(device)
    utils.init_model(model)

    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                           lr=opt.lr,
                           weight_decay=opt.wdecay)
    # optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
    #                        lr=opt.lr,
    #                        weight_decay=opt.wdecay)
    # optimizer = optim.RMSprop(params=filter(lambda p: p.requires_grad, model.parameters()),
    #                           momentum=0.9,
    #                           alpha=0.95,
    #                           lr=opt.lr,
    #                           weight_decay=opt.wdecay)

    scheduler = ReduceLROnPlateau(optimizer, mode='max',factor=0.5, patience=10000)

    param_str = utils.param_str(opt)
    for key, val in param_str.items():
        print(str(key) + ': ' + str(val))
    train(model, res_iters, opt, optimizer, scheduler)

