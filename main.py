import nets
from macros import *
import torch
import utils
import opts
import argparse
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tasks import pattern, flang, agreement
import crash_on_ipy

if __name__ == '__main__':
    parser = argparse. \
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    utils.init_seed(opt.seed)

    build_iters = None
    train = None
    Model = None
    criterion = None

    if opt.task == 'pattern':
        build_iters = pattern.build_iters
        train = pattern.train
        Model = pattern.Model

    if opt.task == 'flang':
        build_iters = flang.build_iters
        train = flang.train
        Model = flang.Model

    if opt.task == 'agreement':
        build_iters = agreement.build_iters
        train = agreement.train
        Model = agreement.Model

    param_iter = {'ftrain': opt.ftrain,
                  'fvalid': opt.fvalid,
                  'bsz': opt.bsz,
                  'device': opt.gpu,
                  'sub_task': opt.sub_task,
                  'num_batches_train': opt.num_batches_train,
                  'num_batches_valid': opt.num_batches_valid,
                  'min_len_train': opt.min_len_train,
                  'min_len_valid': opt.min_len_valid,
                  'max_len_train': opt.max_len_train,
                  'max_len_valid': opt.max_len_valid,
                  'repeat_min_train': opt.repeat_min_train,
                  'repeat_max_train': opt.repeat_max_train,
                  'repeat_min_valid': opt.repeat_min_valid,
                  'repeat_max_valid': opt.repeat_max_valid,
                  'seq_width': opt.seq_width}

    res_iters = build_iters(param_iter)

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
                                  dropout=opt.dropout)
    if opt.enc_type == 'sarnn':
        encoder = nets.EncoderSARNN(idim=opt.edim,
                                    cdim=opt.hdim,
                                    N=opt.N,
                                    M=opt.M,
                                    idrop=opt.idrop,
                                    odrop=opt.odrop)
    if opt.enc_type == 'lstm':
        encoder = nets.EncoderLSTM(idim=opt.edim,
                                    cdim=opt.hdim,
                                    N=opt.N,
                                    M=opt.M,
                                   dropout=opt.dropout)

    if opt.enc_type == 'alstm':
        encoder = nets.EncoderALSTM(idim=opt.edim,
                                    cdim=opt.hdim,
                                    N=opt.N,
                                    M=opt.M,
                                    dropout=opt.dropout)

    model = None
    if embedding is None:
        model = Model(encoder, opt.odim).to(device)
    else:
        model = Model(encoder, embedding, opt.edrop).to(device)
    utils.init_model(model)

    if opt.fload is not None:
        model_fname = opt.fload
        location = {'cuda:' + str(opt.gpu): 'cuda:' + str(opt.gpu)} if opt.gpu != -1 else 'cpu'
        model_path = os.path.join(RES, model_fname)
        model_dict = torch.load(model_path, map_location=location)
        model.load_state_dict(model_dict)
        print('Loaded from ' + model_path)

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

