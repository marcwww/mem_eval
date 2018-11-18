from macros import *
from . import flang_srnn, flang_lstm, flang_alstm, flang_sarnn, flang_ntm, \
    polysemy_lstm, polysemy_alstm, polysemy_sarnn, polysemy_ntm, \
    listops_lstm, listops_srnn, listops_alstm, listops_sarnn, listops_ntm, \
    feval_lstm, feval_alstm, feval_sarnn, feval_srnn, feval_ntm, \
    flang_topnn, \
    sst2_srnn, sst2_sarnn, sst2_lstm, sst2_vecave, \
    sst5_srnn, sst5_lstm, sst5_sarnn, sst5_alstm, sst5_ntm, \
    sr_lstm, sr_alstm, sr_sarnn, sr_ntm, sr_srnn

def general_opts(parser):
    group = parser.add_argument_group('general')
    # group.add_argument('-enc_type', type=str, default='ntm')
    # group.add_argument('-enc_type', type=str, default='srnn')
    # group.add_argument('-enc_type', type=str, default='topnn')
    group.add_argument('-enc_type', type=str, default='sarnn')
    # group.add_argument('-enc_type', type=str, default='vecave')
    # group.add_argument('-enc_type', type=str, default='lstm')
    # group.add_argument('-enc_type', type=str, default='alstm')

    # group.add_argument('-task', type=str, default='listops')
    # group.add_argument('-task', type=str, default='flang')
    # group.add_argument('-task', type=str, default='sst2')
    # group.add_argument('-task', type=str, default='sst5')
    group.add_argument('-task', type=str, default='sr')
    # group.add_argument('-task', type=str, default='feval')
    # group.add_argument('-task', type=str, default='polysemy')
    group.add_argument('-sub_task', type=str, default='overall')

    group.add_argument('-nepoch', type=int, default=100)
    group.add_argument('-save_per', type=int, default=2)
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-test_level', type=int, default=1)

def select_opt(opt, parser):

    if opt.task == 'flang' and opt.enc_type == 'lstm':
        flang_lstm.model_opts(parser)
        flang_lstm.train_opts(parser)

    elif opt.task == 'flang' and opt.enc_type == 'srnn':
        flang_srnn.model_opts(parser)
        flang_srnn.train_opts(parser)

    elif opt.task == 'flang' and opt.enc_type == 'alstm':
        flang_alstm.model_opts(parser)
        flang_alstm.train_opts(parser)

    elif opt.task == 'flang' and opt.enc_type == 'sarnn':
        flang_sarnn.model_opts(parser)
        flang_sarnn.train_opts(parser)

    elif opt.task == 'flang' and opt.enc_type == 'ntm':
        flang_ntm.model_opts(parser)
        flang_ntm.train_opts(parser)

    elif opt.task == 'flang' and opt.enc_type == 'topnn':
        flang_topnn.model_opts(parser)
        flang_topnn.train_opts(parser)

    elif opt.task == 'polysemy' and opt.enc_type == 'lstm':
        polysemy_lstm.model_opts(parser)
        polysemy_lstm.train_opts(parser)

    elif opt.task == 'polysemy' and opt.enc_type == 'alstm':
        polysemy_alstm.model_opts(parser)
        polysemy_alstm.train_opts(parser)

    elif opt.task == 'polysemy' and opt.enc_type == 'sarnn':
        polysemy_sarnn.model_opts(parser)
        polysemy_sarnn.train_opts(parser)

    elif opt.task == 'polysemy' and opt.enc_type == 'ntm':
        polysemy_ntm.model_opts(parser)
        polysemy_ntm.train_opts(parser)

    elif opt.task == 'listops' and opt.enc_type == 'srnn':
        listops_srnn.model_opts(parser)
        listops_srnn.train_opts(parser)

    elif opt.task == 'listops' and opt.enc_type == 'lstm':
        listops_lstm.model_opts(parser)
        listops_lstm.train_opts(parser)

    elif opt.task == 'listops' and opt.enc_type == 'alstm':
        listops_alstm.model_opts(parser)
        listops_alstm.train_opts(parser)

    elif opt.task == 'listops' and opt.enc_type == 'sarnn':
        listops_sarnn.model_opts(parser)
        listops_sarnn.train_opts(parser)

    elif opt.task == 'listops' and opt.enc_type == 'ntm':
        listops_ntm.model_opts(parser)
        listops_ntm.train_opts(parser)

    elif opt.task == 'feval' and opt.enc_type == 'lstm':
        feval_lstm.model_opts(parser)
        feval_lstm.train_opts(parser)

    elif opt.task == 'feval' and opt.enc_type == 'alstm':
        feval_alstm.model_opts(parser)
        feval_alstm.train_opts(parser)

    elif opt.task == 'feval' and opt.enc_type == 'sarnn':
        feval_sarnn.model_opts(parser)
        feval_sarnn.train_opts(parser)

    elif opt.task == 'feval' and opt.enc_type == 'srnn':
        feval_srnn.model_opts(parser)
        feval_srnn.train_opts(parser)

    elif opt.task == 'feval' and opt.enc_type == 'ntm':
        feval_ntm.model_opts(parser)
        feval_ntm.train_opts(parser)

    elif opt.task == 'sst2' and opt.enc_type == 'srnn':
        sst2_srnn.model_opts(parser)
        sst2_srnn.train_opts(parser)

    elif opt.task == 'sst2' and opt.enc_type == 'sarnn':
        sst2_sarnn.model_opts(parser)
        sst2_sarnn.train_opts(parser)

    elif opt.task == 'sst2' and opt.enc_type == 'lstm':
        sst2_lstm.model_opts(parser)
        sst2_lstm.train_opts(parser)

    elif opt.task == 'sst2' and opt.enc_type == 'vecave':
        sst2_vecave.model_opts(parser)
        sst2_vecave.train_opts(parser)

    elif opt.task == 'sst5' and opt.enc_type == 'srnn':
        sst5_srnn.model_opts(parser)
        sst5_srnn.train_opts(parser)

    elif opt.task == 'sst5' and opt.enc_type == 'lstm':
        sst5_lstm.model_opts(parser)
        sst5_lstm.train_opts(parser)

    elif opt.task == 'sst5' and opt.enc_type == 'sarnn':
        sst5_sarnn.model_opts(parser)
        sst5_sarnn.train_opts(parser)

    elif opt.task == 'sst5' and opt.enc_type == 'alstm':
        sst5_alstm.model_opts(parser)
        sst5_alstm.train_opts(parser)

    elif opt.task == 'sst5' and opt.enc_type == 'ntm':
        sst5_ntm.model_opts(parser)
        sst5_ntm.train_opts(parser)

    elif opt.task == 'sr' and opt.enc_type == 'lstm':
        sr_lstm.model_opts(parser)
        sr_lstm.train_opts(parser)

    elif opt.task == 'sr' and opt.enc_type == 'alstm':
        sr_alstm.model_opts(parser)
        sr_alstm.train_opts(parser)

    elif opt.task == 'sr' and opt.enc_type == 'sarnn':
        sr_sarnn.model_opts(parser)
        sr_sarnn.train_opts(parser)

    elif opt.task == 'sr' and opt.enc_type == 'srnn':
        sr_srnn.model_opts(parser)
        sr_srnn.train_opts(parser)

    elif opt.task == 'sr' and opt.enc_type == 'ntm':
        sr_ntm.model_opts(parser)
        sr_ntm.train_opts(parser)

    else:
        raise ModuleNotFoundError

    return parser

