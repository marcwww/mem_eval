from macros import *
from . import flang_lstm, flang_alstm, flang_sarnn, flang_ntm, \
    polysemy_lstm, polysemy_alstm, polysemy_sarnn, polysemy_ntm

def general_opts(parser):
    group = parser.add_argument_group('general')
    # group.add_argument('-enc_type', type=str, default='ntm')
    # group.add_argument('-enc_type', type=str, default='sarnn')
    group.add_argument('-enc_type', type=str, default='lstm')
    # group.add_argument('-enc_type', type=str, default='alstm')

    # group.add_argument('-task', type=str, default='flang')
    group.add_argument('-task', type=str, default='polysemy')
    group.add_argument('-sub_task', type=str, default='overall')

    group.add_argument('-nepoch', type=int, default=30)
    group.add_argument('-save_per', type=int, default=2)
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-test_level', type=int, default=1)

def select_opt(opt, parser):
    if opt.task == 'flang' and opt.enc_type == 'lstm':
        flang_lstm.model_opts(parser)
        flang_lstm.train_opts(parser)

    elif opt.task == 'flang' and opt.enc_type == 'alstm':
        flang_alstm.model_opts(parser)
        flang_alstm.train_opts(parser)

    elif opt.task == 'flang' and opt.enc_type == 'sarnn':
        flang_sarnn.model_opts(parser)
        flang_sarnn.train_opts(parser)

    elif opt.task == 'flang' and opt.enc_type == 'ntm':
        flang_ntm.model_opts(parser)
        flang_ntm.train_opts(parser)

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

    else:
        raise ModuleNotFoundError

    return parser

