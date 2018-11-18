from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=300)
    group.add_argument('-hdim', type=int, default=300)
    group.add_argument('-odim', type=int, default=8)
    group.add_argument('-dropout', type=float, default=0.3)

    group.add_argument('-fix_emb', default=False, action='store_true')
    # group.add_argument('-emb_type', type=str, default='dense')
    group.add_argument('-emb_type', type=str, default='glove.840B.300d')
    # group.add_argument('-N', type=int, default=5)
    group.add_argument('-N', type=int, default=10)
    # group.add_argument('-N', type=int, default=20)
    # group.add_argument('-N', type=int, default=30)
    group.add_argument('-M', type=int, default=300)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)

    group.add_argument('-ftrain', type=str,
                       default=os.path.join(SR,
                                            'train.txt'))
    group.add_argument('-fvalid', type=str,
                       default=os.path.join(SR,
                                            'test.txt'))
    group.add_argument('-ftest', type=str,
                       default=os.path.join(SR,
                                            'test.txt'))

    group.add_argument('-fload', type=str, default=None)
    group.add_argument('-bsz', type=int, default=32)
    # group.add_argument('-bsz', type=int, default=256)
    # group.add_argument('-bsz', type=int, default=64)
    # group.add_argument('-lr', type=float, default=1e-4)
    # group.add_argument('-lr', type=float, default=1e-3)
    group.add_argument('-lr', type=float, default=0.00025)
    # group.add_argument('-lr', type=float, default=5e-4)
    # group.add_argument('-lr', type=float, default=5e-5)
    group.add_argument('-wdecay', type=float, default=1.2e-6)
    # group.add_argument('-wdecay', type=float, default=0.0001)
    # group.add_argument('-lm_coef', type=float, default=1)
    group.add_argument('-lm_coef', type=float, default=0)
    group.add_argument('-gclip', type=float, default=15)
    # group.add_argument('-gclip', type=float, default=1)
    group.add_argument('-seq_len_max', type=int, default=None)
    group.add_argument('-patience', type=int, default=10000)