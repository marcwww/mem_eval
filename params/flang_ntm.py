from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=50)
    group.add_argument('-hdim', type=int, default=50)
    group.add_argument('-odim', type=int, default=8)
    group.add_argument('-dropout', type=float, default=0.1)

    group.add_argument('-fix_emb', default=False, action='store_true')
    group.add_argument('-emb_type', type=str, default='dense')
    group.add_argument('-N', type=int, default=20)
    group.add_argument('-M', type=int, default=50)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)
    group.add_argument('-ftrain', type=str,
                       default=os.path.join(FLANG,
                                            'expr-ntrain601-ntest500-dbound4-dtest4-e1.train.txt'))
    group.add_argument('-fvalid', type=str,
                       default=os.path.join(FLANG,
                                            'expr-ntrain601-ntest500-dbound4-dtest4-e1.valid.txt'))
    group.add_argument('-ftest1', type=str,
                       default=os.path.join(FLANG,
                                            'expr-ntrain10000-ntest1000-dbound4-dtest5-e1.test.txt'))
    group.add_argument('-ftest2', type=str,
                       default=os.path.join(FLANG,
                                            'expr-ntrain10000-ntest1000-dbound4-dtest6-e1.test.txt'))

    group.add_argument('-fload', type=str, default='flang-overall-ntm-1538570204.model')
    group.add_argument('-bsz', type=int, default=32)
    group.add_argument('-lr', type=float, default=5e-3)
    # group.add_argument('-lr', type=float, default=5e-5)
    group.add_argument('-wdecay', type=float, default=1.2e-6)
    # group.add_argument('-wdecay', type=float, default=0.0001)
    # group.add_argument('-lm_coef', type=float, default=1)
    # group.add_argument('-lm_coef', type=float, default=0.5)
    group.add_argument('-lm_coef', type=float, default=0)
    group.add_argument('-gclip', type=float, default=5)