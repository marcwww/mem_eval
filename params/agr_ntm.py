from macros import *


def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=650)
    group.add_argument('-hdim', type=int, default=650)
    group.add_argument('-odim', type=int, default=50)
    # group.add_argument('-dropout', type=float, default=0)
    group.add_argument('-dropout', type=float, default=0.4)

    group.add_argument('-fix_emb', default=False, action='store_true')
    group.add_argument('-emb_type', type=str, default='dense')
    # group.add_argument('-N', type=int, default=2)
    # group.add_argument('-N', type=int, default=5)
    group.add_argument('-N', type=int, default=10)
    # group.add_argument('-N', type=int, default=30)
    # group.add_argument('-M', type=int, default=20)
    group.add_argument('-M', type=int, default=650)
    # group.add_argument('-read_first', action='store_true', default=False)
    group.add_argument('-read_first', action='store_true', default=True)


def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)

    group.add_argument('-ftrain', type=str,
                       default=os.path.join(AGR,
                                            'train.tsv'))
    group.add_argument('-fvalid', type=str,
                       default=os.path.join(AGR,
                                            'valid.tsv'))
    group.add_argument('-ftest', type=str,
                       default=os.path.join(AGR,
                                            'test.tsv'))

    group.add_argument('-fanaly', type=str,
                       default=os.path.join(FEVAL,
                                            'analysis.pa1.mmc3.txt'))

    # group.add_argument('-fload', type=str, default='feval-overall-sarnn-1548860850.model')
    # group.add_argument('-fload', type=str, default='feval-overall-sarnn-1540522028.model')
    # group.add_argument('-fload', type=str, default='feval-overall-sarnn-1543548025.model')
    # group.add_argument('-fload', type=str, default='feval-overall-sarnn-1544106523.model')
    group.add_argument('-fload', type=str, default='feval-overall-sarnn-1542679900.model')
    # group.add_argument('-fload', type=str, default='feval-overall-sarnn-1545555737.model')

    group.add_argument('-bsz', type=int, default=32)
    # group.add_argument('-bsz', type=int, default=128)
    # group.add_argument('-bsz', type=int, default=64)
    group.add_argument('-lr', type=float, default=1e-3)
    # group.add_argument('-lr', type=float, default=5e-4)
    # group.add_argument('-lr', type=float, default=5e-5)
    group.add_argument('-wdecay', type=float, default=1.2e-6)
    # group.add_argument('-wdecay', type=float, default=0.0001)
    # group.add_argument('-lm_coef', type=float, default=1)
    # group.add_argument('-lm_coef', type=float, default=0.5)
    group.add_argument('-lm_coef', type=float, default=0)
    group.add_argument('-gclip', type=float, default=5)
    # group.add_argument('-gclip', type=float, default=1)
    group.add_argument('-seq_len_max', type=int, default=None)
    group.add_argument('-patience', type=int, default=10000)
