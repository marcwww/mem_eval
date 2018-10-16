from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=128)
    group.add_argument('-hdim', type=int, default=128)
    group.add_argument('-odim', type=int, default=8)
    group.add_argument('-dropout', type=float, default=0.2)

    group.add_argument('-fix_emb', default=False, action='store_true')
    group.add_argument('-emb_type', type=str, default='dense')
    group.add_argument('-N', type=int, default=10)
    group.add_argument('-M', type=int, default=128)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)
    group.add_argument('-ftrain', type=str,
                       default=os.path.join(LISTOPS,
                                            'train_d20s.tsv'))
    group.add_argument('-fvalid', type=str,
                       default=os.path.join(LISTOPS,
                                            'test_d20s.tsv'))
    group.add_argument('-ftest', type=str,
                       default=os.path.join(LISTOPS,
                                            'test_d20s.tsv'))

    group.add_argument('-fload', type=str, default=None)
    group.add_argument('-bsz', type=int, default=128)
    group.add_argument('-lr', type=float, default=1e-3)
    # group.add_argument('-lr', type=float, default=5e-5)
    group.add_argument('-wdecay', type=float, default=1e-5)
    # group.add_argument('-wdecay', type=float, default=0.0001)
    # group.add_argument('-lm_coef', type=float, default=1)
    # group.add_argument('-lm_coef', type=float, default=0.5)
    group.add_argument('-lm_coef', type=float, default=0)
    group.add_argument('-gclip', type=float, default=15)
    # group.add_argument('-gclip', type=float, default=1)
    group.add_argument('-seq_len_max', type=int, default=100)