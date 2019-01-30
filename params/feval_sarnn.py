from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=100)
    group.add_argument('-hdim', type=int, default=100)
    group.add_argument('-odim', type=int, default=8)
    group.add_argument('-dropout', type=float, default=0)
    # group.add_argument('-dropout', type=float, default=0.1)

    group.add_argument('-fix_emb', default=False, action='store_true')
    group.add_argument('-emb_type', type=str, default='dense')
    # group.add_argument('-N', type=int, default=2)
    group.add_argument('-N', type=int, default=5)
    # group.add_argument('-N', type=int, default=30)
    # group.add_argument('-M', type=int, default=20)
    group.add_argument('-M', type=int, default=100)
    group.add_argument('-read_first', action='store_true', default=False)
    # group.add_argument('-read_first', action='store_true', default=True)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)

    # group.add_argument('-ftrain', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'train_d30.tsv'))
    # group.add_argument('-fvalid', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'valid_d30.tsv'))
    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30_ef.tsv'))
    group.add_argument('-ftrain', type=str,
                       default=os.path.join(FEVAL,
                                            'train.pa1.mmc3.txt'))
    group.add_argument('-fvalid', type=str,
                       default=os.path.join(FEVAL,
                                            'valid.pa1.mmc3.txt'))
    group.add_argument('-ftest', type=str,
                       default=os.path.join(FEVAL,
                                            'test.pa1.mmc3.txt'))

    # group.add_argument('-ftrain', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'train_d30.ne.small.tsv'))
    # group.add_argument('-fvalid', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'valid_d30.ne.small.tsv'))
    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30.ne.small.tsv'))

    # group.add_argument('-ftrain', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'train_d30.pn1.tsv'))
    # group.add_argument('-fvalid', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'valid_d30.pn1.tsv'))
    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30.pn1.tsv'))

    # group.add_argument('-ftrain', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'train_d30.pn1.more.tsv'))
    # group.add_argument('-fvalid', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'valid_d30.pn1.more.tsv'))
    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30.pn1.more.tsv'))

    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30.pn1.easy.tsv'))
    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30.pn1.hard.tsv'))


    # group.add_argument('-ftrain', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'train_d30.parenthesis.tsv'))
    # group.add_argument('-fvalid', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'valid_d30.parenthesis.tsv'))
    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30.tsv'))
    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30_mono.tsv'))
    # group.add_argument('-ftest', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'test_d30.parenthesis.tsv'))



    # group.add_argument('-fanaly', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'analy_d5.tsv'))
    # group.add_argument('-fanaly', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'analy_d7.tsv'))
    # group.add_argument('-fanaly', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'analy_d10.tsv'))
    # group.add_argument('-fanaly', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'analy_d10_mono.tsv'))
    # group.add_argument('-fanaly', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'analy_d23_ne10.tsv'))
    # group.add_argument('-fanaly', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'change_numerals.txt'))

    group.add_argument('-fanaly', type=str,
                       default=os.path.join(FEVAL,
                                            'analysis.pa1.mmc3.txt'))

    # group.add_argument('-fanaly', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'analy_clustering.tsv'))
    # group.add_argument('-fanaly', type=str,
    #                    default=os.path.join(FEVAL,
    #                                         'analy_clustering_plus_minus.tsv'))


    # group.add_argument('-fload', type=str, default='feval-overall-sarnn-1540522028.model')
    # group.add_argument('-fload', type=str, default='feval-overall-sarnn-1543548025.model')
    group.add_argument('-fload', type=str, default='feval-overall-sarnn-1544106523.model')
    # group.add_argument('-fload', type=str, default='feval-overall-sarnn-1542679900.model')
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