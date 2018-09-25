from macros import *

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-edim', type=int, default=50)
    group.add_argument('-hdim', type=int, default=50)
    group.add_argument('-odim', type=int, default=8)
    group.add_argument('-dropout', type=float, default=0)
    group.add_argument('-fix_emb', default=False, action='store_true')
    # group.add_argument('-emb_type', type=str, default='one-hot')
    group.add_argument('-emb_type', type=str, default='dense')
    # group.add_argument('-enc_type', type=str, default='ntm')
    # group.add_argument('-enc_type', type=str, default='sarnn')
    group.add_argument('-enc_type', type=str, default='lstm')
    # group.add_argument('-enc_type', type=str, default='alstm')
    group.add_argument('-N', type=int, default=40)
    group.add_argument('-M', type=int, default=50)
    # group.add_argument('-T', type=int, default=1)
    group.add_argument('-T', type=int, default=2)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-seed', type=int, default=1000)
    # group.add_argument('-ftrain', type=str, default=os.path.join(PATTERN, 'copy_train1-10.pkl'))
    # group.add_argument('-fvalid', type=str, default=os.path.join(PATTERN, 'copy_valid11-20.pkl'))
    # group.add_argument('-ftrain', type=str,
    #                    default=os.path.join(FLANG,
    #                     'expr-num10000-dbound5-ebound1.train.txt'))
    # group.add_argument('-fvalid', type=str,
    #                    default=os.path.join(FLANG,
    #                     'expr-num1000-dbound10-ebound1.valid.txt'))
    group.add_argument('-ftrain', type=str,
                       default=os.path.join(AGREE,
                        'train.tsv'))
    group.add_argument('-fvalid', type=str,
                       default=os.path.join(AGREE,
                        'valid.tsv'))
    group.add_argument('-fload', type=str, default=None)
    group.add_argument('-bsz', type=int, default=64)
    group.add_argument('-min_freq', type=int, default=0)
    group.add_argument('-nepoch', type=int, default=30)
    group.add_argument('-save_per', type=int, default=2)
    # group.add_argument('-task', type=str, default='flang')
    # group.add_argument('-task', type=str, default='pattern')
    group.add_argument('-task', type=str, default='agreement')
    # group.add_argument('-sub_task', type=str, default='copy')
    # group.add_argument('-sub_task', type=str, default='mirror')
    # group.add_argument('-sub_task', type=str, default='expr')
    group.add_argument('-sub_task', type=str, default='overall')
    group.add_argument('-gpu', type=int, default=-1)
    group.add_argument('-lr', type=float, default=1e-3)
    group.add_argument('-wdecay', type=float, default=0)
    # group.add_argument('-lm_coef', type=float, default=1)
    group.add_argument('-lm_coef', type=float, default=0)

    # for copy-tasks
    group.add_argument('-seq_width', type=int, default=8)
    # group.add_argument('-sub_task', type=str, default='addprim_turn_left')
    # group.add_argument('-sub_task', type=str, default='addprim_jump')
    # group.add_argument('-sub_task', type=str, default='length')
    # group.add_argument('-sub_task', type=str, default='hard')
    # group.add_argument('-sub_task', type=str, default='simple')
    # group.add_argument('-sub_task', type=str, default='repeat')
    group.add_argument('-num_batches_train', type=int, default=5000)
    group.add_argument('-num_batches_valid', type=int, default=1000)
    # group.add_argument('-num_batches_train', type=int, default=500)
    # group.add_argument('-num_batches_valid', type=int, default=10)
    group.add_argument('-min_len_train', type=int, default=1)
    group.add_argument('-max_len_train', type=int, default=10)
    group.add_argument('-repeat_min_train', type=int, default=1)
    group.add_argument('-repeat_max_train', type=int, default=3)
    group.add_argument('-min_len_valid', type=int, default=1)
    group.add_argument('-max_len_valid', type=int, default=20)
    # group.add_argument('-min_len_valid', type=int, default=6)
    # group.add_argument('-max_len_valid', type=int, default=10)
    group.add_argument('-repeat_min_valid', type=int, default=1)
    group.add_argument('-repeat_max_valid', type=int, default=3)