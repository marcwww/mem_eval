import os
ACTS = {'push':0,
        'pop':1,
        'noop':2
        }
PAD = '<pad>'
UNK = '<unk>'
EOS = '<eos>'
SOS = '<sos>'
SEP = '<sep>'
DATA = 'data/'
ANALYSIS = 'analysis/'
RTE = os.path.join(DATA, 'RTE')
RES = 'res/'
PAD_DS = 0
PROP_ENTAIL = os.path.join(DATA, 'prop-entail')
REWRITING = os.path.join(DATA, 'rewriting')
SCAN = os.path.join(DATA, 'scan')
PATTERN = os.path.join(DATA, 'pattern')
LISTOPS = os.path.join(DATA, 'listops')
FLANG = os.path.join(DATA, 'flang')
FEVAL = os.path.join(DATA, 'feval')
AGREE = os.path.join(DATA, 'agreement')
POLYSEMY = os.path.join(DATA, 'polysemy')

MANNS = ['sarnn', 'alstm', 'ntm']
INDICATOR = '<indicator>'

T_REDUCE = 1
T_SHIFT = 0