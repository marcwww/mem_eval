import random
import numpy as np
from macros import *
import crash_on_ipy

vocab = set()
context_map = {}
meaning_map = {}
alphabet = [chr(ord('A') + offset) for offset in range(10)]
ncontext_types = 4

for word in alphabet:
    vocab.add(word)
    context_map[word] = []
    for i in range(ncontext_types):
        context = '%s%i' % (word, i)
        vocab.add(context)
        context_map[word].append(context)
    for meaning, context in enumerate(context_map[word]):
        # meaning_map[context] = context[1:]
        meaning_map[context] = context

def gen_sample(nattractors, num_other):
    assert nattractors > 0
    word = random.choice(alphabet)
    vocab_except = vocab - {word} - set(context_map[word])

    attractors = []
    seq = []
    for _ in range(nattractors):
        attractor = random.choice(context_map[word])
        attractors.append(attractor)
        ipos = random.choice(range(len(seq) + 1))
        seq.insert(ipos, attractor)

    for _ in range(num_other):
        oword = random.choice(list(vocab_except))
        ipos = random.choice(range(len(seq) + 1))
        seq.insert(ipos, oword)

    ipos = random.choice(range(len(seq) + 1))
    seq.insert(ipos, ' '.join([INDICATOR, word]))

    meaning = meaning_map[max(attractors)]

    return ' '.join(seq), meaning

def gen_samples(num, na_min, na_max, no_min, no_max):

    na_arr = np.array(range(na_min, na_max+1))
    p = na_arr
    # p = np.exp(p)
    p = p / np.sum(p)

    samples = []
    for _ in range(num):
        na = int(np.random.choice(a=na_arr, size=1, p=p))
        # na = random.choice(range(na_min, na_max+1))
        no = random.choice(range(no_min, no_max+1))
        seq, meaning = gen_sample(na, no)
        sample = '\t'.join([seq, meaning, str(na), str(no)])
        samples.append(sample)

    return samples

def gen_dataset(gen_train=True, gen_valid=True, gen_test=True):
    num_train = 5000
    num_valid = 2000
    num_test = 2000
    na_min = 1
    na_max = 5
    no_min = 8
    no_max = 12

    train = gen_samples(num_train, na_min, na_max, no_min, no_max)
    valid = gen_samples(num_valid, na_min, na_max, no_min, no_max)
    test = gen_samples(num_test, na_min, na_max, no_min, no_max)

    for type , samples in zip(['train', 'valid', 'test'], [train, valid, test]):
        if type == 'train' and not gen_train:
            continue
        if type == 'valid' and not gen_valid:
            continue
        if type == 'test' and not gen_test:
            continue

        with open('polysemy-re_long_tail-ntrain%d-ntest%d-ntypes%d-na%d_%d-no%d_%d.%s.txt'
                  % (num_train, num_test, ncontext_types, na_min, na_max, no_min, no_max, type), 'w') as f:
            for sample in samples:
                f.write(sample + '\n')

# gen_dataset(False, False, True)
gen_dataset(True, True, True)


# n = 0
# for _ in range(10000):
#     seq, context, meaning = gen_sample(1, 1, 10)
#     if context != meaning:
#         print(seq)
#         print(context, meaning)
#         n += 1
# print(n)
# print(seq)
# print(context, meaning)



# for _ in range(100):
#     seq, context, meaning = gen_sample(5, 2, 10)
#     print(seq)
#     print(context, meaning)











