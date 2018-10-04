import random
from macros import *

vocab = set()
context_map = {}
meaning_map = {}
alphabet = [chr(ord('A') + offset) for offset in range(26)]

for word in alphabet:
    vocab.add(word)
    context_map[word] = []
    for is_close in [0, 1]:
        for direction in ['l', 'r']:
            context = '%s%i%s' % (word, is_close, direction)
            vocab.add(context)
            context_map[word].append(context)
    for meaning, context in enumerate(context_map[word]):
        # meaning_map[context] = meaning
        meaning_map[context] = context[1:]

def analysis_meaning(noises, context, attractor):
    is_close_attractor, direction_attractor = attractor[1], attractor[2]
    is_close, direction = context[1], context[2]

    meaning = None
    if direction_attractor == direction:
        if is_close_attractor == '0':
            if direction == 'l':
                if noises[-1] != attractor:
                    meaning = meaning_map[attractor]
            else:
                # direction == 'r'
                if noises[0] != attractor:
                    meaning = meaning_map[attractor]
        else:
            # is_close_leader == '1'
            if direction == 'l':
                if noises[-1] == attractor:
                    meaning = meaning_map[attractor]
            else:
                # direction == 'r'
                if noises[0] == attractor:
                    meaning = meaning_map[attractor]

    return meaning

def gen_sample(dis, nattractors, num_other):

    word = random.choice(alphabet)
    context = random.choice(context_map[word])

    assert context[0] == word

    is_close, direction = context[1], context[2]
    vocab_except = vocab - set(context_map[word]) - {word}
    meaning = meaning_map[context]
    critical = None
    res = []

    # for critical part
    if is_close == '0':
        assert dis > 0
        noises = []
        attractors = []
        for _ in range(nattractors):
            attractor = random.choice(context_map[word])
            attractors.append(attractor)
            ipos = random.choice(range(len(noises)+1))
            noises.insert(ipos, attractor)

        if dis > nattractors:
            for _ in range(dis - nattractors):
                noise = random.choice(list(vocab_except))
                ipos = random.choice(range(len(noises) + 1))
                noises.insert(ipos, noise)

        critical = [context] + noises + [INDICATOR] + [word] if direction == 'l' else [INDICATOR] + [word] + noises + [context]

        attractors_sorted = sorted(attractors + [context], reverse=True)
        for attractor in attractors_sorted:
            res = analysis_meaning(noises, context, attractor)
            if res != None:
                meaning = res
                break
    else:
        # is_close == '1'
        critical = [context] + [INDICATOR] + [word] if direction == 'l' else [INDICATOR] + [word] + [context]
        nattractors = 0
        dis = 0

    # for other words
    res = critical
    for _ in range(num_other):
        direction = random.choice(['l', 'r'])
        word = random.choice(list(vocab_except))
        if direction == 'l':
            res.insert(0, word)
        else:
            res.append(word)

    return ' '.join(res), context, meaning, str(dis), str(nattractors), str(num_other)

def gen_samples(num, dis_min, dis_max, na_min, na_max, no_min, no_max):
    assert dis_min > 0
    samples = []
    for _ in range(num):
        dis = random.choice(range(dis_min, dis_max + 1))
        na = random.choice(range(na_min, na_max+1))
        no = random.choice(range(no_min, no_max+1))
        sample = gen_sample(dis, na, no)
        sample = '\t'.join(sample)
        samples.append(sample)

    return samples

def gen_dataset(gen_train=True, gen_valid=True, gen_test=True):
    num_train = 5000
    num_valid = 500
    num_test = 1000
    dis_min = 1
    dis_max = 10
    na_min = 0
    na_max = 7
    no_min = 8
    no_max = 12

    train = gen_samples(num_train, dis_min, dis_max, na_min, na_max, no_min, no_max)
    valid = gen_samples(num_valid, dis_min, dis_max, na_min, na_max, no_min, no_max)
    test = gen_samples(num_test, dis_min, dis_max, na_min, na_max, no_min, no_max)

    for type , samples in zip(['train', 'valid', 'test'], [train, valid, test]):
        if type == 'train' and not gen_train:
            continue
        if type == 'valid' and not gen_valid:
            continue
        if type == 'test' and not gen_test:
            continue

        with open('polysemy-ntrain%d-ntest%d-dis%d_%d-na%d_%d-no%d_%d.%s.txt'
                  % (num_train, num_test, dis_min, dis_max, na_min, na_max, no_min, no_max, type), 'w') as f:
            for sample in samples:
                f.write(sample + '\n')

gen_dataset(False, False, True)







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











