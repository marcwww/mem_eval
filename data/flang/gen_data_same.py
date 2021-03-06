import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils
import random
from collections import defaultdict
import copy
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score
import crash_on_ipy

# E -> I | (E * E) | (E / E) | (E + E) | (E - E)
# I -> num
ops = ['*', '/', '+', '-']

class CorrExprGenerator(object):

    def __init__(self, d_bound):
        self.d_bound = d_bound

    def gen(self):
        expr = None
        self.d_general = 0
        while self.d_general != self.d_bound:
            self.d_general = 0
            expr = self.prod_E(0)
        return expr

    def prod_E(self, d):
        branch = np.random.choice(5)
        if d > self.d_general:
            self.d_general = d
        if d >= self.d_bound:
            return self.prod_I()

        if branch == 4:
            return self.prod_I()
        else:
            return ' '.join(['(', self.prod_E(d + 1), ops[branch], self.prod_E(d + 1), ')'])

    def prod_I(self):
        num = np.random.choice(10)
        return str(num)

class IncorrExprGenerator(object):
    def __init__(self, d_bound, e_bound, e_type):
        self.d_bound = d_bound
        self.e_bound = e_bound
        self.e_type = e_type

    def gen(self):
        expr = None
        self.e = 0
        self.d_general = 0
        while self.e != self.e_bound or self.d_general != self.d_bound:
            self.e = 0
            self.d_general = 0
            expr = self.prod_E(0)
        return expr, self.error

    def prod_I(self):
        num = np.random.choice(10)
        return str(num)

    def prod_E(self, d):
        if d > self.d_general:
            self.d_general = d
        if d >= self.d_bound:
            return self.prod_I()

        make_err = False
        if np.random.choice(self.d_bound ** 3) in range(self.e_bound)\
                and self.e < self.e_bound:
            make_err = True
            self.e += 1

        if make_err:
            # means it cannot go into prod_I()
            branch = np.random.choice(4)
            if self.e_type == 'exchange':
                error = np.random.choice(5)
                self.error = error
                if error == 0:
                    return ' '.join([self.prod_E(d + 1), '(', ops[branch], self.prod_E(d + 1), ')'])
                elif error == 1:
                    return ' '.join(['(', ops[branch], self.prod_E(d + 1), self.prod_E(d + 1), ')'])
                elif error == 2:
                    return ' '.join(['(', self.prod_E(d + 1), self.prod_E(d + 1), ops[branch], ')'])
                elif error == 3:
                    return ' '.join(['(', self.prod_E(d + 1), ops[branch], ')', self.prod_E(d + 1)])
                else:
                    return ' '.join([')', self.prod_E(d + 1), ops[branch], self.prod_E(d + 1), '('])

            if self.e_type == 'omit':
                error = np.random.choice(3)
                self.error = error
                if error == 0:
                    return ' '.join([self.prod_E(d + 1), ops[branch], self.prod_E(d + 1), ')'])
                elif error == 1:
                    return ' '.join(['(', self.prod_E(d + 1), self.prod_E(d + 1), ')'])
                else:
                    return ' '.join(['(', self.prod_E(d + 1), ops[branch], self.prod_E(d + 1)])

            if self.e_type == 'redun':
                error = np.random.choice(3)
                self.error = error
                if error == 0:
                    return ' '.join(['(', '(', self.prod_E(d + 1), ops[branch], self.prod_E(d + 1), ')'])
                elif error == 1:
                    return ' '.join(['(', self.prod_E(d + 1), ops[branch], ops[branch], self.prod_E(d + 1), ')'])
                elif error == 2:
                    return ' '.join(['(', self.prod_E(d + 1), ops[branch], self.prod_E(d + 1), ')', ')'])

        else:
            branch = np.random.choice(5)
            if branch == 4:
                return self.prod_I()
            else:
                return ' '.join(['(', self.prod_E(d + 1), ops[branch], self.prod_E(d + 1), ')'])

def gen_data(num_train, num_valid, num_test, d_max, e):
    samples = []
    num_total = num_train + num_valid + num_test

    p = 1/np.array(range(1, d_max+1))
    p = p / p.sum()
    # p = np.array([1/d_max for _ in range(1, d_max+1)])

    ds = defaultdict(int)
    for _ in range(num_total):
        d = np.random.choice(a=range(1, d_max+1), size=1, p=p)
        ds[d[0]] += 1

    for d in ds.keys():
        num = int(ds[d]/3)

        exchange = gen_list_partial(num, d, e, 'exchange')
        samples.extend(exchange)

        omit = gen_list_partial(num, d, e, 'omit')
        samples.extend(omit)

        redun = gen_list_partial(num, d, e, 'redun')
        samples.extend(redun)

    samples = random.sample(samples, k=len(samples))
    train = samples[:num_train]
    valid = samples[num_train: num_train + num_valid]
    test = samples[num_train + num_valid:]

    for data_type, data in zip(['train','valid','test'],[train, valid, test]):
        with open(('expr-ntrain%d-nvalid%d-ntest%d-dmax%d-e%d.%s.txt' %
                   (num_train, num_valid, num_test, d_max, e, data_type)), 'w') as f:
            for line in data:
                f.write(line)

def gen_list_partial(num, d_bound, e_bound, e_type):
    igen = IncorrExprGenerator(d_bound, e_bound, e_type)
    cgen = CorrExprGenerator(d_bound)

    e2s = {'exchange':'0', 'omit':'1', 'redun':'2'}

    res = []
    for i in range(num):
        correct = np.random.choice(2)
        if correct == 1:
            expr = cgen.gen()
        else:
            expr, _ = igen.gen()

        res.append('\t'.join([expr, str(correct), str(d_bound), e2s[e_type]]) + '\n')

    return res

def gen_data_partial(num, d_bound, e_bound, e_type, dataset_type):
    igen = IncorrExprGenerator(d_bound, e_bound, e_type)
    cgen = CorrExprGenerator(d_bound)

    with open(('expr-n%d-d%d-e%d-%s.%s.txt' %
               (num, d_bound, e_bound, e_type, dataset_type)), 'w') as f:

        for i in range(num):
            correct = np.random.choice(2)
            if correct == 1:
                expr = cgen.gen()
                error = -1
            else:
                expr, error = igen.gen()
            f.write('\t'.join([expr, str(correct), str(error)]) + '\n')

if __name__ == '__main__':

    # gen_data(200, 500, 4, 4, 1)
    # gen_data(10001, 2000, 1000, 10, 1)
    gen_data(10000, 2000, 1000, 10, 1)
    # gen_data(2001, 2000, 1000, 10, 1)
    # gen_data(5000, 2000, 1000, 10, 1)

    # # exchange:
    # gen_data_partial(10000, 2, 1, 'exchange','train')
    # gen_data_partial(10000, 3, 1, 'exchange','train')
    #
    # gen_data_partial(1000, 4, 1, 'exchange', 'valid')
    # gen_data_partial(1000, 5, 1, 'exchange', 'valid')
    #
    # # omit:
    # gen_data_partial(10000, 2, 1, 'omit','train')
    # gen_data_partial(10000, 3, 1, 'omit','train')
    #
    # gen_data_partial(1000, 4, 1, 'omit', 'valid')
    # gen_data_partial(1000, 5, 1, 'omit', 'valid')
    #
    # # redundant:
    # gen_data_partial(10000, 2, 1, 'redun', 'train')
    # gen_data_partial(10000, 3, 1, 'redun', 'train')
    #
    # gen_data_partial(1000, 4, 1, 'redun', 'valid')
    # gen_data_partial(1000, 5, 1, 'redun', 'valid')
