import torchtext
from macros import *
from torchtext.data import Dataset
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm
import numpy as np
import utils
import random
import copy
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score
import crash_on_ipy

class CorrExprGenerator(object):

    def __init__(self, d_bound):
        self.d_bound = d_bound

    def gen(self):
        self.d_general = 0
        return self.prod_E(0), self.d_general

    def prod_I(self):
        num = np.random.choice(10)
        return str(num)

    def prod_T(self, d):
        branch = np.random.choice(3)
        if d > self.d_general:
            self.d_general = d
        if d >= self.d_bound:
            return self.prod_I()

        if branch == 0:
            return ' '.join([self.prod_T(d + 1), '*', self.prod_F(d + 1)])
        elif branch == 1:
            return ' '.join([self.prod_T(d + 1), '/', self.prod_F(d + 1)])
        else:
            return self.prod_F(d + 1)

    def prod_E(self, d):
        branch = np.random.choice(3)
        if d > self.d_general:
            self.d_general = d
        if d >= self.d_bound:
            return self.prod_I()

        if branch == 0:
            return ' '.join([self.prod_E(d + 1), '+', self.prod_T(d + 1)])
        elif branch == 1:
            return ' '.join([self.prod_E(d + 1), '-', self.prod_T(d + 1)])
        else:
            return self.prod_T(d + 1)

    def prod_F(self, d):
        branch = np.random.choice(2)
        if d > self.d_general:
            self.d_general = d
        if d >= self.d_bound:
            return self.prod_I()

        if branch == 0:
            return self.prod_I()
        else:
            return ' '.join(['(', self.prod_E(d + 1), ')'])

class IncorrExprGenerator(object):
    def __init__(self, d_bound, e_bound):
        self.d_bound = d_bound
        self.e_bound = e_bound

    def gen(self):
        self.e = 0
        self.d_general = 0
        expr = None
        while self.e == 0:
            expr = self.prod_E(0)
        return expr, self.d_general, self.e

    def prod_I(self):
        num = np.random.choice(10)
        return str(num)

    def prod_T(self, d):
        branch = np.random.choice(3)
        error = np.random.choice(3)
        if d > self.d_general:
            self.d_general = d
        if d >= self.d_bound:
            return self.prod_I()
        if self.e >= self.e_bound:
            error = 0
        if error != 0 and branch != 2:
            # print('T', branch, error)
            self.e += 1

        if branch == 0:
            if error == 0:
                return ' '.join([self.prod_T(d + 1), '*', self.prod_F(d + 1)])
            elif error == 1:
                return ' '.join(['*', self.prod_T(d + 1), self.prod_F(d + 1)])
            else:
                return ' '.join([self.prod_T(d + 1), self.prod_F(d + 1), '*'])

        elif branch == 1:
            if error == 0:
                return ' '.join([self.prod_T(d + 1), '/', self.prod_F(d + 1)])
            elif error == 1:
                return ' '.join(['/', self.prod_T(d + 1) , self.prod_F(d + 1)])
            else:
                return ' '.join([self.prod_T(d + 1), self.prod_F(d + 1), '/'])

        else:
            return self.prod_F(d + 1)

    def prod_E(self, d):
        branch = np.random.choice(3)
        error = np.random.choice(3)
        if d > self.d_general:
            self.d_general = d
        if d >= self.d_bound:
            return self.prod_I()
        if self.e >= self.e_bound:
            error = 0
        if error != 0 and branch != 2:
            # print('E', branch, error)
            self.e += 1

        if branch == 0:
            if error == 0:
                return ' '.join([self.prod_E(d + 1), '+', self.prod_T(d + 1)])
            elif error == 1:
                return ' '.join(['+', self.prod_E(d + 1) , self.prod_T(d + 1)])
            else:
                return ' '.join([self.prod_E(d + 1), self.prod_T(d + 1), '+'])

        elif branch == 1:
            if error == 0:
                return ' '.join([self.prod_E(d + 1), '-', self.prod_T(d + 1)])
            elif error == 1:
                return ' '.join(['-', self.prod_E(d + 1) , self.prod_T(d + 1)])
            else:
                return ' '.join([self.prod_E(d + 1), self.prod_T(d + 1), '-'])

        else:
            return self.prod_T(d + 1)

    def prod_F(self, d):
        branch = np.random.choice(2)
        error = np.random.choice(4)
        if d > self.d_general:
            self.d_general = d
        if d >= self.d_bound:
            return self.prod_I()
        if self.e >= self.e_bound:
            error = 0
        if error != 0 and branch != 0:
            # print('F', branch, error)
            self.e += 1

        if branch == 0:
            return self.prod_I()
        else:
            if error == 0:
                return ' '.join(['(', self.prod_E(d + 1), ')'])
            elif error == 1:
                return ' '.join([self.prod_E(d + 1), '(' , ')'])
            elif error == 2:
                return ' '.join(['(', ')', self.prod_E(d + 1)])
            else:
                return ' '.join([ ')', self.prod_E(d + 1), '('])

def gen_data(num, d_bound, e_bound, dataset_type):
    igen = IncorrExprGenerator(d_bound, e_bound)
    cgen = CorrExprGenerator(d_bound)
    
    with open(('expr-num%d-dbound%d-ebound%d.%s.txt' %
               (num, d_bound, e_bound, dataset_type)), 'w') as f:

        for i in range(num):
            correct = np.random.choice(2)
            if correct == 1:
                expr, d = cgen.gen()
                e = 0
            else:
                expr, d, e  = igen.gen()
            f.write('\t'.join([expr, str(correct), str(d), str(e)]) + '\n')

if __name__ == '__main__':
    # for i in range(100):
    #     print(expr_correct(2))
    # igen = IncorrExprGenerator(5, 1)
    # cgen = CorrExprGenerator(5)
    # for i in range(100):
    #     print(cgen.gen())

    gen_data(10000, 3, 1, 'train')
    gen_data(1000, 3, 1, 'valid')