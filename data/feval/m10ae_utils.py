import numpy as np
import random
import matplotlib.pyplot as plt
from nltk import Tree
import os

MAX_DEPTH = 30
PROB_BRANCH = 0.4
VALUES = range(1, 10)
OPS_E = [0, 1]
OPS_T = [2, 3]
# OPS_T = [2]
NTYPES = ['e', 't']
OP_MAP = ['+', '-', '*', '/']
# OP_MAP = ['+', '-', '*']
PNESTED = 1

def MSU(nlst, modd=True):
    VALUES = range(1, 10)
    NUMERALS = list(map(str, VALUES)) + ['0']
    OP_MAP = ['+', '-', '*', '/']
    OPS = OP_MAP

    def m10eval(op, a0, a1):
        if op == '/':
            res = int(a0) // int(a1) if not modd else int(a0) % int(a1)
        else:
            res = eval(a0 + op + a1)
        res = res % 10
        return str(res)

    def reducible(mem, ninp):
        if len(mem) < 2:
            return False

        top, sec = mem[0], mem[1]
        if top in OPS and sec in NUMERALS:
            return True
        elif top in NUMERALS and sec[0] in NUMERALS and sec[1] in OPS:
            if sec[1] in ['+', '-'] and ninp not in ['*', '/']:
                return True
            if sec[1] in ['*', '/']:
                return True
            return False
        elif top == ')' and sec[0] == '(' and sec[1] in NUMERALS:
            return True
        elif top in NUMERALS and sec == '(' and ninp == ')':
            return True

        return False

    def reduce(mem):
        top = mem.pop(0)
        sec = mem.pop(0)

        if top in OPS and sec in NUMERALS:
            reduced = (sec, top)
        elif top in NUMERALS and sec[0] in NUMERALS and sec[1] in OPS:
            reduced = m10eval(sec[1], sec[0], top)
        elif top == ')' and sec[0] == '(' and sec[1] in NUMERALS:
            reduced = sec[1]
        elif top in NUMERALS and sec == '(':
            reduced = (sec, top)
        else:
            raise NotImplementedError

        return reduced

    stack = []
    reduce_lst = []
    msu = 0
    for t, n in enumerate(nlst):
        stack.insert(0, n)
        if len(stack) > msu:
            msu = len(stack)
        # reduce_lst.append(0)
        if t != len(nlst) - 1:
            ninp = nlst[t + 1]
        else:
            ninp = None

        r = 0
        while reducible(stack, ninp):
            stack.insert(0, reduce(stack))
            r += 1
            # reduce_lst.append(1)
        reduce_lst.append(r)

    return msu, stack[0]


def gen_expr(depth, ntype, pnested):
    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1 + 0.1

    if r > PROB_BRANCH:
        if ntype == 'f':
            value = random.choice(VALUES)
            return value
        elif ntype == 't':
            return gen_expr(depth + 1, 'f', pnested)
        elif ntype == 'e':
            return gen_expr(depth + 1, 't', pnested)
    else:
        if ntype == 'f':
            if pnested < PNESTED:
                e = gen_expr(depth + 2, 'e', pnested + 1)
                t = [['(', e], ')']
            else:
                value = random.choice(VALUES)
                return value
        elif ntype == 't':
            op = random.choice(OPS_T)
            v1, v2 = gen_expr(depth + 2, 't', pnested), gen_expr(depth + 1, 'f', pnested)
            t = [[v1, OP_MAP[op]], v2]
        else:
            assert ntype == 'e'
            op = random.choice(OPS_E)
            v1, v2 = gen_expr(depth + 2, 'e', pnested), gen_expr(depth + 1, 't', pnested)
            t = [[v1, OP_MAP[op]], v2]

    return t

def gen_tree():
    return gen_expr(0, 'e', 0)

def to_nlst(t):
    return list(filter(lambda x: x not in ['[', ']', ',', '\'', ' '], str(t)))

def to_value(t):
    if not isinstance(t, list):
        return t

    l = t[0]
    r = t[1]
    if l[1] in OP_MAP:
        v1, op = (to_value(l[0]), l[1])
        v2 = to_value(r)
        return eval(''.join([str(v1), op, str(v2)]))
    else:
        assert l[0] == '('
        v = to_value(l[1])
        return v

def to_sd(t):
    if not isinstance(t, list):
        d = []
        h = 0
    else:
        l, r = t
        d_l, h_l = to_sd(l)
        d_r, h_r = to_sd(r)
        h = max(h_l, h_r) + 1
        d = d_l + [h] + d_r

    return d, h

def to_value_sd(sd_lst, node_lst):
    if len(sd_lst) == 0:
        node = node_lst[0]
        v = node
    else:
        i = np.argmax(sd_lst)
        child_l, v_l = to_value_sd(sd_lst[:i], node_lst[:i+1])
        child_r, v_r = to_value_sd(sd_lst[i+1:], node_lst[i+1:])
        node = [child_l, child_r]
        if isinstance(v_l, list) and not isinstance(v_r, list):
            if v_l[1] in OP_MAP:
                v1, op = v_l[0], v_l[1]
                v2 = v_r
                v = str(eval(''.join([v1, op, v2])))
            else:
                assert v_l[0] == '(', str(v_l)
                v = str(v_l[1])
        else:
            assert not isinstance(v_l, list) and not isinstance(v_r, list)
            v = [v_l, v_r]

    return node, v

def ave_len(es):
    lens = []
    for e in es:
        lens.append(e[0].split())
    return np.average(lens), np.var(lens)

def plot_sd(t):
    sd_lst, _ = to_sd(t)
    nlst = to_nlst(t)
    fig, ax = plt.subplots()
    fig.dpi = 200
    plt.bar(np.arange(len(sd_lst)), sd_lst)
    plt.xticks(np.arange(len(nlst))-0.5, nlst)
    plt.grid(axis='y', linestyle='--')