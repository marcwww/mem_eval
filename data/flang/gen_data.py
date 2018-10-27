import numpy as np
import random

MAX_DEPTH = 20
PROB_LEAF = 0.25
VALUES = range(1, 10)
OPS_E = [0, 1]
OPS_T = [2, 3]
NTYPES = ['e', 't']
OP_MAP = ['+', '-', '*', '/']

def gen_expr(depth, ntype):

    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1

    if r > PROB_LEAF:
        if ntype == 'f':
            value = random.choice(VALUES)
            return value
        elif ntype == 't':
            return gen_expr(depth + 1, 'f')
        elif ntype == 'e':
            return gen_expr(depth + 1, 't')

    else:
        if ntype == 'e':
            op = random.choice(OPS_E)
            v1, v2 = gen_expr(depth + 2, 'e'), gen_expr(depth + 1, 't')
            t = ((v1, OP_MAP[op]), v2)
        elif ntype == 't':
            op = random.choice(OPS_T)
            v1, v2 = gen_expr(depth + 2, 't'), gen_expr(depth + 1, 'f')
            t = ((v1, OP_MAP[op]), v2)
        else:
            t = gen_expr(depth + 1, 'f')

    return t

def to_value(t):
    if not isinstance(t, tuple):
        return t

    l = t[0]
    r = t[1]
    v1, op = (to_value(l[0]), l[1])
    v2 = to_value(r)

    return eval(''.join([str(v1), op, str(v2)]))

def to_value_sd(sd_lst, node_lst):
    if len(sd_lst) == 0:
        node = node_lst[0]
        v = node
    else:
        i = np.argmax(sd_lst)
        # print(max(sd_lst))
        child_l, v_l = to_value_sd(sd_lst[:i], node_lst[:i+1])
        child_r, v_r = to_value_sd(sd_lst[i+1:], node_lst[i+1:])
        # print(child_l)
        # print(child_r)

        node = (child_l, child_r)
        if isinstance(v_l, tuple) and not isinstance(v_r, tuple):
            v1, op = v_l[0], v_l[1]
            v2 = v_r
            v = str(eval(''.join([v1, op, v2])))
        else:
            assert not isinstance(v_l, tuple) and not isinstance(v_r, tuple)
            v = (v_l, v_r)

    return node, v

def to_sd(t):
    if not isinstance(t, tuple):
        d = []
        h = 0
    else:
        l, r = t
        d_l, h_l = to_sd(l)
        d_r, h_r = to_sd(r)
        h = max(h_l, h_r) + 1
        d = d_l + [h] + d_r

    return d, h

def to_nlst(t):
    return list(filter(lambda x: x not in ['(', ')', ',', '\'', ' '], str(t)))

def gen_tree():
    return gen_expr(0, 'e')


nodes = '3 + 3 - 6 * 8 / 1 / 7 + 4 / 1 * 1 + 9 * 8 * 1 * 3 * 2 * 6 + 3'.split()
ds = '1 2 3 7 1 2 3 4 5 6 8 9 1 2 3 4 10 11 1 2 3 4 5 6 7 8 9 10 12 13'.split()
# ds = [int(d) for d in ds]
ds = list(map(int, ds))
# print(len(nodes))
# print(len(ds))
print(to_value_sd(ds, nodes))
print(eval(''.join(nodes)))


