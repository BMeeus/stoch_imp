import numpy as np
import sympy as sp

n = 3
mu = sp.symbols("mu", real=True)
muEq = 0

rr = 1
rl = 1
t = 1

a = 1
b = 0
mul = a*mu
mur = -b*mu

def resratep(base, m):
    return base / (sp.exp(-m)+1)

def resratem(base, m):
    return base - resratep(base, m)

def add_trans(i, j, rate):
    wbase[i, j] = rate
    wbase[j, i] = rate
    return

def pprint(o):
    print(sp.pretty_print(o))
    return

def w(x):
    return winterim.subs({mu: x})

wbase = sp.zeros(n+2, n+2)
wbase[0, 1] = resratem(rl, mul)
wbase[1, 0] = resratep(rl, mul)
wbase[-2, -1] = resratem(rr, mur)
wbase[-1, -2] = resratep(rr, mur)

for i in range(1, n):
    add_trans(i, i+1, t)

wbaseEq = wbase.subs({mu: muEq})

winterim = wbase[:-1, :-1]
winterim[-1, 0] = wbase[-2, -1]
winterim[0, -1] = wbase[-1, -2]

for i in range(n+1):
    winterim[i, i] = - sum(winterim[:, i])

weq = w(0)

w1 = sp.diff(winterim, mu).subs({mu: 0})

#TODO continue to follow three-sites.nb

pprint(w1)