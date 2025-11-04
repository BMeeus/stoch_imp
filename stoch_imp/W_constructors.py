from .WMatrix import *
from sympy import exp, symbols

def nstar(n, driven=None, **kwargs):
    if 'ds' not in kwargs.keys():
        ds = symbols('mu', real=True)
        kwargs['ds'] = ds
    else:
        ds = kwargs['ds']

    W = WMatrix(n, **kwargs)

    if driven is None:
        driven = [n]

    for i in range(2, n+1):
        if i in driven:
            W.add_trans(1, i, r=exp(ds))
        else:
            W.add_trans(1, i, r=1)

    return W, ds

def nloop(n, driven=None, **kwargs):
    if 'ds' not in kwargs.keys():
        ds = symbols('mu', real=True)
        kwargs['ds'] = ds
    else:
        ds = kwargs['ds']

    W = WMatrix(n, **kwargs)

    if driven is None:
        driven = [1]

    for i in range(1, n+1):
        if i in driven:
            W.add_trans(i, i+1, r=exp(ds))
        else:
            W.add_trans(i, i+1, r=1)

    return W, ds


def nball(n, driven:list=None, **kwargs):
    if 'ds' not in kwargs.keys():
        ds = symbols('mu', real=True)
        kwargs['ds'] = ds
    else:
        ds = kwargs['ds']

    W = WMatrix(n, **kwargs)

    if driven is None:
        driven = [[1, i] for i in range(2, n+1)]
    else:
        for elem in driven:
            if isinstance(elem, int):
                driven.remove(elem)
                driven += [[elem, i].sort() for i in range(1, n+1) if i != elem]

    for i in range(1, n + 1):
        for j in range(i, n+1):
            if [i, j] in driven:
                W.add_trans(i, j, r=exp(ds))
            else:
                W.add_trans(i, j, r=1)

    return W, ds