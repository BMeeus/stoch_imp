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