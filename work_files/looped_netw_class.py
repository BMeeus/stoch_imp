import stoch_imp as si
import sympy as sp


n = 4

f, om = sp.symbols("f, omega", real=True)
fEq = 0
t = 1

w = si.WMatrix(n, f, fEq)

for i in range(1, n+1):
    w.add_trans(i, (i+1)%n, t*sp.exp(f))

w.add_trans(1, 3, t*sp.exp(f))

print(w.calc_weq())
print(w.calc_w1())
