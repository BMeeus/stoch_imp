import stoch_imp as si
import sympy as sp

n = 50                            # Number of sites (excl. basins)
name = f"{n}_loop"
foldername = "n_sites_loop"
F = sp.symbols("F", real=True)   # variable driving force
FEq = 0                          # base value of driving

t = 1.0

w = si.WMatrix(n, F)

for i in range(n):
    w.add_trans(i, i + 1, t * sp.exp(F))

w.add_trans(1, 3, t * sp.exp(F))
print(w)

weq = w.weq()
w1 = w.w1()

print(weq)
print(w1)
print(weq.peq())
print(weq.check_db())