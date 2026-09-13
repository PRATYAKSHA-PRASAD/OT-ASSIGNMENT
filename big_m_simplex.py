"""
 BIG-M SIMPLEX METHOD  -  A Generic, From-Scratch Implementation
Case Study
A furniture company manufactures two products: TABLES (x1) and CHAIRS (x2).

    minimize    Z = 4 x1 + 1 x2      (cost, in hundreds of INR, per unit)

    subject to
        3 x1 +  x2  = 3      (a blending / mix requirement that must be met
                               exactly)
        4 x1 + 3 x2 >= 6     (a minimum production commitment to a client)
        1 x1 + 2 x2 <= 4     (a raw-material / capacity restriction)
        x1, x2 >= 0

This is the classical textbook Big-M example (it needs an artificial
variable for the "=" row and another for the ">=" row), so it is a good,
well-known demonstration of every mechanism of the Big-M method:
slack, surplus AND artificial variables all appear in the same model.

Because "M" is a symbolic "very large number" we do NOT plug in a numeric
value for it (that causes round-off errors).  Instead every tableau entry
is stored as a pair  (const_part, M_part)  representing   const + M * m.
Comparisons (choosing the entering / leaving variable, and the optimality
test) are done lexicographically on (M_part, const_part) exactly the way
a human would compare "which number is more negative" when M is assumed
to be an arbitrarily large positive number.
"""

import numpy as np
from fractions import Fraction
class BigM:
    __slots__ = ("c", "m")   # c = constant part, m = coefficient of M

    def __init__(self, c=0, m=0):
        self.c = Fraction(c)
        self.m = Fraction(m)

    def __add__(self, o):
        return BigM(self.c + o.c, self.m + o.m)

    def __sub__(self, o):
        return BigM(self.c - o.c, self.m - o.m)

    def __mul__(self, k):                # multiply by an ordinary scalar
        k = Fraction(k)
        return BigM(self.c * k, self.m * k)

    __rmul__ = __mul__

    def __truediv__(self, k):
        k = Fraction(k)
        return BigM(self.c / k, self.m / k)

    def is_negative(self):
        """True if, for M arbitrarily large & positive, this value < 0."""
        if self.m != 0:
            return self.m < 0
        return self.c < 0

    def is_positive(self):
        """True if, for M arbitrarily large & positive, this value > 0."""
        if self.m != 0:
            return self.m > 0
        return self.c > 0

    def __lt__(self, other):
        if self.m != other.m:
            return self.m < other.m
        return self.c < other.c

    def __repr__(self):
        if self.m == 0:
            return f"{float(self.c):.3g}"
        sign = "+" if self.m > 0 else "-"
        if self.c == 0:
            return f"{sign}{abs(float(self.m)):.3g}M"
        return f"{float(self.c):.3g}{sign}{abs(float(self.m)):.3g}M"

class BigMSimplex:

    def __init__(self, c, A, b, senses, var_names):
        self.n_orig = len(c)
        self.m = len(b)
        self.var_names = list(var_names)
        self.senses = senses

        A = [row[:] for row in A]
        c = c[:]

        self.col_names = list(var_names)
        extra_cols = []         
        artificial_rows = []

        n_slack_surplus_artificial = 0
        for i, s in enumerate(senses):
            if s == "<=":
                name = f"s{i+1}"
                self.col_names.append(name)
                for r in range(self.m):
                    A[r].append(1 if r == i else 0)
                c.append(BigM(0, 0))
            elif s == ">=":
                sname = f"e{i+1}"          # surplus
                self.col_names.append(sname)
                for r in range(self.m):
                    A[r].append(-1 if r == i else 0)
                c.append(BigM(0, 0))

                aname = f"A{i+1}"          # artificial
                self.col_names.append(aname)
                for r in range(self.m):
                    A[r].append(1 if r == i else 0)
                c.append(BigM(0, 1))
                artificial_rows.append((i, self.col_names.index(aname)))
            elif s == "=":
                aname = f"A{i+1}"
                self.col_names.append(aname)
                for r in range(self.m):
                    A[r].append(1 if r == i else 0)
                c.append(BigM(0, 1))
                artificial_rows.append((i, self.col_names.index(aname)))
            else:
                raise ValueError("sense must be '<=', '=' or '>='")

        self.A = np.array([[Fraction(v) for v in row] for row in A], dtype=object)
        self.b = np.array([Fraction(v) for v in b], dtype=object)
        self.c = np.array(
            [v if isinstance(v, BigM) else BigM(v, 0) for v in c], dtype=object
        )
        self.n_cols = self.A.shape[1]

        # basis = artificial where present, else slack
        self.basis = [None] * self.m
        for i, s in enumerate(senses):
            if s == "<=":
                self.basis[i] = self.col_names.index(f"s{i+1}")
        for (row, col) in artificial_rows:
            self.basis[row] = col

        self.iterations = []      # store tableau snapshots for reporting

    def _zj_minus_cj(self):
        z = []
        for j in range(self.n_cols):
            zj = BigM(0, 0)
            for i in range(self.m):
                zj = zj + self.c[self.basis[i]] * self.A[i][j]
            z.append(zj - self.c[j])
        return z

    def solve(self, max_iter=50, verbose=True):
        it = 0
        while True:
            zj_cj = self._zj_minus_cj()
            self._snapshot(it, zj_cj)

            entering = None
            most_positive = None
            for j, val in enumerate(zj_cj):
                if val.is_positive():
                    if most_positive is None or most_positive < val:
                        most_positive = val
                        entering = j

            if entering is None:
                break   # optimal

            ratios = []
            for i in range(self.m):
                a_ij = self.A[i][entering]
                if a_ij > 0:
                    ratios.append((self.b[i] / a_ij, i))
            if not ratios:
                raise RuntimeError("Problem is unbounded.")
            _, leaving_row = min(ratios, key=lambda t: (t[0],))

            self._pivot(leaving_row, entering)
            self.basis[leaving_row] = entering
            it += 1
            if it > max_iter:
                raise RuntimeError("Too many iterations - check the model.")

        return self._extract_solution()

    def _pivot(self, r, c):
        piv = self.A[r][c]
        self.A[r] = [v / piv for v in self.A[r]]
        self.b[r] = self.b[r] / piv
        for i in range(self.m):
            if i == r:
                continue
            factor = self.A[i][c]
            if factor == 0:
                continue
            self.A[i] = [self.A[i][j] - factor * self.A[r][j] for j in range(self.n_cols)]
            self.b[i] = self.b[i] - factor * self.b[r]

    def _snapshot(self, it, zj_cj):
        self.iterations.append(
            {
                "iter": it,
                "basis": [self.col_names[b] for b in self.basis],
                "A": [list(self.A[i]) for i in range(self.m)],   # true deep copy
                "xb": [self.b[i] for i in range(self.m)],
                "zj_cj": zj_cj[:],
            }
        )

    def _extract_solution(self):
        x = {name: Fraction(0) for name in self.col_names}
        for i in range(self.m):
            x[self.col_names[self.basis[i]]] = self.b[i]

        z = BigM(0, 0)
        for i in range(self.m):
            z = z + self.c[self.basis[i]] * self.b[i]

        artificial_nonzero = any(
            x[name] != 0 for name in self.col_names if name.startswith("A")
        )
        return {
            "x": x,
            "Z": z,
            "feasible": not artificial_nonzero,
        }
    def print_iterations(self):
        for snap in self.iterations:
            print(f"\n--- Simplex Tableau : Iteration {snap['iter']} ---")
            header = f"{'Basis':<8}" + "".join(f"{n:>14}" for n in self.col_names) + f"{'RHS':>12}"
            print(header)
            for i in range(self.m):
                row_vals = "".join(f"{str(snap['A'][i][j]):>14}" for j in range(self.n_cols))
                print(f"{snap['basis'][i]:<8}{row_vals}{str(snap['xb'][i]):>12}")
            zrow = "".join(f"{str(v):>14}" for v in snap["zj_cj"])
            print(f"{'Zj-Cj':<8}{zrow}")

if __name__ == "__main__":
    print("=" * 78)
    print(" BIG-M SIMPLEX METHOD - Product Mix (Cost Minimisation) LPP")
    print("=" * 78)
    print(
        """
    Decision variables :
        x1 = number of TABLES produced
        x2 = number of CHAIRS produced

    LP Model:
        Minimize   Z = 4 x1 + 1 x2

        subject to
            3 x1 +   x2  =  3      (exact material-blend requirement)
            4 x1 + 3 x2 >=  6      (minimum committed supply to client)
              x1 + 2 x2 <=  4      (machine-hour capacity)
            x1, x2 >= 0

    Standard form (slack s3, surplus e2, artificials A1, A2):
        3 x1 +  x2            + A1                = 3
        4 x1 + 3x2      - e2       + A2            = 6
        1 x1 + 2x2 + s3                            = 4
    """
    )

    c = [4, 1]
    A = [
        [3, 1],
        [4, 3],
        [1, 2],
    ]
    b = [3, 6, 4]
    senses = ["=", ">=", "<="]
    var_names = ["x1", "x2"]

    solver = BigMSimplex(c, A, b, senses, var_names)
    result = solver.solve()
    solver.print_iterations()

    print("\n" + "=" * 78)
    print(" OPTIMAL SOLUTION")
    print("=" * 78)
    for name in var_names:
        val = result["x"][name]
        print(f"   {name} = {val}  ( = {float(val):.4f} )")
    print(f"\n   Optimal Objective Value  Z* = {result['Z'].c}  ( = {float(result['Z'].c):.4f} )")
    print(f"   Solution feasible (all artificial variables = 0): {result['feasible']}")

    from scipy.optimize import linprog
    A_ub = [[1, 2]]          # x1 + 2x2 <= 4
    b_ub = [4]
    A_eq_lp = [[3, 1]]       # 3x1 + x2 = 3
    b_eq_lp = [3]
    # 4x1 + 3x2 >= 6   -->   -4x1 -3x2 <= -6
    A_ub.append([-4, -3])
    b_ub.append(-6)

    res = linprog(c=[4, 1], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_lp, b_eq=b_eq_lp,
                   bounds=[(0, None), (0, None)], method="highs")

    print("\n" + "-" * 78)
    print(" Cross-check against SciPy linprog (HiGHS solver)")
    print("-" * 78)
    print(f"   SciPy   x1 = {res.x[0]:.4f} , x2 = {res.x[1]:.4f} , Z = {res.fun:.4f}")
    print(f"   Big-M   x1 = {float(result['x']['x1']):.4f} , x2 = {float(result['x']['x2']):.4f}"
          f" , Z = {float(result['Z'].c):.4f}")
    match = (abs(res.x[0] - float(result['x']['x1'])) < 1e-6 and
             abs(res.x[1] - float(result['x']['x2'])) < 1e-6)
    print(f"   -> Results MATCH: {match}")
