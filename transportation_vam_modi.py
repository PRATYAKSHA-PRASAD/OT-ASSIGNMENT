"""
 TRANSPORTATION PROBLEM
 (i)  Vogel's Approximation Method (VAM)  -> Initial Basic Feasible Solution
 (ii) MODI (Modified Distribution) Method -> Test & Improve to Optimality
Case Study
A cement company has THREE manufacturing plants (sources) that must supply
FOUR regional warehouses (destinations).

    Plants (Supply, in '000 bags):   P1 = 20   P2 = 30   P3 = 25
    Warehouses (Demand, in '000 bags): W1 = 10  W2 = 25  W3 = 20  W4 = 20

    Unit transportation cost (Rs. per '000 bags):

                 W1    W2    W3    W4
            P1    4     6     8     8
            P2    6     8     6     7
            P3    5     7     6     8

    Total supply = 20+30+25 = 75  =  Total demand = 10+25+20+20 = 75
    (the problem is BALANCED, so no dummy row/column is required)

Goal: find the shipment plan (how many '000 bags to send from each plant to
each warehouse) that minimises total transportation cost.
"""

from fractions import Fraction

def vogels_approximation_method(supply, demand, cost, src_names, dst_names, verbose=True):
    supply = supply[:]
    demand = demand[:]
    m, n = len(supply), len(demand)
    allocation = [[0] * n for _ in range(m)]
    active_rows = set(range(m))
    active_cols = set(range(n))

    step = 1
    while active_rows and active_cols:
        row_pen, col_pen = {}, {}
        for i in active_rows:
            costs = sorted(cost[i][j] for j in active_cols)
            row_pen[i] = (costs[1] - costs[0]) if len(costs) > 1 else costs[0]
        for j in active_cols:
            costs = sorted(cost[i][j] for i in active_rows)
            col_pen[j] = (costs[1] - costs[0]) if len(costs) > 1 else costs[0]

        best = max(
            [("row", i, row_pen[i]) for i in active_rows]
            + [("col", j, col_pen[j]) for j in active_cols],
            key=lambda t: t[2],
        )
        kind, idx, pen = best

        if kind == "row":
            j = min(active_cols, key=lambda c: cost[idx][c])
            i = idx
        else:
            i = min(active_rows, key=lambda r: cost[r][idx])
            j = idx

        qty = min(supply[i], demand[j])
        allocation[i][j] = qty
        supply[i] -= qty
        demand[j] -= qty

        if verbose:
            print(f"Step {step:2d}: penalty({kind}={src_names[idx] if kind=='row' else dst_names[idx]})"
                  f"={pen}  ->  allocate {qty} units to cell "
                  f"({src_names[i]}, {dst_names[j]})  [cost/unit = {cost[i][j]}]"
                  f"   remaining supply[{src_names[i]}]={supply[i]}, demand[{dst_names[j]}]={demand[j]}")
        step += 1

        if supply[i] == 0:
            active_rows.discard(i)
        if demand[j] == 0:
            active_cols.discard(j)

    total_cost = sum(allocation[i][j] * cost[i][j] for i in range(m) for j in range(n))
    return allocation, total_cost


def modi_method(allocation, cost, supply, demand, src_names, dst_names, verbose=True):
    m, n = len(supply), len(demand)
    alloc = [row[:] for row in allocation]

    def basic_cells():
        return [(i, j) for i in range(m) for j in range(n) if alloc[i][j] > 0]

    def fix_degeneracy():
        needed = m + n - 1
        cells = basic_cells()
        while len(cells) < needed:
             candidates = sorted(
                ((i, j) for i in range(m) for j in range(n) if alloc[i][j] == 0),
                key=lambda ij: cost[ij[0]][ij[1]],
            )
            placed = False
            for (i, j) in candidates:
                alloc[i][j] = Fraction(0)  
                trial = basic_cells()
                if len(trial) == needed:
                    placed = True
                    break
                else:
                    alloc[i][j] = 0
            if not placed:
                break
            cells = basic_cells()

    iteration = 1
    while True:
        fix_degeneracy()
        cells = basic_cells()

        u = [None] * m
        v = [None] * n
        u[0] = Fraction(0)
        changed = True
        while changed:
            changed = False
            for (i, j) in cells:
                if u[i] is not None and v[j] is None:
                    v[j] = Fraction(cost[i][j]) - u[i]
                    changed = True
                elif v[j] is not None and u[i] is None:
                    u[i] = Fraction(cost[i][j]) - v[j]
                    changed = True

        deltas = {}
        for i in range(m):
            for j in range(n):
                if alloc[i][j] == 0 and (i, j) not in cells:
                    deltas[(i, j)] = Fraction(cost[i][j]) - (u[i] + v[j])

        negative_cells = {k: d for k, d in deltas.items() if d < 0}
        if not negative_cells:
            print(f"Optimal at iteration {iteration}")
            break

        entering = min(negative_cells, key=lambda k: negative_cells[k])
        loop = _find_loop(entering, cells)
        if loop is None:
            raise RuntimeError("Could not find a closed loop - check the table for degeneracy.")

        minus_cells = loop[1::2]
        theta = min(alloc[i][j] for (i, j) in minus_cells)

        for idx, (i, j) in enumerate(loop):
            if idx % 2 == 0:
                alloc[i][j] += theta
            else:
                alloc[i][j] -= theta

        iteration += 1
        if iteration > 30:
            raise RuntimeError("Too many MODI iterations.")

    total_cost = sum(alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))
    return alloc, total_cost


def _find_loop(start, basic_cells):
    cells = list(set(basic_cells) | {start})

    def search(path, visited_rows_cols):
        current = path[-1]
        if len(path) > 3 and current == start:
            return path[:-1]
        moving_along_row = (len(path) % 2 == 1)
        candidates = []
        for c in cells:
            if c == path[-1]:
                continue
            if moving_along_row and c[0] == current[0]:
                candidates.append(c)
            elif (not moving_along_row) and c[1] == current[1]:
                candidates.append(c)
        for nxt in candidates:
            if nxt in path[1:]:
                if nxt == start and len(path) >= 3:
                    return path + [nxt]
                continue
            result = search(path + [nxt], visited_rows_cols)
            if result:
                return result
        return None

    result = search([start], None)
    return result


def _print_table(alloc, cost, src_names, dst_names):
    m, n = len(src_names), len(dst_names)
    header = f"{'':>6}" + "".join(f"{d:>12}" for d in dst_names)
    print(header)
    for i in range(m):
        row = f"{src_names[i]:>6}"
        for j in range(n):
            a = alloc[i][j]
            cell = f"{a}@{cost[i][j]}" if a != 0 else "  -  "
            row += f"{cell:>12}"
        print(row)

if __name__ == "__main__":
    src_names = ["P1", "P2", "P3"]
    dst_names = ["W1", "W2", "W3", "W4"]
    supply = [20, 30, 25]
    demand = [10, 25, 20, 20]
    cost = [
        [4, 6, 8, 8],
        [6, 8, 6, 7],
        [5, 7, 6, 8],
    ]

    print("=" * 78)
    print(" TRANSPORTATION PROBLEM - Cement Plants -> Regional Warehouses")
    print("=" * 78)
    print(f"Supply  : { {src_names[i]: supply[i] for i in range(3)} }")
    print(f"Demand  : { {dst_names[j]: demand[j] for j in range(4)} }")
    print(f"Total supply = {sum(supply)}   Total demand = {sum(demand)}  "
          f"-> {'BALANCED' if sum(supply) == sum(demand) else 'UNBALANCED'}")
    print("\nCost matrix (Rs. per '000 bags):")
    header = f"{'':>6}" + "".join(f"{d:>8}" for d in dst_names)
    print(header)
    for i in range(3):
        row = f"{src_names[i]:>6}" + "".join(f"{cost[i][j]:>8}" for j in range(4))
        print(row)
    print("\n" + "=" * 78)
    print(" STEP 1: VOGEL'S APPROXIMATION METHOD (VAM) -> Initial BFS")
    print("=" * 78)
    vam_alloc, vam_cost = vogels_approximation_method(supply, demand, cost, src_names, dst_names)

    print("\nInitial Basic Feasible Solution (from VAM):")
    _print_table(vam_alloc, cost, src_names, dst_names)
    print(f"\nTotal cost of VAM initial solution = {vam_cost}")
    print("\n" + "=" * 78)
    print(" STEP 2: MODI METHOD -> Test optimality & improve the VAM solution")
    print("=" * 78)
    final_alloc, final_cost = modi_method(vam_alloc, cost, supply, demand, src_names, dst_names)
    print("\n" + "=" * 78)
    print(" FINAL OPTIMAL SHIPMENT PLAN")
    print("=" * 78)
    _print_table(final_alloc, cost, src_names, dst_names)
    print(f"\nMinimum Total Transportation Cost = Rs. {final_cost}  (in '000s)")
    from scipy.optimize import linprog
    import numpy as np

    n_src, n_dst = 3, 4
    c_vec = [cost[i][j] for i in range(n_src) for j in range(n_dst)]
    A_eq = []
    b_eq = []
    for i in range(n_src):
        row = [0] * (n_src * n_dst)
        for j in range(n_dst):
            row[i * n_dst + j] = 1
        A_eq.append(row)
        b_eq.append(supply[i])
    for j in range(n_dst):
        row = [0] * (n_src * n_dst)
        for i in range(n_src):
            row[i * n_dst + j] = 1
        A_eq.append(row)
        b_eq.append(demand[j])

    res = linprog(c=c_vec, A_eq=A_eq, b_eq=b_eq,
                   bounds=[(0, None)] * (n_src * n_dst), method="highs")

    print("\n" + "-" * 78)
    print(" Cross-check against SciPy linprog (HiGHS exact LP solver)")
    print("-" * 78)
    print(f"   SciPy optimal cost   = Rs. {res.fun:.4f}")
    print(f"   VAM + MODI cost      = Rs. {float(final_cost):.4f}")
    print(f"   -> Results MATCH: {abs(res.fun - float(final_cost)) < 1e-6}")
