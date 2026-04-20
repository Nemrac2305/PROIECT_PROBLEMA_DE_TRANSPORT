import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional

TOL = 1e-9

# =========================================================
# VALIDARE INPUT
# =========================================================
def validate_input(supply, demand, costs):
    if len(supply) == 0:
        raise ValueError("Lista disponibilitatilor nu poate fi vida.")
    if len(demand) == 0:
        raise ValueError("Lista necesarelor nu poate fi vida.")
    supply_arr = np.array(supply, dtype=float)
    demand_arr = np.array(demand, dtype=float)
    costs_arr  = np.array(costs,  dtype=float)
    if costs_arr.ndim != 2:
        raise ValueError("Matricea costurilor trebuie sa fie bidimensionala.")
    if costs_arr.shape != (len(supply), len(demand)):
        raise ValueError(f"Dimensiunea matricei costurilor {costs_arr.shape} != ({len(supply)},{len(demand)}).")
    if np.any(supply_arr < 0): raise ValueError("Disponibilitatile trebuie sa fie nenegative.")
    if np.any(demand_arr < 0): raise ValueError("Necesarele trebuie sa fie nenegative.")
    if np.any(costs_arr  < 0): raise ValueError("Costurile trebuie sa fie nenegative.")


# =========================================================
# UTILITARE AFISARE
# =========================================================
def fmt(x: float) -> str:
    if abs(x) < TOL:
        return "0"
    if abs(x - round(x)) < TOL:
        return str(int(round(x)))
    return f"{x:.4f}"


def print_transport_table(
    x: np.ndarray,
    costs: np.ndarray,
    basis: Set[Tuple[int, int]],
    supply: List[float],
    demand: List[float],
    title: str = "Tabel",
    source_labels: List[str] = None,
    dest_labels: List[str] = None,
):
    """
    Afiseaza tabelul de transport in format similar cu seminarul:
      - fiecare celula: col 6 chars, cost in colt sus-dreapta, alocare centrata
      - celulele bazice sunt marcate cu [ ]
      - marginile: disponibilitati pe dreapta, necesare jos
    """
    m, n = x.shape
    if source_labels is None:
        source_labels = [f"A{i+1}" for i in range(m)]
    if dest_labels is None:
        dest_labels = [f"B{j+1}" for j in range(n)]

    cell_w = 8  # latimea unei celule
    label_w = 5

    sep = "-" * (label_w + 2 + (cell_w + 1) * n + 7)
    print(f"\n{'='*len(sep)}")
    print(f"  {title}")
    print(sep)

    # Header destinatii
    hdr = " " * (label_w + 3)
    for lbl in dest_labels:
        hdr += f"{lbl:^{cell_w+1}}"
    hdr += "   D (disp.)"
    print(hdr)
    print(sep)

    for i in range(m):
        row_top    = " " * (label_w + 3)
        row_mid    = f"{source_labels[i]:<{label_w}} | "
        row_bot    = " " * (label_w + 3)
        for j in range(n):
            c_str = fmt(costs[i, j])
            if (i, j) in basis:
                alloc = fmt(x[i, j])
                # top: empty, mid: [alloc], bot: cost small right-aligned
                top_cell  = f"{'':^{cell_w}}"
                mid_cell  = f"[{alloc:^{cell_w-2}}]"
                bot_cell  = f"{c_str:>{cell_w}}"
            else:
                top_cell  = f"{'':^{cell_w}}"
                mid_cell  = f"{c_str:^{cell_w}}"
                bot_cell  = f"{'':^{cell_w}}"
            row_top += top_cell + " "
            row_mid += mid_cell + " "
            row_bot += bot_cell + " "

        d_str = fmt(supply[i])
        row_top += f"  "
        row_mid += f"  {d_str}"
        print(row_top)
        print(row_mid)
        print(row_bot)
        if i < m - 1:
            print(" " * (label_w + 3) + "-" * ((cell_w + 1) * n))

    print(sep)

    # Footer necesare
    foot = " " * (label_w + 3)
    for j in range(n):
        foot += f"{fmt(demand[j]):^{cell_w+1}}"
    print("N (nec.) " + foot.strip())
    print(sep)


def print_potential_table(
    costs: np.ndarray,
    basis: Set[Tuple[int, int]],
    u: np.ndarray,
    v: np.ndarray,
    source_labels: List[str] = None,
    dest_labels: List[str] = None,
):
    """
    Afiseaza tabelul C~ (costuri modificate) si Delta.
    """
    m, n = costs.shape
    if source_labels is None:
        source_labels = [f"A{i+1}" for i in range(m)]
    if dest_labels is None:
        dest_labels = [f"B{j+1}" for j in range(n)]

    cell_w = 8
    label_w = 5

    sep = "-" * (label_w + 2 + (cell_w + 1) * (n + 2))
    print(f"\n  Potentiale: u = [{', '.join(fmt(ui) for ui in u)}]")
    print(f"              v = [{', '.join(fmt(vj) for vj in v)}]")
    print(f"\n  Tabel C~ si Delta_ij  (B = celula bazica, altfel delta)")
    print(sep)

    # Header
    hdr = f"{'':^{label_w}}  | "
    for lbl in dest_labels:
        hdr += f"{lbl:^{cell_w+1}}"
    hdr += f"  {'u_i':^{cell_w}}"
    print(hdr)
    print(sep)

    all_positive = True
    entering_cell = None
    min_delta = float('inf')

    rows_data = []
    for i in range(m):
        row = f"{source_labels[i]:<{label_w}} | "
        for j in range(n):
            if (i, j) in basis:
                cell_str = f"{'B':^{cell_w}}"
            else:
                delta = costs[i, j] - (u[i] + v[j])
                cell_str = f"{fmt(delta):^{cell_w}}"
                if delta < -TOL:
                    all_positive = False
                    if delta < min_delta:
                        min_delta = delta
                        entering_cell = (i, j)
            row += cell_str + " "
        row += f"  {fmt(u[i]):^{cell_w}}"
        rows_data.append(row)

    for row in rows_data:
        print(row)

    print(sep)
    foot = f"{'v_j':<{label_w}}    "
    for j in range(n):
        foot += f"{fmt(v[j]):^{cell_w+1}}"
    print(foot)
    print(sep)

    if all_positive:
        print("  >> TO: toate Delta_ij >= 0  =>  DA  =>  STOP (solutie optima)")
    else:
        print(f"  >> TO: exista Delta_ij < 0  =>  NU  =>  continuam iteratia")
        if entering_cell:
            print(f"  >> Celula de intrare in baza: ({entering_cell[0]+1}, {entering_cell[1]+1})  Delta = {fmt(min_delta)}")

    return all_positive, entering_cell, min_delta


# =========================================================
# PAS 1: ECHILIBRARE
# =========================================================
def balance_problem(supply, demand, costs):
    supply = [float(s) for s in supply]
    demand = [float(d) for d in demand]
    costs  = np.array(costs, dtype=float)
    m, n   = costs.shape
    sum_s, sum_d = sum(supply), sum(demand)
    added = None

    if abs(sum_s - sum_d) <= TOL:
        return supply, demand, costs, added

    if sum_s > sum_d:
        diff   = sum_s - sum_d
        costs  = np.hstack([costs, np.zeros((m, 1))])
        demand.append(diff)
        added  = ("destination", len(demand) - 1, diff)
    else:
        diff   = sum_d - sum_s
        costs  = np.vstack([costs, np.zeros((1, n))])
        supply.append(diff)
        added  = ("source", len(supply) - 1, diff)

    return supply, demand, costs, added


# =========================================================
# UTILITATE: CICLU
# =========================================================
def creates_cycle(basis: Set[Tuple[int, int]], m: int, n: int) -> bool:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    for r in range(m):
        parent[("r", r)] = ("r", r)
    for c in range(n):
        parent[("c", c)] = ("c", c)

    for (r, c) in basis:
        if not union(("r", r), ("c", c)):
            return True
    return False


# =========================================================
# PAS 2: SOLUTIE INITIALA NORD-VEST (cu tehnica epsilon
#         pentru solutii degenerate)
# =========================================================
def northwest_corner(
    supply_orig: List[float],
    demand_orig: List[float],
    costs: np.ndarray,
    verbose: bool = True
) -> Tuple[np.ndarray, Set[Tuple[int, int]], bool, np.ndarray, List[float], List[float]]:
    """
    Aplica metoda Nord-Vest.
    Daca solutia este degenerata (NC < m+n-1), aplica tehnica epsilon (perturbatie).
    Returneaza:
      x, basis, is_degenerate, costs_used, supply_used, demand_used
    """
    m = len(supply_orig)
    n = len(demand_orig)
    target = m + n - 1

    def run_nv(supply_in, demand_in):
        s = list(supply_in)
        d = list(demand_in)
        x = np.zeros((m, n), dtype=float)
        basis: Set[Tuple[int, int]] = set()
        i = j = 0

        while i < m and j < n:
            amount = min(s[i], d[j])
            x[i, j] = amount
            basis.add((i, j))
            s[i] -= amount
            d[j] -= amount

            s_zero = abs(s[i]) <= TOL
            d_zero = abs(d[j]) <= TOL

            if s_zero and d_zero:
                # Degenerare: adauga celula epsilon daca mai avem spatiu
                if i < m - 1 and j < n - 1:
                    cand = (i, j + 1)
                    trial = set(basis) | {cand}
                    if not creates_cycle(trial, m, n):
                        basis.add(cand)
                    else:
                        cand = (i + 1, j)
                        trial = set(basis) | {cand}
                        if not creates_cycle(trial, m, n):
                            basis.add(cand)
                i += 1
                j += 1
            elif s_zero:
                i += 1
            else:
                j += 1

        # Completeaza baza daca inca lipsesc celule
        for r in range(m):
            for c in range(n):
                if len(basis) >= target:
                    break
                if (r, c) not in basis:
                    trial = set(basis) | {(r, c)}
                    if not creates_cycle(trial, m, n):
                        basis.add((r, c))
            if len(basis) >= target:
                break

        return x, basis

    x, basis = run_nv(supply_orig, demand_orig)
    nc = len(basis)
    v_val = target  # m + n - 1

    if nc == v_val:
        if verbose:
            print(f"\n  NC = {nc}  |  V = m+n-1 = {v_val}  =>  NC = V  =>  Solutie nedegenerata")
        return x, basis, False, costs, supply_orig, demand_orig

    # ---- Solutie degenerata => tehnica epsilon ----
    if verbose:
        print(f"\n  NC = {nc}  |  V = m+n-1 = {v_val}  =>  NC < V  =>  Solutie DEGENERATA")
        print("  => Aplicam tehnica perturbarii (Tehnica Epsilon)")
        print("     Rescriem tabelul initial cu disponibilitati a_i + epsilon,")
        print(f"     si necesarul ultimei destinatii b_{n} + {m}*epsilon")

    # Perturbam: a_i -> a_i + epsilon  (reprezentat ca a_i + 1 in structura)
    # d_j -> d_j  pentru j < n,  d_n -> d_n + m*epsilon
    # In practica, lucram simbolic: tinem un vector eps_supply si eps_demand
    # si la final setam epsilon=0. Pentru a asigura nedegenerarea, adaugam
    # valori mici DISTINCTE la fiecare disponibilitate.
    EPS = 1e-6
    supply_eps = [s + EPS for s in supply_orig]
    demand_eps = list(demand_orig)
    demand_eps[-1] += m * EPS   # ultimul necesar primeste m*epsilon

    x_eps, basis_eps = run_nv(supply_eps, demand_eps)
    nc_eps = len(basis_eps)

    if verbose:
        print(f"  Dupa perturbatie: NC = {nc_eps}  |  V = {v_val}")
        if nc_eps == v_val:
            print("  => NC = V  =>  Solutie nedegenerata (cu epsilon)")
        else:
            print(f"  ATENTIE: inca {v_val - nc_eps} celule lipsa, completam manual cu epsilon=0")

    # Resetam valorile epsilon la 0 in matricea x (tinem doar structura bazei)
    x_clean = np.zeros((m, n), dtype=float)
    for (r, c) in basis_eps:
        x_clean[r, c] = max(0.0, round(x_eps[r, c] - EPS, 8) if c == n-1 else x_eps[r, c])
    x_clean[np.abs(x_clean) <= TOL] = 0.0

    return x_clean, basis_eps, True, costs, supply_orig, demand_orig


# =========================================================
# PAS 3: POTENTIALE
# =========================================================
def compute_potentials(costs: np.ndarray, basis: Set[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    m, n = costs.shape
    u: List[Optional[float]] = [None] * m
    v: List[Optional[float]] = [None] * n
    u[0] = 0.0
    changed = True
    while changed:
        changed = False
        for (r, c) in basis:
            if u[r] is not None and v[c] is None:
                v[c] = costs[r, c] - u[r]; changed = True
            elif v[c] is not None and u[r] is None:
                u[r] = costs[r, c] - v[c]; changed = True
    if any(val is None for val in u) or any(val is None for val in v):
        raise RuntimeError("Baza este deconectata. Potentialele nu pot fi calculate.")
    return np.array(u, dtype=float), np.array(v, dtype=float)


# =========================================================
# PAS 4: DELTA
# =========================================================
def compute_deltas(costs, basis, u, v) -> Dict[Tuple[int,int], float]:
    m, n = costs.shape
    return {
        (r, c): costs[r, c] - (u[r] + v[c])
        for r in range(m) for c in range(n)
        if (r, c) not in basis
    }


# =========================================================
# PAS 5: CIRCUIT DE AMELIORARE
# =========================================================
def extract_cycle_cells(basis: Set[Tuple[int,int]], entering: Tuple[int,int]) -> Set[Tuple[int,int]]:
    cells = set(basis) | {entering}
    changed = True
    while changed:
        changed = False
        row_count = defaultdict(int)
        col_count = defaultdict(int)
        for (r, c) in cells:
            row_count[r] += 1
            col_count[c]  += 1
        to_remove = {cell for cell in cells
                     if row_count[cell[0]] < 2 or col_count[cell[1]] < 2}
        if to_remove:
            cells -= to_remove
            changed = True
    if entering not in cells or len(cells) < 4:
        raise RuntimeError("Nu s-a putut determina circuitul de ameliorare.")
    return cells


def order_cycle(cycle_cells: Set[Tuple[int,int]], entering: Tuple[int,int]) -> List[Tuple[int,int]]:
    row_map = defaultdict(list)
    col_map = defaultdict(list)
    for cell in cycle_cells:
        row_map[cell[0]].append(cell)
        col_map[cell[1]].append(cell)

    for r, cells in row_map.items():
        if len(cells) != 2:
            raise RuntimeError(f"Linia {r+1} nu are exact 2 celule in circuit.")
    for c, cells in col_map.items():
        if len(cells) != 2:
            raise RuntimeError(f"Coloana {c+1} nu are exact 2 celule in circuit.")

    def attempt(start_on_row: bool):
        path = [entering]
        current = entering
        on_row = start_on_row
        for _ in range(len(cycle_cells) - 1):
            candidates = row_map[current[0]] if on_row else col_map[current[1]]
            a, b = candidates
            nxt = b if a == current else a
            if nxt in path:
                return None
            path.append(nxt)
            current = nxt
            on_row = not on_row
        # verifica inchidere
        candidates = row_map[current[0]] if on_row else col_map[current[1]]
        a, b = candidates
        if (b if a == current else a) != entering:
            return None
        return path

    path = attempt(True) or attempt(False)
    if path is None:
        raise RuntimeError("Nu s-a putut ordona circuitul.")
    return path


# =========================================================
# PAS 6: PIVOTARE
# =========================================================
def pivot(x, basis, cycle_path):
    minus_cells = [cycle_path[i] for i in range(1, len(cycle_path), 2)]
    theta = min(x[r, c] for (r, c) in minus_cells)
    leaving = sorted((r, c) for (r, c) in minus_cells if abs(x[r, c] - theta) <= TOL)[0]

    for idx, (r, c) in enumerate(cycle_path):
        x[r, c] += theta if idx % 2 == 0 else -theta

    x[np.abs(x) <= TOL] = 0.0
    basis.add(cycle_path[0])
    basis.remove(leaving)
    return theta, leaving


# =========================================================
# SOLVER PRINCIPAL
# =========================================================
def solve_transportation(supply, demand, costs, verbose=True, max_iter=200):
    validate_input(supply, demand, costs)

    supply_orig = list(supply)
    demand_orig = list(demand)

    supply_b, demand_b, costs_b, added = balance_problem(supply, demand, costs)
    m, n = costs_b.shape
    target = m + n - 1

    # Etichete
    source_labels = [f"A{i+1}" for i in range(len(supply_orig))]
    dest_labels   = [f"B{j+1}" for j in range(len(demand_orig))]
    if added:
        kind, idx, qty = added
        if kind == "destination":
            dest_labels.append(f"B{idx+1}*")
        else:
            source_labels.append(f"A{idx+1}*")

    if verbose:
        print("\n" + "="*70)
        print("  [PAS 1] VERIFICARE / ECHILIBRARE")
        print("="*70)
        print(f"  Suma disponibilitati (ΣD) = {fmt(sum(supply_b))}")
        print(f"  Suma necesare       (ΣN) = {fmt(sum(demand_b))}")
        if added is None:
            print("  => Problema echilibrata (ΣD = ΣN)  =>  PTE")
        else:
            kind, idx, qty = added
            if kind == "destination":
                print(f"  => ΣD > ΣN  =>  Problema NEECHILIBRATA")
                print(f"     S-a adaugat destinatie fictiva B{idx+1}* cu necesar = {fmt(qty)}")
                print(f"     (costurile pe coloana fictiva sunt 0)")
                print(f"  => Dupa echilibrare: PTE")
            else:
                print(f"  => ΣN > ΣD  =>  Problema NEECHILIBRATA")
                print(f"     S-a adaugat sursa fictiva A{idx+1}* cu disponibil = {fmt(qty)}")
                print(f"     (costurile pe linia fictiva sunt 0)")
                print(f"  => Dupa echilibrare: PTE")

        print("\n" + "="*70)
        print("  [PAS 1] TABEL INITIAL (dupa echilibrare)")
        print("="*70)
        # Afisam tabelul cu costurile (fara alocare inca)
        print_transport_table(
            np.zeros((m, n)), costs_b, set(), supply_b, demand_b,
            "Tabel costuri initial", source_labels, dest_labels
        )

    # ---- NORD-VEST ----
    if verbose:
        print("\n" + "="*70)
        print("  [PAS 1] METODA NORD-VEST (N-V)  =>  Solutie initiala")
        print("="*70)
        print(f"  V = m + n - 1 = {m} + {n} - 1 = {target}")

    x, basis, is_deg, costs_used, supply_used, demand_used = northwest_corner(
        supply_b, demand_b, costs_b, verbose
    )

    nc = len(basis)
    if verbose:
        print(f"  NC (celule bazice) = {nc}")
        print_transport_table(x, costs_b, basis, supply_b, demand_b,
                              "Solutia initiala N-V", source_labels, dest_labels)
        cost0 = float(np.sum(x * costs_b))
        print(f"  Cost initial f0 = {fmt(cost0)}")

    # ---- ITERATII ----
    if verbose:
        print("\n" + "="*70)
        print("  [PAS 2] ITERATII (Algoritmul de rezolvare PTES)")
        print("="*70)

    for iteration in range(1, max_iter + 1):
        total_cost = float(np.sum(x * costs_b))

        if len(basis) != target:
            raise RuntimeError(f"Baza invalida la iteratia {iteration}: {len(basis)} != {target}")

        u, v = compute_potentials(costs_b, basis)

        if verbose:
            print(f"\n{'─'*70}")
            print(f"  Iteratia I_{iteration-1}  |  f = {fmt(total_cost)}")
            print(f"{'─'*70}")
            print_transport_table(x, costs_b, basis, supply_b, demand_b,
                                  f"Tabelul T_{iteration-1}", source_labels, dest_labels)

        is_optimal, entering, min_delta = print_potential_table(
            costs_b, basis, u, v, source_labels, dest_labels
        ) if verbose else _check_optimal(costs_b, basis, u, v)

        if is_optimal:
            if verbose:
                print(f"\n  => I_{iteration-1} = I_stop  =>  STOP")
            break

        if not verbose:
            _, entering, min_delta = _check_optimal_full(costs_b, basis, u, v)

        # Circuit
        cycle_cells = extract_cycle_cells(basis, entering)
        cycle_path  = order_cycle(cycle_cells, entering)

        if verbose:
            disp = [(r+1, c+1) for (r, c) in cycle_path]
            signs = ["+θ" if k % 2 == 0 else "-θ" for k in range(len(disp))]
            circuit_str = " -> ".join(f"({r},{c}){s}" for (r,c),s in zip(disp, signs))
            print(f"\n  Circuit: {circuit_str}")

        theta, leaving = pivot(x, basis, cycle_path)

        if verbose:
            print(f"  θ = min{{celule (-)}}")
            print(f"      = {fmt(theta)}")
            print(f"  Celula care iese din baza: ({leaving[0]+1}, {leaving[1]+1})")
            new_cost = float(np.sum(x * costs_b))
            print(f"  f_{iteration} = {fmt(new_cost)}")

    else:
        raise RuntimeError("S-a depasit numarul maxim de iteratii!")

    # ---- REZULTAT FINAL ----
    final_cost = float(np.sum(x * costs_b))

    if verbose:
        print("\n" + "*"*70)
        print("  [PAS 3] SOLUTIA OPTIMA")
        print("*"*70)
        print_transport_table(x, costs_b, basis, supply_b, demand_b,
                              "Solutia optima", source_labels, dest_labels)
        print(f"\n  Cost minim = f_min = {fmt(final_cost)}")

        # Afisam alocarea clara
        print("\n  Alocarile optime x_ij:")
        for (r, c) in sorted(basis):
            if x[r, c] > TOL:
                print(f"    x_{r+1}{c+1} = {fmt(x[r,c])}", end="")
                if added and ((added[0]=="destination" and c==n-1) or (added[0]=="source" and r==m-1)):
                    print("  (fictiv)", end="")
                print()

        if added:
            kind, idx, qty = added
            if kind == "destination":
                print(f"\n  Nota: B{idx+1}* este destinatie fictiva (surplus la furnizori).")
            else:
                print(f"\n  Nota: A{idx+1}* este sursa fictiva (deficit la beneficiari).")

    return x, final_cost


def _check_optimal(costs, basis, u, v):
    m, n = costs.shape
    for r in range(m):
        for c in range(n):
            if (r, c) not in basis:
                delta = costs[r, c] - (u[r] + v[c])
                if delta < -TOL:
                    return False, (r, c), delta
    return True, None, 0.0


def _check_optimal_full(costs, basis, u, v):
    m, n = costs.shape
    best = (None, float('inf'))
    for r in range(m):
        for c in range(n):
            if (r, c) not in basis:
                delta = costs[r, c] - (u[r] + v[c])
                if delta < best[1]:
                    best = ((r, c), delta)
    cell, d = best
    return d >= -TOL, cell, d


# =========================================================
# MAIN
# =========================================================
def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   PROBLEMA DE TRANSPORT — SOLVER DIDACTIC (metoda potentialelor) ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("Introduceti datele problemei:")

    try:
        m = int(input("  Nr. surse (furnizori) m = ").strip())
        n = int(input("  Nr. destinatii (beneficiari) n = ").strip())
        if m <= 0 or n <= 0:
            raise ValueError("m si n trebuie sa fie > 0.")

        supply_raw = input(f"  Disponibilitati a_1..a_{m} (separate prin spatiu): ").split()
        if len(supply_raw) != m:
            raise ValueError(f"Trebuie exact {m} disponibilitati.")
        supply = [float(v) for v in supply_raw]

        demand_raw = input(f"  Necesare    b_1..b_{n} (separate prin spatiu): ").split()
        if len(demand_raw) != n:
            raise ValueError(f"Trebuie exact {n} necesare.")
        demand = [float(v) for v in demand_raw]

        print(f"  Matricea costurilor c_ij  ({m} linii x {n} coloane):")
        costs = []
        for i in range(m):
            row_raw = input(f"    Linia {i+1} (A{i+1}): ").split()
            if len(row_raw) != n:
                raise ValueError(f"Linia {i+1} trebuie sa contina {n} valori.")
            costs.append([float(v) for v in row_raw])

        solve_transportation(supply, demand, costs, verbose=True)

    except KeyboardInterrupt:
        print("\n\nIntrerupt de utilizator.")
    except Exception as e:
        print(f"\nEroare: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
