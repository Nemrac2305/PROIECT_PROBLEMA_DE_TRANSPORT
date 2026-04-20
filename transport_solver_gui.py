from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple
from collections import defaultdict

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


TOL = 1e-9
AZURE_DARK = "#1e3a8a"
AZURE = "#2563eb"
AZURE_SOFT = "#dbeafe"
AZURE_BG = "#eff6ff"
SLATE = "#334155"
SUCCESS = "#d9f2d9"
SUCCESS_BG = "#eaf8ea"
DANGER = "#f8d7da"
DANGER_BG = "#fdecef"
CYCLE_PLUS = "#dcfce7"
CYCLE_MINUS = "#fee2e2"
CYCLE_ENTER = "#bfdbfe"
CYCLE_LEAVE = "#fecaca"


@dataclass
class TransportProblem:
    supply: List[float]
    demand: List[float]
    costs: np.ndarray
    source_labels: List[str]
    dest_labels: List[str]
    added_dummy: Optional[Tuple[str, int, float]] = None


@dataclass
class IterationState:
    iteration: int
    x: np.ndarray
    basis: Set[Tuple[int, int]]
    total_cost: float
    u: np.ndarray
    v: np.ndarray
    equations: List[str]
    delta_matrix: np.ndarray
    deltas: Dict[Tuple[int, int], float]
    entering: Optional[Tuple[int, int]]
    leaving: Optional[Tuple[int, int]]
    theta: Optional[float]
    cycle_path: List[Tuple[int, int]]
    status: str
    summary: str
    log_lines: List[str]
    cost_history: List[float]


class TransportSolverDidactic:
    """Motor didactic pentru problema de transport (minimizare)."""

    def validate_input(self, supply: List[float], demand: List[float], costs: List[List[float]]) -> None:
        if not supply:
            raise ValueError("Lista disponibilitatilor nu poate fi vida.")
        if not demand:
            raise ValueError("Lista necesarelor nu poate fi vida.")

        supply_arr = np.array(supply, dtype=float)
        demand_arr = np.array(demand, dtype=float)
        costs_arr = np.array(costs, dtype=float)

        if costs_arr.ndim != 2:
            raise ValueError("Matricea costurilor trebuie sa fie bidimensionala.")
        if costs_arr.shape != (len(supply), len(demand)):
            raise ValueError(
                f"Dimensiunea matricei costurilor este {costs_arr.shape}, "
                f"dar ar trebui sa fie ({len(supply)}, {len(demand)})."
            )
        if not np.all(np.isfinite(supply_arr)) or not np.all(np.isfinite(demand_arr)) or not np.all(np.isfinite(costs_arr)):
            raise ValueError("Toate valorile trebuie sa fie numerice finite.")
        if np.any(supply_arr < 0):
            raise ValueError("Disponibilitatile trebuie sa fie nenegative.")
        if np.any(demand_arr < 0):
            raise ValueError("Necesarele trebuie sa fie nenegative.")
        if np.any(costs_arr < 0):
            raise ValueError("Costurile trebuie sa fie nenegative.")

    def balance_problem(
        self,
        supply: List[float],
        demand: List[float],
        costs: List[List[float]],
        source_labels: Optional[List[str]] = None,
        dest_labels: Optional[List[str]] = None,
    ) -> TransportProblem:
        self.validate_input(supply, demand, costs)
        supply_bal = [float(x) for x in supply]
        demand_bal = [float(x) for x in demand]
        cost_bal = np.array(costs, dtype=float)

        m, n = cost_bal.shape
        source_labels = source_labels or [f"S{i + 1}" for i in range(m)]
        dest_labels = dest_labels or [f"D{j + 1}" for j in range(n)]
        added_dummy = None

        total_supply = float(sum(supply_bal))
        total_demand = float(sum(demand_bal))

        if abs(total_supply - total_demand) <= TOL:
            return TransportProblem(supply_bal, demand_bal, cost_bal, source_labels, dest_labels, None)

        if total_supply > total_demand:
            diff = total_supply - total_demand
            cost_bal = np.hstack([cost_bal, np.zeros((m, 1), dtype=float)])
            demand_bal.append(diff)
            dest_labels = dest_labels + [f"Df{len(dest_labels) + 1}"]
            added_dummy = ("destination", len(demand_bal) - 1, diff)
        else:
            diff = total_demand - total_supply
            cost_bal = np.vstack([cost_bal, np.zeros((1, n), dtype=float)])
            supply_bal.append(diff)
            source_labels = source_labels + [f"Sf{len(source_labels) + 1}"]
            added_dummy = ("source", len(supply_bal) - 1, diff)

        return TransportProblem(supply_bal, demand_bal, cost_bal, source_labels, dest_labels, added_dummy)

    def creates_cycle(self, basis: Set[Tuple[int, int]], m: int, n: int) -> bool:
        parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

        def find(node: Tuple[str, int]) -> Tuple[str, int]:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: Tuple[str, int], b: Tuple[str, int]) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            parent[rb] = ra
            return True

        for r in range(m):
            parent[("r", r)] = ("r", r)
        for c in range(n):
            parent[("c", c)] = ("c", c)

        for r, c in basis:
            if not union(("r", r), ("c", c)):
                return True
        return False

    def northwest_corner(self, problem: TransportProblem) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
        supply = problem.supply.copy()
        demand = problem.demand.copy()
        m, n = len(supply), len(demand)
        x = np.zeros((m, n), dtype=float)
        basis: Set[Tuple[int, int]] = set()
        i = 0
        j = 0

        while i < m and j < n:
            amount = min(supply[i], demand[j])
            x[i, j] = amount
            basis.add((i, j))
            supply[i] -= amount
            demand[j] -= amount

            row_done = abs(supply[i]) <= TOL
            col_done = abs(demand[j]) <= TOL

            if row_done and col_done:
                if i < m - 1 and j < n - 1:
                    candidate = (i, j + 1)
                    trial = set(basis)
                    trial.add(candidate)
                    if not self.creates_cycle(trial, m, n):
                        basis.add(candidate)
                    else:
                        candidate = (i + 1, j)
                        trial = set(basis)
                        trial.add(candidate)
                        if not self.creates_cycle(trial, m, n):
                            basis.add(candidate)
                i += 1
                j += 1
            elif row_done:
                i += 1
            elif col_done:
                j += 1
            else:
                raise RuntimeError("Metoda Nord-Vest a intrat intr-o stare imposibila.")

        target = m + n - 1
        if len(basis) < target:
            for r in range(m):
                for c in range(n):
                    if (r, c) in basis:
                        continue
                    trial = set(basis)
                    trial.add((r, c))
                    if not self.creates_cycle(trial, m, n):
                        basis.add((r, c))
                    if len(basis) == target:
                        break
                if len(basis) == target:
                    break

        if len(basis) != target:
            raise RuntimeError(
                f"Baza initiala este invalida: {len(basis)} celule bazice in loc de {target}."
            )
        x[np.abs(x) <= TOL] = 0.0
        return x, basis

    def compute_potentials(
        self, problem: TransportProblem, basis: Set[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        m, n = problem.costs.shape
        u: List[Optional[float]] = [None] * m
        v: List[Optional[float]] = [None] * n
        equations: List[str] = ["u1 = 0"]
        u[0] = 0.0

        changed = True
        while changed:
            changed = False
            for r, c in sorted(basis):
                cost = problem.costs[r, c]
                if u[r] is not None and v[c] is None:
                    v[c] = cost - u[r]
                    equations.append(
                        f"u{r + 1} + v{c + 1} = c{r + 1}{c + 1} = {self.format_num(cost)}  =>  "
                        f"v{c + 1} = {self.format_num(v[c])}"
                    )
                    changed = True
                elif v[c] is not None and u[r] is None:
                    u[r] = cost - v[c]
                    equations.append(
                        f"u{r + 1} + v{c + 1} = c{r + 1}{c + 1} = {self.format_num(cost)}  =>  "
                        f"u{r + 1} = {self.format_num(u[r])}"
                    )
                    changed = True

        if any(value is None for value in u) or any(value is None for value in v):
            raise RuntimeError("Baza este deconectata. Potentialele nu pot fi calculate complet.")

        return np.array(u, dtype=float), np.array(v, dtype=float), equations

    def compute_deltas(
        self, problem: TransportProblem, basis: Set[Tuple[int, int]], u: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], Optional[Tuple[int, int]]]:
        m, n = problem.costs.shape
        delta_matrix = np.full((m, n), np.nan, dtype=float)
        deltas: Dict[Tuple[int, int], float] = {}
        entering: Optional[Tuple[int, int]] = None
        min_delta = 0.0

        for r in range(m):
            for c in range(n):
                if (r, c) in basis:
                    continue
                delta = float(problem.costs[r, c] - (u[r] + v[c]))
                delta_matrix[r, c] = delta
                deltas[(r, c)] = delta
                if entering is None or delta < min_delta - TOL:
                    entering = (r, c)
                    min_delta = delta

        if entering is not None and min_delta >= -TOL:
            entering = None
        return delta_matrix, deltas, entering

    def extract_cycle_cells(self, basis: Set[Tuple[int, int]], entering: Tuple[int, int]) -> Set[Tuple[int, int]]:
        cells = set(basis)
        cells.add(entering)
        changed = True
        while changed:
            changed = False
            row_count = defaultdict(int)
            col_count = defaultdict(int)
            for r, c in cells:
                row_count[r] += 1
                col_count[c] += 1
            to_remove = {
                cell for cell in cells if row_count[cell[0]] < 2 or col_count[cell[1]] < 2
            }
            if to_remove:
                cells -= to_remove
                changed = True
        if entering not in cells or len(cells) < 4:
            raise RuntimeError("Nu s-a putut determina circuitul de ameliorare.")
        return cells

    def order_cycle(self, cycle_cells: Set[Tuple[int, int]], entering: Tuple[int, int]) -> List[Tuple[int, int]]:
        row_map: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        col_map: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for cell in cycle_cells:
            row_map[cell[0]].append(cell)
            col_map[cell[1]].append(cell)

        for r, row_cells in row_map.items():
            if len(row_cells) != 2:
                raise RuntimeError(f"Linia {r + 1} nu are exact 2 celule in circuit.")
        for c, col_cells in col_map.items():
            if len(col_cells) != 2:
                raise RuntimeError(f"Coloana {c + 1} nu are exact 2 celule in circuit.")

        def attempt(start_with_row: bool) -> Optional[List[Tuple[int, int]]]:
            path = [entering]
            current = entering
            on_row = start_with_row
            for _ in range(len(cycle_cells) - 1):
                candidates = row_map[current[0]] if on_row else col_map[current[1]]
                a, b = candidates
                nxt = b if a == current else a
                if nxt in path:
                    return None
                path.append(nxt)
                current = nxt
                on_row = not on_row
            candidates = row_map[current[0]] if on_row else col_map[current[1]]
            a, b = candidates
            nxt = b if a == current else a
            return path if nxt == entering else None

        path = attempt(True)
        if path is None:
            path = attempt(False)
        if path is None:
            raise RuntimeError("Nu s-a putut ordona circuitul de ameliorare.")
        return path

    def pivot(
        self, x: np.ndarray, basis: Set[Tuple[int, int]], cycle_path: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, Set[Tuple[int, int]], float, Tuple[int, int]]:
        x_new = np.array(x, dtype=float, copy=True)
        basis_new = set(basis)
        minus_cells = [cycle_path[i] for i in range(1, len(cycle_path), 2)]
        theta = min(x_new[r, c] for r, c in minus_cells)
        leaving_candidates = [(r, c) for r, c in minus_cells if abs(x_new[r, c] - theta) <= TOL]
        leaving = sorted(leaving_candidates)[0]

        for idx, (r, c) in enumerate(cycle_path):
            if idx % 2 == 0:
                x_new[r, c] += theta
            else:
                x_new[r, c] -= theta

        x_new[np.abs(x_new) <= TOL] = 0.0
        basis_new.add(cycle_path[0])
        basis_new.remove(leaving)
        return x_new, basis_new, float(theta), leaving

    def total_cost(self, x: np.ndarray, costs: np.ndarray) -> float:
        return float(np.sum(x * costs))

    def iteration_generator(self, problem: TransportProblem) -> Generator[IterationState, None, None]:
        x, basis = self.northwest_corner(problem)
        cost_history = [self.total_cost(x, problem.costs)]
        iteration = 0

        while True:
            u, v, equations = self.compute_potentials(problem, basis)
            delta_matrix, deltas, entering = self.compute_deltas(problem, basis, u, v)
            cycle_path: List[Tuple[int, int]] = []
            theta = None
            leaving = None
            log_lines = [
                f"Iteratia I{iteration}",
                f"Cost curent f(x) = {self.format_num(cost_history[-1])}",
                f"Celule bazice = {len(basis)} (tinta m+n-1 = {problem.costs.shape[0] + problem.costs.shape[1] - 1})",
            ]

            if entering is None:
                status = "optimal"
                summary = "Toate valorile Delta_ij pentru celulele nebazice sunt >= 0. Solutia este optima."
                log_lines.append(summary)
                state = IterationState(
                    iteration=iteration,
                    x=np.array(x, copy=True),
                    basis=set(basis),
                    total_cost=cost_history[-1],
                    u=u,
                    v=v,
                    equations=equations,
                    delta_matrix=delta_matrix,
                    deltas=deltas,
                    entering=None,
                    leaving=None,
                    theta=None,
                    cycle_path=[],
                    status=status,
                    summary=summary,
                    log_lines=log_lines,
                    cost_history=list(cost_history),
                )
                yield state
                return

            cycle_cells = self.extract_cycle_cells(basis, entering)
            cycle_path = self.order_cycle(cycle_cells, entering)
            x_next, basis_next, theta, leaving = self.pivot(x, basis, cycle_path)
            next_cost = self.total_cost(x_next, problem.costs)
            improvement = cost_history[-1] - next_cost
            summary = (
                f"Celula de intrare este ({entering[0] + 1}, {entering[1] + 1}) cu Delta = {self.format_num(deltas[entering])}. "
                f"Circuitul poligonal determina theta = {self.format_num(theta)} si iesirea celulei "
                f"({leaving[0] + 1}, {leaving[1] + 1}). Costul va scadea cu {self.format_num(improvement)}."
            )
            log_lines.extend(
                [
                    f"Celula de intrare: ({entering[0] + 1}, {entering[1] + 1}) cu Delta = {self.format_num(deltas[entering])}",
                    "Circuit: " + self.describe_cycle(cycle_path),
                    f"theta = {self.format_num(theta)}",
                    f"Celula care iese din baza: ({leaving[0] + 1}, {leaving[1] + 1})",
                    f"Cost dupa pivot: {self.format_num(next_cost)}",
                ]
            )
            state = IterationState(
                iteration=iteration,
                x=np.array(x, copy=True),
                basis=set(basis),
                total_cost=cost_history[-1],
                u=u,
                v=v,
                equations=equations,
                delta_matrix=delta_matrix,
                deltas=deltas,
                entering=entering,
                leaving=leaving,
                theta=theta,
                cycle_path=cycle_path,
                status="continue",
                summary=summary,
                log_lines=log_lines,
                cost_history=list(cost_history),
            )
            yield state
            x = x_next
            basis = basis_next
            cost_history.append(next_cost)
            iteration += 1

    @staticmethod
    def format_num(value: float) -> str:
        if value is None:
            return "-"
        if abs(value) <= TOL:
            return "0"
        if abs(value - round(value)) <= TOL:
            return str(int(round(value)))
        return f"{value:.4f}"

    def describe_cycle(self, cycle_path: List[Tuple[int, int]]) -> str:
        parts = []
        for idx, (r, c) in enumerate(cycle_path):
            sign = "+" if idx % 2 == 0 else "-"
            parts.append(f"({r + 1},{c + 1}){sign}")
        if cycle_path:
            parts.append(f"({cycle_path[0][0] + 1},{cycle_path[0][1] + 1})")
        return " -> ".join(parts)


class NumericMatrixGrid(ttk.Frame):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title_label = tk.Label(
            self,
            text=title,
            bg=AZURE_DARK,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=7,
            anchor="w",
        )
        self.title_label.pack(fill="x", pady=(0, 6))
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True)

    def clear(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()

    def set_data(
        self,
        data: Optional[np.ndarray],
        row_headers: Optional[List[str]] = None,
        col_headers: Optional[List[str]] = None,
        highlight_cells: Optional[Set[Tuple[int, int]]] = None,
        special_text: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> None:
        self.clear()
        if data is None:
            tk.Label(self.body, text="-", bg="white", fg="#64748b", width=12, relief="solid", bd=1).grid(row=0, column=0)
            return

        arr = np.array(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows, cols = arr.shape
        row_headers = row_headers or [str(i + 1) for i in range(rows)]
        col_headers = col_headers or [str(j + 1) for j in range(cols)]
        highlight_cells = highlight_cells or set()
        special_text = special_text or {}

        def make_label(r: int, c: int, text: str, bg: str, fg: str = "#0f172a", bold: bool = False) -> None:
            lbl = tk.Label(
                self.body,
                text=text,
                bg=bg,
                fg=fg,
                relief="solid",
                bd=1,
                padx=8,
                pady=6,
                font=("Segoe UI", 9, "bold" if bold else "normal"),
                width=12,
            )
            lbl.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)

        make_label(0, 0, "", AZURE_SOFT, AZURE_DARK, True)
        for j, header in enumerate(col_headers, start=1):
            make_label(0, j, header, AZURE_SOFT, AZURE_DARK, True)
        for i in range(rows):
            make_label(i + 1, 0, row_headers[i], AZURE_BG, AZURE_DARK, True)
            for j in range(cols):
                text = special_text.get((i, j))
                if text is None:
                    if np.isnan(arr[i, j]):
                        text = "B"
                    else:
                        text = TransportSolverDidactic.format_num(arr[i, j])
                bg = SUCCESS_BG if (i, j) in highlight_cells else "white"
                fg = "#64748b" if text == "B" else "#0f172a"
                make_label(i + 1, j + 1, text, bg, fg)

        for i in range(rows + 1):
            self.body.grid_rowconfigure(i, weight=1)
        for j in range(cols + 1):
            self.body.grid_columnconfigure(j, weight=1)


class TransportDisplayGrid(ttk.Frame):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.title_label = tk.Label(
            self,
            text=title,
            bg=AZURE_DARK,
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=7,
            anchor="w",
        )
        self.title_label.pack(fill="x", pady=(0, 6))
        self.body = ttk.Frame(self)
        self.body.pack(fill="both", expand=True)

    def clear(self) -> None:
        for child in self.body.winfo_children():
            child.destroy()

    def set_problem_state(self, problem: TransportProblem, x: np.ndarray, basis: Set[Tuple[int, int]], entering: Optional[Tuple[int, int]] = None,
                          leaving: Optional[Tuple[int, int]] = None, cycle_path: Optional[List[Tuple[int, int]]] = None) -> None:
        self.clear()
        cycle_path = cycle_path or []
        plus_cells = {cycle_path[i] for i in range(0, len(cycle_path), 2)}
        minus_cells = {cycle_path[i] for i in range(1, len(cycle_path), 2)}
        m, n = problem.costs.shape

        def header(r: int, c: int, text: str, bg: str = AZURE_SOFT, fg: str = AZURE_DARK) -> None:
            lbl = tk.Label(
                self.body,
                text=text,
                bg=bg,
                fg=fg,
                relief="solid",
                bd=1,
                padx=6,
                pady=6,
                font=("Segoe UI", 9, "bold"),
            )
            lbl.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)

        header(0, 0, "S/D")
        for j, label in enumerate(problem.dest_labels, start=1):
            header(0, j, label)
        header(0, n + 1, "a_i")

        for i, label in enumerate(problem.source_labels, start=1):
            header(i, 0, label, AZURE_BG)
        for j, value in enumerate(problem.demand, start=1):
            header(m + 1, j, TransportSolverDidactic.format_num(value), AZURE_BG)
        header(m + 1, 0, "b_j", AZURE_BG)
        header(m + 1, n + 1, TransportSolverDidactic.format_num(sum(problem.supply)), AZURE_BG)

        for i in range(m):
            supply_value = TransportSolverDidactic.format_num(problem.supply[i])
            header(i + 1, n + 1, supply_value, AZURE_BG)
            for j in range(n):
                frame = tk.Frame(self.body, bg="white", relief="solid", bd=1, width=120, height=74)
                frame.grid(row=i + 1, column=j + 1, sticky="nsew", padx=1, pady=1)
                frame.grid_propagate(False)

                bg = "white"
                if (i, j) == entering:
                    bg = CYCLE_ENTER
                if (i, j) in plus_cells:
                    bg = CYCLE_PLUS
                if (i, j) in minus_cells:
                    bg = CYCLE_MINUS
                if (i, j) == leaving:
                    bg = CYCLE_LEAVE
                frame.configure(bg=bg)

                cost_label = tk.Label(
                    frame,
                    text=TransportSolverDidactic.format_num(problem.costs[i, j]),
                    bg=bg,
                    fg=AZURE_DARK,
                    font=("Segoe UI", 8, "bold"),
                    anchor="ne",
                )
                cost_label.place(relx=0.95, rely=0.08, anchor="ne")

                allocation = TransportSolverDidactic.format_num(x[i, j]) if abs(x[i, j]) > TOL else "0"
                text = f"[{allocation}]" if (i, j) in basis else allocation
                alloc_label = tk.Label(
                    frame,
                    text=text,
                    bg=bg,
                    fg="#0f172a",
                    font=("Segoe UI", 12, "bold" if (i, j) in basis else "normal"),
                )
                alloc_label.place(relx=0.5, rely=0.55, anchor="center")

        for r in range(m + 2):
            self.body.grid_rowconfigure(r, weight=1)
        for c in range(n + 2):
            self.body.grid_columnconfigure(c, weight=1)


class InputTransportGrid(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.cost_entries: List[List[tk.Entry]] = []
        self.supply_entries: List[tk.Entry] = []
        self.demand_entries: List[tk.Entry] = []
        self.m = 0
        self.n = 0
        self.vcmd = None

    def build(self, m: int, n: int, vcmd) -> None:
        self.vcmd = vcmd
        self.m = m
        self.n = n
        for child in self.winfo_children():
            child.destroy()
        self.cost_entries = []
        self.supply_entries = []
        self.demand_entries = []

        def head(r: int, c: int, text: str, bg: str = AZURE_SOFT) -> None:
            lbl = tk.Label(
                self,
                text=text,
                bg=bg,
                fg=AZURE_DARK,
                font=("Segoe UI", 9, "bold"),
                padx=8,
                pady=6,
                relief="solid",
                bd=1,
            )
            lbl.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)

        head(0, 0, "S/D")
        for j in range(n):
            head(0, j + 1, f"D{j + 1}")
        head(0, n + 1, "a_i")

        for i in range(m):
            head(i + 1, 0, f"S{i + 1}", AZURE_BG)
            row_entries: List[tk.Entry] = []
            for j in range(n):
                entry = tk.Entry(
                    self,
                    width=10,
                    justify="center",
                    bg="white",
                    relief="solid",
                    bd=1,
                    validate="key",
                    validatecommand=vcmd,
                )
                entry.grid(row=i + 1, column=j + 1, sticky="nsew", padx=1, pady=1, ipady=8)
                row_entries.append(entry)
            self.cost_entries.append(row_entries)

            supply_entry = tk.Entry(
                self,
                width=10,
                justify="center",
                bg="#f8fafc",
                relief="solid",
                bd=1,
                validate="key",
                validatecommand=vcmd,
            )
            supply_entry.grid(row=i + 1, column=n + 1, sticky="nsew", padx=1, pady=1, ipady=8)
            self.supply_entries.append(supply_entry)

        head(m + 1, 0, "b_j", AZURE_BG)
        for j in range(n):
            demand_entry = tk.Entry(
                self,
                width=10,
                justify="center",
                bg="#f8fafc",
                relief="solid",
                bd=1,
                validate="key",
                validatecommand=vcmd,
            )
            demand_entry.grid(row=m + 1, column=j + 1, sticky="nsew", padx=1, pady=1, ipady=8)
            self.demand_entries.append(demand_entry)
        head(m + 1, n + 1, "Σ", AZURE_BG)

        for r in range(m + 2):
            self.grid_rowconfigure(r, weight=1)
        for c in range(n + 2):
            self.grid_columnconfigure(c, weight=1)

    def get_data(self) -> Tuple[List[float], List[float], List[List[float]]]:
        supply = [self._parse_entry(entry, f"a{i + 1}") for i, entry in enumerate(self.supply_entries)]
        demand = [self._parse_entry(entry, f"b{j + 1}") for j, entry in enumerate(self.demand_entries)]
        costs = []
        for i, row in enumerate(self.cost_entries):
            costs.append([self._parse_entry(entry, f"c{i + 1}{j + 1}") for j, entry in enumerate(row)])
        return supply, demand, costs

    def set_data(self, supply: List[float], demand: List[float], costs: List[List[float]]) -> None:
        if len(supply) != self.m or len(demand) != self.n:
            raise ValueError("Dimensiunile datelor nu corespund grilei curente.")
        for i in range(self.m):
            self.supply_entries[i].delete(0, tk.END)
            self.supply_entries[i].insert(0, str(supply[i]))
            for j in range(self.n):
                self.cost_entries[i][j].delete(0, tk.END)
                self.cost_entries[i][j].insert(0, str(costs[i][j]))
        for j in range(self.n):
            self.demand_entries[j].delete(0, tk.END)
            self.demand_entries[j].insert(0, str(demand[j]))

    @staticmethod
    def _parse_entry(entry: tk.Entry, label: str) -> float:
        raw = entry.get().strip().replace(",", ".")
        if raw == "":
            raise ValueError(f"Campul {label} este gol.")
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Campul {label} trebuie sa contina o valoare numerica.") from exc


class TransportApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Solver Didactic Problema de Transport - Azure Edition")
        self.root.geometry("1600x980")
        self.root.minsize(1360, 840)
        self.root.configure(bg=AZURE_BG)

        self.solver = TransportSolverDidactic()
        self.problem: Optional[TransportProblem] = None
        self.generator: Optional[Generator[IterationState, None, None]] = None
        self.current_state: Optional[IterationState] = None
        self.last_optimal_state: Optional[IterationState] = None

        self.m_var = tk.IntVar(value=3)
        self.n_var = tk.IntVar(value=4)

        self._build_style()
        self._build_ui()
        self._build_input_grid()
        self.load_example_iphone()

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=AZURE_BG)
        style.configure("TLabel", background=AZURE_BG, foreground=SLATE)
        style.configure("TLabelframe", background=AZURE_BG, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=AZURE_BG, foreground=AZURE_DARK, font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", background=AZURE_BG, foreground=AZURE_DARK, font=("Segoe UI", 16, "bold"))
        style.configure("Info.TLabel", background=AZURE_BG, foreground=SLATE)
        style.configure("Big.TLabel", background=AZURE_BG, foreground=AZURE, font=("Segoe UI", 22, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background=AZURE, foreground="white", padding=(10, 7))
        style.map("Accent.TButton", background=[("active", AZURE_DARK), ("pressed", AZURE_DARK)])
        style.configure("Soft.TButton", font=("Segoe UI", 9), background=AZURE_SOFT, foreground=AZURE_DARK, padding=(8, 6))
        style.map("Soft.TButton", background=[("active", "#bfdbfe")])
        style.configure("TNotebook", background=AZURE_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"), background=AZURE_SOFT, foreground=AZURE_DARK)
        style.map("TNotebook.Tab", background=[("selected", "white"), ("active", "#bfdbfe")])
        style.configure("Treeview", rowheight=24, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", background=AZURE_SOFT, foreground=AZURE_DARK, font=("Segoe UI", 9, "bold"))

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook

        self.tab_input = ttk.Frame(notebook, padding=10)
        self.tab_process = ttk.Frame(notebook, padding=10)
        self.tab_result = ttk.Frame(notebook, padding=10)
        self.tab_convergence = ttk.Frame(notebook, padding=10)

        notebook.add(self.tab_input, text="1. Date Initiale")
        notebook.add(self.tab_process, text="2. Proces Iterativ")
        notebook.add(self.tab_result, text="3. Drumul Optim")
        notebook.add(self.tab_convergence, text="4. Analiza de Convergenta")

        self._build_tab_input()
        self._build_tab_process()
        self._build_tab_result()
        self._build_tab_convergence()

    def _build_tab_input(self) -> None:
        self.tab_input.columnconfigure(0, weight=3)
        self.tab_input.columnconfigure(1, weight=2)
        self.tab_input.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(self.tab_input, text="Configurare problema", padding=12)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(controls, text="Numar surse (m):").grid(row=0, column=0, padx=5, pady=4, sticky="w")
        ttk.Entry(controls, textvariable=self.m_var, width=6).grid(row=0, column=1, padx=5, pady=4, sticky="w")
        ttk.Label(controls, text="Numar destinatii (n):").grid(row=0, column=2, padx=5, pady=4, sticky="w")
        ttk.Entry(controls, textvariable=self.n_var, width=6).grid(row=0, column=3, padx=5, pady=4, sticky="w")

        ttk.Button(controls, text="Construieste grila", style="Soft.TButton", command=self._build_input_grid).grid(row=0, column=4, padx=6, pady=4)
        ttk.Button(controls, text="Initializare", style="Accent.TButton", command=self.initialize_problem).grid(row=0, column=5, padx=6, pady=4)
        ttk.Button(controls, text="Pasul Urmator", style="Accent.TButton", command=self.next_step).grid(row=0, column=6, padx=6, pady=4)
        ttk.Button(controls, text="Rezolva complet", style="Accent.TButton", command=self.solve_all).grid(row=0, column=7, padx=6, pady=4)
        ttk.Button(controls, text="Exemplu iPhone 18 Pro", style="Soft.TButton", command=self.load_example_iphone).grid(row=0, column=8, padx=6, pady=4)
        ttk.Button(controls, text="Import JSON", style="Soft.TButton", command=self.import_json).grid(row=0, column=9, padx=6, pady=4)
        ttk.Button(controls, text="Export JSON", style="Soft.TButton", command=self.export_json).grid(row=0, column=10, padx=6, pady=4)

        grid_box = ttk.LabelFrame(self.tab_input, text="Matricea datelor", padding=10)
        grid_box.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        grid_box.rowconfigure(0, weight=1)
        grid_box.columnconfigure(0, weight=1)
        self.input_grid = InputTransportGrid(grid_box)
        self.input_grid.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(self.tab_input)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        help_box = ttk.LabelFrame(right, text="Ghid", padding=8)
        help_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(
            help_box,
            text=(
                "- Completeaza costurile in interiorul matricei.\n"
                "- Introdu disponibilitatile a_i in coloana din dreapta.\n"
                "- Introdu necesarele b_j pe ultima linie.\n"
                "- Initializare echilibreaza automat problema si construieste solutia Nord-Vest."
            ),
            style="Info.TLabel",
            justify="left",
        ).pack(anchor="w")

        self.init_summary = ttk.Label(right, text="Problema nu este initializata.", style="Info.TLabel")
        self.init_summary.grid(row=1, column=0, sticky="nw", pady=(0, 6))

        log_box = ttk.LabelFrame(right, text="Log de initializare", padding=8)
        log_box.grid(row=2, column=0, sticky="nsew")
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.init_log = ScrolledText(log_box, height=14, wrap="word", font=("Consolas", 9))
        self.init_log.grid(row=0, column=0, sticky="nsew")
        self.init_log.configure(state="disabled")

    def _build_tab_process(self) -> None:
        self.tab_process.columnconfigure(0, weight=3)
        self.tab_process.columnconfigure(1, weight=2)
        self.tab_process.rowconfigure(0, weight=1)

        left = ttk.Frame(self.tab_process)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self.transport_grid = TransportDisplayGrid(left, "Tabelul de transport didactic")
        self.transport_grid.grid(row=0, column=0, sticky="nsew")
        self.delta_grid = NumericMatrixGrid(left, "Tabelul Delta (Delta_ij = c_ij - u_i - v_j)")
        self.delta_grid.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        right = ttk.Frame(self.tab_process)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        summary_box = ttk.LabelFrame(right, text="Sumar iteratie", padding=8)
        summary_box.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.iteration_summary = ttk.Label(summary_box, text="Nu exista iteratii rulate.", style="Info.TLabel", justify="left")
        self.iteration_summary.pack(anchor="w")

        potentials_box = ttk.LabelFrame(right, text="Sistemul de ecuatii pentru potentiale", padding=8)
        potentials_box.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        potentials_box.columnconfigure(0, weight=1)
        potentials_box.rowconfigure(0, weight=1)
        self.potentials_text = ScrolledText(potentials_box, wrap="word", font=("Consolas", 9))
        self.potentials_text.grid(row=0, column=0, sticky="nsew")
        self.potentials_text.configure(state="disabled")

        circuit_box = ttk.LabelFrame(right, text="Log de circuit si pivotare", padding=8)
        circuit_box.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        circuit_box.columnconfigure(0, weight=1)
        circuit_box.rowconfigure(0, weight=1)
        self.circuit_text = ScrolledText(circuit_box, wrap="word", font=("Consolas", 9))
        self.circuit_text.grid(row=0, column=0, sticky="nsew")
        self.circuit_text.configure(state="disabled")

        log_box = ttk.LabelFrame(right, text="Jurnal didactic", padding=8)
        log_box.grid(row=3, column=0, sticky="nsew")
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.process_log = ScrolledText(log_box, wrap="word", font=("Consolas", 9))
        self.process_log.grid(row=0, column=0, sticky="nsew")
        self.process_log.configure(state="disabled")

    def _build_tab_result(self) -> None:
        self.tab_result.columnconfigure(0, weight=2)
        self.tab_result.columnconfigure(1, weight=3)
        self.tab_result.rowconfigure(1, weight=1)
        self.tab_result.rowconfigure(2, weight=1)

        top = ttk.Frame(self.tab_result)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(top, text="Cost total:", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.total_cost_label = ttk.Label(top, text="-", style="Big.TLabel")
        self.total_cost_label.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.result_status = ttk.Label(top, text="Nu exista inca o solutie optima.", style="Info.TLabel")
        self.result_status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        table_box = ttk.LabelFrame(self.tab_result, text="Lista finala a livrarilor", padding=8)
        table_box.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        table_box.columnconfigure(0, weight=1)
        table_box.rowconfigure(0, weight=1)
        self.shipments_tree = ttk.Treeview(table_box, columns=("src", "dst", "qty", "unit", "cost"), show="headings")
        for col, text, width in [
            ("src", "Sursa", 130),
            ("dst", "Destinatie", 130),
            ("qty", "Cantitate", 110),
            ("unit", "Cost unitar", 110),
            ("cost", "Cost ruta", 120),
        ]:
            self.shipments_tree.heading(col, text=text)
            self.shipments_tree.column(col, width=width, anchor="center")
        self.shipments_tree.grid(row=0, column=0, sticky="nsew")
        ttk.Scrollbar(table_box, orient="vertical", command=self.shipments_tree.yview).grid(row=0, column=1, sticky="ns")
        self.shipments_tree.configure(yscrollcommand=lambda *args: None)

        network_box = ttk.LabelFrame(self.tab_result, text="Grafic de retea (matrice bipartita)", padding=8)
        network_box.grid(row=1, column=1, rowspan=2, sticky="nsew")
        network_box.rowconfigure(0, weight=1)
        network_box.columnconfigure(0, weight=1)
        self.network_figure = Figure(figsize=(7, 5), dpi=100)
        self.network_ax = self.network_figure.add_subplot(111)
        self.network_canvas = FigureCanvasTkAgg(self.network_figure, master=network_box)
        self.network_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        flow_box = ttk.LabelFrame(self.tab_result, text="Tabel final de transport", padding=8)
        flow_box.grid(row=2, column=0, sticky="nsew", padx=(0, 10), pady=(10, 0))
        flow_box.rowconfigure(0, weight=1)
        flow_box.columnconfigure(0, weight=1)
        self.final_transport_grid = TransportDisplayGrid(flow_box, "Alocari optime")
        self.final_transport_grid.grid(row=0, column=0, sticky="nsew")

    def _build_tab_convergence(self) -> None:
        self.tab_convergence.columnconfigure(0, weight=3)
        self.tab_convergence.columnconfigure(1, weight=2)
        self.tab_convergence.rowconfigure(1, weight=1)

        info = ttk.LabelFrame(self.tab_convergence, text="Interpretare", padding=8)
        info.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.conv_info = ttk.Label(info, text="Graficul va afisa scaderea costului total de la iteratia 0 pana la optim.", style="Info.TLabel")
        self.conv_info.pack(anchor="w")

        chart_box = ttk.LabelFrame(self.tab_convergence, text="Convergenta costului", padding=8)
        chart_box.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        chart_box.rowconfigure(0, weight=1)
        chart_box.columnconfigure(0, weight=1)
        self.conv_figure = Figure(figsize=(7, 4), dpi=100)
        self.conv_ax = self.conv_figure.add_subplot(111)
        self.conv_canvas = FigureCanvasTkAgg(self.conv_figure, master=chart_box)
        self.conv_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        hist_box = ttk.LabelFrame(self.tab_convergence, text="Istoric iteratii", padding=8)
        hist_box.grid(row=1, column=1, sticky="nsew")
        hist_box.rowconfigure(0, weight=1)
        hist_box.columnconfigure(0, weight=1)
        self.history_tree = ttk.Treeview(hist_box, columns=("iter", "cost"), show="headings")
        self.history_tree.heading("iter", text="Iteratia")
        self.history_tree.heading("cost", text="Cost total")
        self.history_tree.column("iter", width=100, anchor="center")
        self.history_tree.column("cost", width=160, anchor="center")
        self.history_tree.grid(row=0, column=0, sticky="nsew")

    def _build_input_grid(self) -> None:
        try:
            m = int(self.m_var.get())
            n = int(self.n_var.get())
            if m <= 0 or n <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Date invalide", "m si n trebuie sa fie numere intregi pozitive.")
            return
        vcmd = (self.root.register(self._validate_numeric), "%P")
        self.input_grid.build(m, n, vcmd)
        self.clear_runtime_views(reset_problem=False)

    def _validate_numeric(self, value: str) -> bool:
        if value == "":
            return True
        value = value.replace(",", ".")
        if value in {"-", "+", ".", "-.", "+."}:
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False

    def initialize_problem(self) -> None:
        try:
            supply, demand, costs = self.input_grid.get_data()
            source_labels = [f"S{i + 1}" for i in range(len(supply))]
            dest_labels = [f"D{j + 1}" for j in range(len(demand))]
            self.problem = self.solver.balance_problem(supply, demand, costs, source_labels, dest_labels)
            self.generator = self.solver.iteration_generator(self.problem)
            self.current_state = None
            self.last_optimal_state = None
            self.clear_runtime_views(reset_problem=False)

            lines = [
                f"Suma disponibilitatilor = {TransportSolverDidactic.format_num(sum(supply))}",
                f"Suma necesarelor = {TransportSolverDidactic.format_num(sum(demand))}",
            ]
            if self.problem.added_dummy is None:
                lines.append("Problema este deja echilibrata.")
            else:
                kind, idx, qty = self.problem.added_dummy
                if kind == "destination":
                    lines.append(
                        f"S-a adaugat o destinatie fictiva {self.problem.dest_labels[idx]} cu necesar {TransportSolverDidactic.format_num(qty)} si costuri 0."
                    )
                else:
                    lines.append(
                        f"S-a adaugat o sursa fictiva {self.problem.source_labels[idx]} cu disponibil {TransportSolverDidactic.format_num(qty)} si costuri 0."
                    )
            x0, basis0 = self.solver.northwest_corner(self.problem)
            lines.append(
                f"Solutia initiala Nord-Vest are costul {TransportSolverDidactic.format_num(self.solver.total_cost(x0, self.problem.costs))} si {len(basis0)} celule bazice."
            )
            self.init_summary.configure(text="\n".join(lines))
            self._set_text(self.init_log, "\n".join(lines))
            self.transport_grid.set_problem_state(self.problem, x0, basis0)
            self.final_transport_grid.set_problem_state(self.problem, x0, basis0)
            self.iteration_summary.configure(text="Problema a fost initializata. Foloseste Pasul Urmator pentru a calcula iteratia I0.")
            self.notebook.select(self.tab_process)
        except Exception as exc:
            messagebox.showerror("Initializare esuata", str(exc))

    def next_step(self) -> None:
        if self.problem is None or self.generator is None:
            self.initialize_problem()
            if self.generator is None:
                return
        try:
            state = next(self.generator)
            self.current_state = state
            if state.status == "optimal":
                self.last_optimal_state = state
            self.display_state(state)
            if state.status == "optimal":
                self.display_optimal_result(state)
                self.notebook.select(self.tab_result)
        except StopIteration:
            return
        except Exception as exc:
            messagebox.showerror("Eroare la iteratie", str(exc))

    def solve_all(self) -> None:
        if self.problem is None or self.generator is None:
            self.initialize_problem()
            if self.generator is None:
                return
        while True:
            try:
                state = next(self.generator)
                self.current_state = state
                if state.status == "optimal":
                    self.last_optimal_state = state
                self.display_state(state)
                if state.status == "optimal":
                    self.display_optimal_result(state)
                    self.notebook.select(self.tab_result)
                    break
            except StopIteration:
                break
            except Exception as exc:
                messagebox.showerror("Eroare la rezolvare", str(exc))
                break

    def display_state(self, state: IterationState) -> None:
        if self.problem is None:
            return
        status_text = "STOP" if state.status == "optimal" else "CONTINUA"
        self.iteration_summary.configure(
            text=(
                f"Iteratia I{state.iteration} | Cost curent = {TransportSolverDidactic.format_num(state.total_cost)} | "
                f"Status = {status_text}\n{state.summary}"
            )
        )
        self.transport_grid.set_problem_state(
            self.problem,
            state.x,
            state.basis,
            entering=state.entering,
            leaving=state.leaving,
            cycle_path=state.cycle_path,
        )
        special = {(r, c): "B" for r, c in state.basis}
        highlights = set()
        if state.entering is not None:
            highlights.add(state.entering)
        self.delta_grid.set_data(
            state.delta_matrix,
            row_headers=self.problem.source_labels,
            col_headers=self.problem.dest_labels,
            highlight_cells=highlights,
            special_text=special,
        )
        potentials = [*state.equations, "", f"u = {[TransportSolverDidactic.format_num(x) for x in state.u]}", f"v = {[TransportSolverDidactic.format_num(x) for x in state.v]}"]
        self._set_text(self.potentials_text, "\n".join(potentials))

        circuit_lines = []
        if state.entering is not None:
            circuit_lines.append(f"Celula de intrare: ({state.entering[0] + 1}, {state.entering[1] + 1})")
            circuit_lines.append("Circuit poligonal: " + self.solver.describe_cycle(state.cycle_path))
            circuit_lines.append(f"theta = {TransportSolverDidactic.format_num(state.theta)}")
            if state.leaving is not None:
                circuit_lines.append(f"Celula care iese din baza: ({state.leaving[0] + 1}, {state.leaving[1] + 1})")
        else:
            circuit_lines.append("Nu mai exista celule de intrare. Solutia curenta este optima.")
        self._set_text(self.circuit_text, "\n".join(circuit_lines))
        self._append_text(self.process_log, "\n".join(state.log_lines) + "\n" + ("-" * 60) + "\n")
        self.update_convergence(state.cost_history)

    def display_optimal_result(self, state: IterationState) -> None:
        if self.problem is None:
            return
        self.total_cost_label.configure(text=f"f(x) = {TransportSolverDidactic.format_num(state.total_cost)}")
        self.result_status.configure(text="S-a atins solutia optima pentru problema de transport.")
        self._clear_tree(self.shipments_tree)
        for i, j in sorted(state.basis):
            qty = state.x[i, j]
            if qty <= TOL:
                continue
            unit_cost = self.problem.costs[i, j]
            route_cost = qty * unit_cost
            self.shipments_tree.insert(
                "",
                "end",
                values=(
                    self.problem.source_labels[i],
                    self.problem.dest_labels[j],
                    TransportSolverDidactic.format_num(qty),
                    TransportSolverDidactic.format_num(unit_cost),
                    TransportSolverDidactic.format_num(route_cost),
                ),
            )
        self.final_transport_grid.set_problem_state(self.problem, state.x, state.basis)
        self.draw_network(state)
        self.update_convergence(state.cost_history)

    def draw_network(self, state: IterationState) -> None:
        if self.problem is None:
            return
        ax = self.network_ax
        ax.clear()
        m = len(self.problem.source_labels)
        n = len(self.problem.dest_labels)
        src_y = list(reversed(np.linspace(0.1, 0.9, m)))
        dst_y = list(reversed(np.linspace(0.1, 0.9, n)))
        src_x, dst_x = 0.15, 0.85

        for i, label in enumerate(self.problem.source_labels):
            ax.scatter([src_x], [src_y[i]], s=700)
            ax.text(src_x - 0.06, src_y[i], f"{label}\n({TransportSolverDidactic.format_num(self.problem.supply[i])})", ha="right", va="center", fontsize=9)
        for j, label in enumerate(self.problem.dest_labels):
            ax.scatter([dst_x], [dst_y[j]], s=700)
            ax.text(dst_x + 0.06, dst_y[j], f"{label}\n({TransportSolverDidactic.format_num(self.problem.demand[j])})", ha="left", va="center", fontsize=9)

        for i, j in sorted(state.basis):
            qty = state.x[i, j]
            if qty <= TOL:
                continue
            ax.plot([src_x, dst_x], [src_y[i], dst_y[j]], linewidth=1.8)
            mid_x = (src_x + dst_x) / 2
            mid_y = (src_y[i] + dst_y[j]) / 2
            ax.text(mid_x, mid_y, TransportSolverDidactic.format_num(qty), fontsize=9, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Retea surse - destinatii")
        ax.axis("off")
        self.network_figure.tight_layout()
        self.network_canvas.draw_idle()

    def update_convergence(self, cost_history: List[float]) -> None:
        self.conv_ax.clear()
        self.conv_ax.set_title("Scaderea costului total")
        self.conv_ax.set_xlabel("Iteratia")
        self.conv_ax.set_ylabel("f(x)")
        self._clear_tree(self.history_tree)
        if cost_history:
            x_vals = list(range(len(cost_history)))
            self.conv_ax.plot(x_vals, cost_history, marker="o")
            self.conv_ax.grid(True, alpha=0.3)
            for idx, cost in enumerate(cost_history):
                self.history_tree.insert("", "end", values=(f"I{idx}", TransportSolverDidactic.format_num(cost)))
            first = cost_history[0]
            last = cost_history[-1]
            self.conv_info.configure(
                text=(
                    f"Cost initial = {TransportSolverDidactic.format_num(first)} | "
                    f"Cost curent/final = {TransportSolverDidactic.format_num(last)} | "
                    f"Reducere = {TransportSolverDidactic.format_num(first - last)}"
                )
            )
        self.conv_figure.tight_layout()
        self.conv_canvas.draw_idle()

    def export_json(self) -> None:
        try:
            supply, demand, costs = self.input_grid.get_data()
        except Exception as exc:
            messagebox.showerror("Export esuat", str(exc))
            return
        payload = {
            "m": len(supply),
            "n": len(demand),
            "supply": supply,
            "demand": demand,
            "costs": costs,
        }
        path = filedialog.asksaveasfilename(
            title="Salveaza problema in JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("Export reusit", "Problema a fost salvata in format JSON.")

    def import_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Importa problema din JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            m = int(payload["m"])
            n = int(payload["n"])
            self.m_var.set(m)
            self.n_var.set(n)
            self._build_input_grid()
            self.input_grid.set_data(payload["supply"], payload["demand"], payload["costs"])
            self.initialize_problem()
        except Exception as exc:
            messagebox.showerror("Import esuat", str(exc))

    def load_example_iphone(self) -> None:
        self.m_var.set(3)
        self.n_var.set(4)
        self._build_input_grid()
        supply = [140, 110, 150]
        demand = [80, 100, 120, 100]
        costs = [
            [7, 10, 5, 15],
            [14, 8, 11, 22],
            [15, 13, 10, 20],
        ]
        self.input_grid.set_data(supply, demand, costs)
        self.initialize_problem()

    def clear_runtime_views(self, reset_problem: bool = True) -> None:
        if reset_problem:
            self.problem = None
            self.generator = None
            self.current_state = None
            self.last_optimal_state = None
        self.iteration_summary.configure(text="Nu exista iteratii rulate.")
        self._set_text(self.potentials_text, "")
        self._set_text(self.circuit_text, "")
        self._set_text(self.process_log, "")
        self.delta_grid.set_data(None)
        self.total_cost_label.configure(text="-")
        self.result_status.configure(text="Nu exista inca o solutie optima.")
        self.transport_grid.clear()
        self.final_transport_grid.clear()
        self._clear_tree(self.shipments_tree)
        self.update_convergence([])
        self.network_ax.clear()
        self.network_ax.set_title("Retea surse - destinatii")
        self.network_ax.axis("off")
        self.network_canvas.draw_idle()

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state="disabled")

    def _append_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state="disabled")

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)


def main() -> None:
    root = tk.Tk()
    app = TransportApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
