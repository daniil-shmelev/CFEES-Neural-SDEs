"""Generate the figures referenced by rna_torus_benchmark.tex."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPORT_DIR.parent / "results"

HLO_NSWEEP_JSON = RESULTS_DIR / "rna_hlo_n_small.json"
HLO_DIM_JSON = RESULTS_DIR / "rna_hlo_dim_wide.json"
TRAIN_SUMMARY_JSON = RESULTS_DIR / "rna_sweep_full" / "summary.json"

# CG2 hits an internal AssertionError in equinox.internal._loop.bounded
# at this n_steps on our config; CFEES25 reaches at least n_steps = 3000.
CG2_FAILURE_NSTEPS = 300

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)

SOLVER_LABEL = {"cg2": "CG2", "cfees25": "CFEES25", "cg4": "CG4"}
SOLVER_COLOR = {"cg2": "#1f77b4", "cfees25": "#d62728", "cg4": "#2ca02c"}
SOLVER_MARKER = {"cg2": "o", "cfees25": "s", "cg4": "^"}


def _load_hlo(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("cells", [])


def _load_train() -> list[dict]:
    if not TRAIN_SUMMARY_JSON.exists():
        return []
    return json.loads(TRAIN_SUMMARY_JSON.read_text()).get("cells", [])


def fig_compute(out_path: Path) -> None:
    """Two-panel: compile-time XLA scratch vs n_steps and vs torus dimension."""
    n_cells = [
        c for c in _load_hlo(HLO_NSWEEP_JSON)
        if not c.get("oom") and c.get("returncode", 0) == 0 and c.get("residues_per_state") == 1
    ]
    k_cells = [
        c for c in _load_hlo(HLO_DIM_JSON)
        if not c.get("oom") and c.get("returncode", 0) == 0
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    ax = axes[0]
    by_solver: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for c in n_cells:
        mb = float(c["temp_bytes"]) / 2**20
        by_solver[c["solver"]].append((int(c["n_steps"]), mb))
    for solver, items in by_solver.items():
        items.sort()
        xs = [n for n, _ in items]
        ys = [mb for _, mb in items]
        ax.plot(
            xs,
            ys,
            marker=SOLVER_MARKER.get(solver, "o"),
            color=SOLVER_COLOR.get(solver, None),
            label=SOLVER_LABEL.get(solver, solver),
            markersize=6,
            linewidth=1.5,
        )

    # CG2 hits an internal AssertionError at n_steps >= CG2_FAILURE_NSTEPS.
    ax.axvline(
        CG2_FAILURE_NSTEPS,
        color=SOLVER_COLOR["cg2"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label=r"CG2 fails for $n_{\mathrm{steps}} \geq 300$",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"integration steps $n_{\mathrm{steps}}$")
    ax.set_ylabel("XLA scratch (MiB, compile-time)")
    ax.set_title(r"fixed $d=7$, batch $32$, hidden $128$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center right", frameon=True, fontsize=7)
    ax.set_ylim(0, max(c["temp_bytes"] / 2**20 for c in n_cells) * 1.4)

    ax = axes[1]
    by_solver = defaultdict(list)
    for c in k_cells:
        mb = float(c["temp_bytes"]) / 2**20
        by_solver[c["solver"]].append((int(c["num_angles"]), mb))
    for solver, items in by_solver.items():
        items.sort()
        xs = [d for d, _ in items]
        ys = [mb for _, mb in items]
        ax.plot(
            xs,
            ys,
            marker=SOLVER_MARKER.get(solver, "o"),
            color=SOLVER_COLOR.get(solver, None),
            label=SOLVER_LABEL.get(solver, solver),
            markersize=6,
            linewidth=1.5,
        )
    ax.set_xlabel(r"torus dimension $d = 7k$")
    ax.set_ylabel("XLA scratch (MiB, compile-time)")
    ax.set_title(r"fixed $n_{\mathrm{steps}} = 50$, batch $32$, hidden $128$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    fig_compute(REPORT_DIR / "fig_compute.pdf")
    print(f"wrote figures to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
