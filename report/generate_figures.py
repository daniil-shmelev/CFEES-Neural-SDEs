"""Generate the figures referenced by rna_torus_benchmark.tex."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPORT_DIR.parent / "results"

BENCHMARK_JSON = RESULTS_DIR / "rna_integrator_benchmark.json"

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

SOLVER_LABEL = {"cg2": "CG2", "cfees25": "CFEES25", "cg4": "CG4", "cfees27": "CFEES27"}
SOLVER_COLOR = {
    "cg2": "#1f77b4",
    "cfees25": "#d62728",
    "cg4": "#2ca02c",
    "cfees27": "#9467bd",
}
SOLVER_MARKER = {"cg2": "o", "cfees25": "s", "cg4": "^", "cfees27": "D"}


def _load_benchmark() -> dict[str, dict]:
    if not BENCHMARK_JSON.exists():
        return {}
    return json.loads(BENCHMARK_JSON.read_text())


def _cells(records: dict[str, dict]) -> list[dict]:
    return [r for r in records.values() if "error" not in r]


def _cells_at(records: dict[str, dict], *, num_angles: int | None = None, n_steps: int | None = None) -> list[dict]:
    out = []
    for r in _cells(records):
        if num_angles is not None and int(r["num_angles"]) != num_angles:
            continue
        if n_steps is not None and int(r["n_steps"]) != n_steps:
            continue
        out.append(r)
    return out


def _group_by_solver(rows: list[dict], *, x_field: str, y_field: str, y_scale: float = 1.0) -> dict[str, list[tuple[float, float]]]:
    by_solver: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        y = r.get(y_field)
        if y is None:
            continue
        by_solver[r["solver"]].append((float(r[x_field]), float(y) * y_scale))
    for solver in by_solver:
        by_solver[solver].sort()
    return by_solver


def _plot_lines(ax, by_solver: dict[str, list[tuple[float, float]]]) -> None:
    for solver in sorted(by_solver):
        items = by_solver[solver]
        xs = [x for x, _ in items]
        ys = [y for _, y in items]
        ax.plot(
            xs,
            ys,
            marker=SOLVER_MARKER.get(solver, "o"),
            color=SOLVER_COLOR.get(solver, None),
            label=SOLVER_LABEL.get(solver, solver),
            markersize=6,
            linewidth=1.5,
        )


def fig_compute(out_path: Path) -> None:
    """Compile-time RSS footprint: vs n_steps (d=7) and vs d (n_steps=50)."""
    records = _load_benchmark()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    n_rows = _cells_at(records, num_angles=7)
    n_by_solver = _group_by_solver(n_rows, x_field="n_steps", y_field="compile_footprint_mb")
    _plot_lines(axes[0], n_by_solver)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"integration steps $n_{\mathrm{steps}}$")
    axes[0].set_ylabel("RSS compile footprint (MiB)")
    axes[0].set_title(r"fixed $d=7$, batch $32$")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    k_rows = _cells_at(records, n_steps=50)
    k_by_solver = _group_by_solver(k_rows, x_field="num_angles", y_field="compile_footprint_mb")
    _plot_lines(axes[1], k_by_solver)
    axes[1].set_xlabel(r"torus dimension $d$")
    axes[1].set_ylabel("RSS compile footprint (MiB)")
    axes[1].set_title(r"fixed $n_{\mathrm{steps}} = 50$, batch $32$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_timing(out_path: Path) -> None:
    """Forward+backward wall-clock: vs n_steps (d=7) and vs d (n_steps=50)."""
    records = _load_benchmark()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    n_rows = _cells_at(records, num_angles=7)
    n_by_solver = _group_by_solver(n_rows, x_field="n_steps", y_field="mean_s", y_scale=1000.0)
    _plot_lines(axes[0], n_by_solver)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"integration steps $n_{\mathrm{steps}}$")
    axes[0].set_ylabel("forward+backward (ms)")
    axes[0].set_title(r"fixed $d=7$, batch $32$")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    k_rows = _cells_at(records, n_steps=50)
    k_by_solver = _group_by_solver(k_rows, x_field="num_angles", y_field="mean_s", y_scale=1000.0)
    _plot_lines(axes[1], k_by_solver)
    axes[1].set_xlabel(r"torus dimension $d$")
    axes[1].set_ylabel("forward+backward (ms)")
    axes[1].set_title(r"fixed $n_{\mathrm{steps}} = 50$, batch $32$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig_hlo_scratch(out_path: Path) -> None:
    """XLA compile-time scratch buffer (HLO): vs n_steps (d=7) and vs d (n_steps=50)."""
    records = _load_benchmark()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    n_rows = _cells_at(records, num_angles=7)
    n_by_solver = _group_by_solver(
        n_rows, x_field="n_steps", y_field="temp_bytes", y_scale=1.0 / (1 << 20)
    )
    _plot_lines(axes[0], n_by_solver)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"integration steps $n_{\mathrm{steps}}$")
    axes[0].set_ylabel("XLA scratch (MiB)")
    axes[0].set_title(r"fixed $d=7$, batch $32$")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    k_rows = _cells_at(records, n_steps=50)
    k_by_solver = _group_by_solver(
        k_rows, x_field="num_angles", y_field="temp_bytes", y_scale=1.0 / (1 << 20)
    )
    _plot_lines(axes[1], k_by_solver)
    axes[1].set_xlabel(r"torus dimension $d$")
    axes[1].set_ylabel("XLA scratch (MiB)")
    axes[1].set_title(r"fixed $n_{\mathrm{steps}} = 50$, batch $32$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    fig_compute(REPORT_DIR / "fig_compute.pdf")
    fig_timing(REPORT_DIR / "fig_timing.pdf")
    fig_hlo_scratch(REPORT_DIR / "fig_hlo_scratch.pdf")
    print(f"wrote figures to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
