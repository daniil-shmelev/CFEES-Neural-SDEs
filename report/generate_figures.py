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


def _simulate_torus_trajectory(
    d: int = 2,
    n_steps: int = 600,
    dt: float = 0.008,
    drift_rate: tuple[float, float] = (0.9, 1.3),
    diffusion_scale: float = 0.35,
    seed: int = 3,
) -> np.ndarray:
    """Integrate a T^d SDE forward via CFEES25 and return the saved path."""
    import jax
    import jax.numpy as jnp
    from diffrax import (
        ControlTerm,
        MultiTerm,
        ODETerm,
        SaveAt,
        VirtualBrownianTree,
        diffeqsolve,
    )
    from georax import CFEES25, GeometricTerm

    from models.torus import Torus

    geometry = Torus(d)
    t1 = n_steps * dt
    key = jax.random.key(seed)
    brownian = VirtualBrownianTree(
        t0=0.0, t1=t1, tol=dt / 4.0, shape=(d,), key=key
    )
    drift_vec = jnp.asarray(drift_rate[:d], dtype=jnp.float32)

    def drift(t, y, args):
        del t, y, args
        return drift_vec

    def diffusion(t, y, args):
        del t, args
        return diffusion_scale * jnp.eye(d, dtype=y.dtype)

    term = GeometricTerm(
        inner=MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian)),
        geometry=geometry,
    )
    ts = jnp.linspace(0.0, t1, n_steps + 1)
    sol = diffeqsolve(
        term,
        CFEES25(),
        t0=0.0,
        t1=t1,
        dt0=dt,
        y0=jnp.zeros((d,), dtype=jnp.float32),
        saveat=SaveAt(ts=ts),
        max_steps=n_steps + 8,
    )
    return np.asarray(sol.ys)


def _embed_torus(theta: float | np.ndarray, phi: float | np.ndarray, R: float, r: float):
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


def _break_wrapped(thetas: np.ndarray) -> np.ndarray:
    """Insert NaNs between samples that cross a $\\pm\\pi$ boundary.

    The solver wraps its state to $(-\\pi, \\pi]$, so two consecutive samples
    near the boundary can be close on the torus while lying on opposite sides
    in the ambient embedding; a straight line between them would cut through
    the interior. Splitting the polyline at those jumps keeps the drawing on
    the manifold.
    """
    out = [thetas[0]]
    for i in range(1, len(thetas)):
        jump = np.any(np.abs(thetas[i] - thetas[i - 1]) > np.pi)
        if jump:
            out.append(np.full_like(thetas[i], np.nan))
        out.append(thetas[i])
    return np.asarray(out)


def fig_torus_trajectory(out_path: Path) -> None:
    """3D embedding of a sample SDE trajectory on $\\Torus^{2}$ via CFEES25."""
    thetas = _simulate_torus_trajectory()
    thetas_broken = _break_wrapped(thetas)

    R, r = 3.0, 1.0
    x, y, z = _embed_torus(thetas_broken[:, 0], thetas_broken[:, 1], R, r)

    u = np.linspace(0.0, 2.0 * np.pi, 80)
    v = np.linspace(0.0, 2.0 * np.pi, 36)
    U, V = np.meshgrid(u, v)
    Xt, Yt, Zt = _embed_torus(U, V, R, r)

    fig = plt.figure(figsize=(5.2, 3.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(
        Xt, Yt, Zt,
        color="0.8",
        linewidth=0.3,
        rstride=2,
        cstride=3,
    )
    ax.plot(x, y, z, color="#d62728", linewidth=1.2, solid_capstyle="round")

    valid = ~np.isnan(x)
    ax.scatter(x[valid][0], y[valid][0], z[valid][0], color="#2ca02c", s=50, depthshade=False, label="start", zorder=5)
    ax.scatter(x[valid][-1], y[valid][-1], z[valid][-1], color="#1f77b4", s=50, depthshade=False, label="end", zorder=5)

    limit = R + r
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-r * 1.8, r * 1.8)
    ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.set_axis_off()
    ax.view_init(elev=32, azim=-55)
    ax.legend(loc="upper left", frameon=False, fontsize=8, bbox_to_anchor=(0.02, 0.95))
    fig.subplots_adjust(left=-0.05, right=1.05, top=1.08, bottom=-0.08)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    fig_compute(REPORT_DIR / "fig_compute.pdf")
    fig_timing(REPORT_DIR / "fig_timing.pdf")
    fig_hlo_scratch(REPORT_DIR / "fig_hlo_scratch.pdf")
    fig_torus_trajectory(REPORT_DIR / "fig_trajectory.pdf")
    print(f"wrote figures to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
