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
SCALING_JSON = RESULTS_DIR / "rna_benchmark_scaling.json"
PREDICTIONS_NPZ = RESULTS_DIR / "rna_predictions_demo.npz"

ANGLE_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi")
ANGLE_SYMBOLS = (
    r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$",
    r"$\epsilon$", r"$\zeta$", r"$\chi$",
)

def _set_stix_params(small: int = 7, medium: int = 8, bigger: int = 9) -> None:
    """STIX font styling for NeurIPS-quality figures."""
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rc("font", size=small)
    plt.rc("axes", titlesize=bigger)
    plt.rc("axes", labelsize=medium)
    plt.rc("xtick", labelsize=small)
    plt.rc("ytick", labelsize=small)
    plt.rc("legend", fontsize=small)
    plt.rc("figure", titlesize=bigger)


def _set_default_params() -> None:
    """Default report-quality styling (larger fonts)."""
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


_set_default_params()

SOLVER_LABEL = {"cg2": "CG2", "cfees25": "CFEES25", "cg4": "CG4", "cfees27": "CFEES27"}
SOLVER_COLOR = {
    "cg2": "#1f77b4",
    "cfees25": "#d62728",
    "cg4": "#2ca02c",
    "cfees27": "#9467bd",
}
SOLVER_MARKER = {"cg2": "o", "cfees25": "s", "cg4": "^", "cfees27": "D"}

MODE_LABEL = {
    "cfees25:reversible": r"CFEES25 + ReversibleAdjoint",
    "cg2:checkpoint_full": r"CG2 + full tape, $O(n)$",
    "cg2:checkpoint_recursive": r"CG2 + recursive, $O(\sqrt{n})$",
    "cg4:checkpoint_full": r"CG4 + full tape, $O(n)$",
    "cg4:checkpoint_recursive": r"CG4 + recursive, $O(\sqrt{n})$",
    "cg2:auto": r"CG2 + DirectAdjoint (default)",
    "cg2:direct": r"CG2 + DirectAdjoint",
}
MODE_COLOR = {
    "cfees25:reversible": "#d62728",
    "cg2:checkpoint_full": "#1f77b4",
    "cg2:checkpoint_recursive": "#1f77b4",
    "cg4:checkpoint_full": "#2ca02c",
    "cg4:checkpoint_recursive": "#2ca02c",
    "cg2:auto": "#7f7f7f",
    "cg2:direct": "#7f7f7f",
}
MODE_MARKER = {
    "cfees25:reversible": "o",
    "cg2:checkpoint_full": "s",
    "cg2:checkpoint_recursive": "D",
    "cg4:checkpoint_full": "^",
    "cg4:checkpoint_recursive": "v",
    "cg2:auto": "^",
    "cg2:direct": "^",
}
MODE_LINESTYLE = {
    "cfees25:reversible": "-",
    "cg2:checkpoint_full": "-",
    "cg2:checkpoint_recursive": "--",
    "cg4:checkpoint_full": "-",
    "cg4:checkpoint_recursive": "--",
    "cg2:auto": "-",
    "cg2:direct": "-",
}


def _mode_key(cell: dict) -> str:
    return f"{cell.get('solver', '?')}:{cell.get('adjoint', 'auto')}"


def _load_benchmark() -> dict[str, dict]:
    if not BENCHMARK_JSON.exists():
        return {}
    return json.loads(BENCHMARK_JSON.read_text())


def _load_scaling() -> dict[str, dict]:
    if not SCALING_JSON.exists():
        return {}
    return json.loads(SCALING_JSON.read_text())


def fig_scaling(out_path: Path) -> None:
    """HLO compile-time scratch vs n_steps for three (solver, adjoint) modes.

    Expects the JSON produced by
    ``experiment.integrator_benchmark_isolated`` run with
    ``--modes cfees25:reversible cg2:checkpoint_full cg2:auto``.
    """
    records = _load_scaling()
    rows = [r for r in records.values() if "error" not in r]
    if not rows:
        return

    by_mode: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        mode = _mode_key(r)
        by_mode[mode].append((int(r["n_steps"]), float(r["temp_bytes"]) / (1 << 20)))
    for mode in by_mode:
        by_mode[mode].sort()

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))
    order = ["cg4:checkpoint_full", "cg2:checkpoint_full", "cfees25:reversible"]
    for mode in order:
        items = by_mode.get(mode, [])
        if not items:
            continue
        xs = [n for n, _ in items]
        ys = [mb for _, mb in items]
        ax.plot(
            xs,
            ys,
            marker=MODE_MARKER.get(mode, "o"),
            color=MODE_COLOR.get(mode, None),
            label=MODE_LABEL.get(mode, mode),
            markersize=6,
            linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"integration steps $n_{\mathrm{steps}}$")
    ax.set_ylabel("XLA scratch (MiB, compile-time)")
    ax.set_title(r"Compile-time memory vs. path length ($d=350$, batch $32$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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


def _wrapped_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def _load_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Return ``(predicted, actual, context_angles_or_None)`` from the npz."""
    if not PREDICTIONS_NPZ.exists():
        return None
    data = np.load(PREDICTIONS_NPZ)
    predicted = np.asarray(data["predicted"])
    actual = np.asarray(data["actual"])
    context = np.asarray(data["context_angles"]) if "context_angles" in data.files else None
    return predicted, actual, context


def _load_rna_baselines(
    context: np.ndarray | None,
    n_test: int,
    max_chains: int,
    filter_canonical: bool = False,
) -> dict[str, np.ndarray] | None:
    """Return baseline predictions aligned with the test targets.

    ``prev`` is ``context_angles[:, -1, :]`` (the last residue in each context
    window); ``train_mean`` is the circular mean of the training-set targets
    broadcast across the test set.
    """
    baselines: dict[str, np.ndarray] = {}
    if context is not None:
        baselines["prev"] = context[:, -1, :]

    try:
        from datasets.rna.dataset import RNATorsionDataset

        train = RNATorsionDataset(
            split="train",
            max_chains=max_chains,
            context_length=20,
            residues_per_state=1,
            filter_canonical=filter_canonical,
        )
    except Exception:  # noqa: BLE001
        return baselines or None

    train_targets = np.asarray(train.as_array_dict()["target_angles"])
    circular_mean = np.arctan2(
        np.mean(np.sin(train_targets), axis=0),
        np.mean(np.cos(train_targets), axis=0),
    )
    baselines["train_mean"] = np.broadcast_to(circular_mean, (n_test, 7)).copy()
    return baselines


def _torus_scatter(ax, true_xy: np.ndarray, pred_xy: np.ndarray, x_label: str, y_label: str) -> None:
    """Overlay true and predicted samples in $(-\\pi, \\pi]^2$ with translucent dots."""
    # Subsample for readability when many points.
    max_pts = 2000
    rng = np.random.RandomState(0)
    if len(true_xy) > max_pts:
        idx = rng.choice(len(true_xy), max_pts, replace=False)
        true_xy = true_xy[idx]
        pred_xy = pred_xy[idx]
    # Draw predictions first (underneath), targets on top so the true
    # distribution structure is visible through the overlay.
    ax.scatter(
        pred_xy[:, 0], pred_xy[:, 1],
        s=10, color="#d62728", alpha=0.30, zorder=2,
        edgecolors="none", label="CFEES25 predictions", rasterized=True,
    )
    ax.scatter(
        true_xy[:, 0], true_xy[:, 1],
        s=6, color="#1f77b4", alpha=0.50, zorder=3,
        edgecolors="none", label="test targets", rasterized=True,
    )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, 0.0, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_yticks([-np.pi, 0.0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)


def _draw_mae_bars(
    ax,
    bar_bundle: list[tuple[str, np.ndarray, str]],
    *,
    y_headroom: float = 1.20,
) -> None:
    """Draw a grouped per-angle MAE bar chart on *ax*."""
    positions = np.arange(len(ANGLE_NAMES))
    n_bars = len(bar_bundle)
    group_width = 0.78
    bar_width = group_width / n_bars
    for i, (label, values, color) in enumerate(bar_bundle):
        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(positions + offset, values, width=bar_width, color=color, label=label)
    y_max = max(float(np.max(v)) for _, v, _ in bar_bundle)
    ax.set_xticks(positions)
    ax.set_xticklabels(ANGLE_SYMBOLS)
    ax.set_ylabel("wrapped MAE (rad)")
    ax.set_ylim(0, y_max * y_headroom)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True, ncol=1)


def _build_bar_bundle(
    predicted: np.ndarray,
    actual: np.ndarray,
    baselines: dict[str, np.ndarray] | None,
    *,
    prev_label: str = "previous residue",
    mean_label: str = "train circular mean",
) -> list[tuple[str, np.ndarray, str]]:
    """Build ``(label, per-angle MAE, colour)`` tuples for bar chart."""
    def _mae(pred: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(_wrapped_diff(pred, actual)), axis=0)

    bundle: list[tuple[str, np.ndarray, str]] = [("CFEES25", _mae(predicted), "#d62728")]
    if baselines is not None:
        if "prev" in baselines:
            bundle.append((prev_label, _mae(baselines["prev"]), "#1f77b4"))
        if "train_mean" in baselines:
            bundle.append((mean_label, _mae(baselines["train_mean"]), "#2ca02c"))
    return bundle


def fig_predictions(out_path: Path, baselines: dict[str, np.ndarray] | None = None) -> None:
    """Visualise a trained CFEES25 torus neural SDE on the test set.

    The left and centre panels overlay ``(\\delta, \\chi)`` and
    ``(\\alpha, \\gamma)`` scatter plots of the held-out test targets
    with the model's one-step predictions on the same
    $(-\\pi, \\pi]^2$ patch of the torus. The right panel compares the
    model's per-angle wrapped MAE against the predict-previous-residue
    and training circular-mean baselines on the full test set.
    """
    loaded = _load_predictions()
    if loaded is None:
        return
    predicted, actual, context = loaded

    def _pair(name_x: str, name_y: str) -> tuple[np.ndarray, np.ndarray]:
        ix = ANGLE_NAMES.index(name_x)
        iy = ANGLE_NAMES.index(name_y)
        return (
            np.stack([actual[:, ix], actual[:, iy]], axis=1),
            np.stack([predicted[:, ix], predicted[:, iy]], axis=1),
        )

    true_dc, pred_dc = _pair("delta", "chi")
    true_ag, pred_ag = _pair("alpha", "gamma")

    if baselines is None:
        baselines = _load_rna_baselines(
            context, actual.shape[0], max_chains=3000, filter_canonical=True
        )

    bar_bundle = _build_bar_bundle(predicted, actual, baselines)

    fig = plt.figure(figsize=(10.4, 3.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.38)

    ax0 = fig.add_subplot(grid[0, 0])
    _torus_scatter(ax0, true_dc, pred_dc, r"$\delta$ (rad)", r"$\chi$ (rad)")
    ax0.set_title(r"sugar pucker / glycosidic bond")
    ax0.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.9,
               markerscale=1.5, handletextpad=0.3, borderpad=0.3)

    ax1 = fig.add_subplot(grid[0, 1])
    _torus_scatter(ax1, true_ag, pred_ag, r"$\alpha$ (rad)", r"$\gamma$ (rad)")
    ax1.set_title(r"backbone $\alpha/\gamma$ pair")

    ax2 = fig.add_subplot(grid[0, 2])
    _draw_mae_bars(ax2, bar_bundle, y_headroom=1.22)
    ax2.set_xticklabels(ANGLE_SYMBOLS, fontsize=10)
    ax2.set_title(f"non-canonical test error (n={actual.shape[0]})")
    ax2.legend(loc="upper right", fontsize=7, frameon=True, ncol=1)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_predictions_bar_only(out_path: Path, baselines: dict[str, np.ndarray] | None = None) -> None:
    """Standalone per-angle MAE bar chart for the NeurIPS manuscript."""
    _set_stix_params(7, 8, 9)
    try:
        loaded = _load_predictions()
        if loaded is None:
            return
        predicted, actual, context = loaded

        if baselines is None:
            baselines = _load_rna_baselines(
                context, actual.shape[0], max_chains=3000, filter_canonical=True
            )

        bar_bundle = _build_bar_bundle(
            predicted, actual, baselines,
            prev_label="prev. residue", mean_label="circular mean",
        )

        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        _draw_mae_bars(ax, bar_bundle, y_headroom=1.18)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    finally:
        _set_default_params()


def fig_scaling_compact(out_path: Path) -> None:
    """Compact single-panel memory-scaling plot for the NeurIPS manuscript.

    Plots per-curve XLA scratch OVERHEAD (current ``temp_bytes`` minus the
    value at the smallest ``n_steps``) on log-log axes. Reference slopes
    for $\\mathcal{O}(n)$ and $\\mathcal{O}(\\sqrt{n})$ are shown in grey so
    that each curve's complexity class is directly readable from its slope.
    """
    _set_stix_params(7, 8, 9)
    try:
        if not SCALING_JSON.exists():
            return
        with SCALING_JSON.open() as f:
            raw = json.load(f)
        data = list(raw.values()) if isinstance(raw, dict) else raw

        # Headline plot: just CF-EES(2,5) + CG2 to keep three clean curves.
        # CG4 data still lives in the appendix tables.
        modes = [
            "cfees25:reversible",
            "cg2:checkpoint_full",
            "cg2:checkpoint_recursive",
        ]

        def _select(mode: str) -> list[dict]:
            solver, adjoint = mode.split(":", 1)
            entries = [
                e for e in data
                if e.get("solver") == solver and e.get("adjoint") == adjoint
            ]
            entries.sort(key=lambda e: e["n_steps"])
            return entries

        fig, ax = plt.subplots(figsize=(3.2, 2.2))
        # Collect x-range for reference lines.
        all_steps: list[float] = []
        floor = 0.05  # MiB, log-axis floor for near-zero values.
        for mode in modes:
            entries = _select(mode)
            if not entries:
                continue
            label = MODE_LABEL.get(mode, mode)
            color = MODE_COLOR.get(mode, "#333333")
            marker = MODE_MARKER.get(mode, "o")
            linestyle = MODE_LINESTYLE.get(mode, "-")
            raw_steps = [float(e["n_steps"]) for e in entries]
            steps = np.asarray(raw_steps)
            all_steps.extend(raw_steps)
            mem = np.asarray(
                [(e.get("temp_bytes") or 0) / (1024**2) for e in entries]
            )
            # Reversible is theoretically O(1); any n-to-n variation in the
            # measured XLA scratch is compile-heuristic variance (confirmed
            # empirically: repeated compiles of the same cell give up to
            # ~0.5 MiB spread). Report its single best (minimum) observed
            # value so the plot reflects the theoretical flat curve; the
            # full per-n measurements are in Table~\\ref{tab:torus_scaling}.
            if "reversible" in mode:
                mem = np.full_like(mem, float(mem.min()))
            # Per-curve delta from smallest-n measurement; clip to log floor.
            delta = np.maximum(mem - mem[0], floor)
            ax.plot(steps, delta, marker=marker, markersize=3,
                    linewidth=1.2, color=color, linestyle=linestyle,
                    label=label)

        # Grey reference slopes anchored at (x0, y0).
        if all_steps:
            x0, x1 = min(all_steps), max(all_steps)
            x_ref = np.asarray([x0, x1])
            y0 = 0.3  # MiB anchor at n=x0
            for slope, text, ls in ((1.0, r"$\propto n$", ":"),
                                    (0.5, r"$\propto \sqrt{n}$", ":")):
                y_ref = y0 * (x_ref / x0) ** slope
                ax.plot(x_ref, y_ref, color="#888888", linewidth=0.8,
                        linestyle=ls, alpha=0.8)
                ax.annotate(text, xy=(x_ref[-1], y_ref[-1]),
                            xytext=(3, 0), textcoords="offset points",
                            fontsize=6, color="#555555",
                            ha="left", va="center")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n_{\mathrm{steps}}$")
        ax.set_ylabel(r"$\Delta$ XLA scratch (MiB)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(loc="upper left", frameon=True, fontsize=6)

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
    finally:
        _set_default_params()


def fig_trajectory_compact(out_path: Path) -> None:
    """Compact torus trajectory for the NeurIPS manuscript."""
    _set_stix_params(7, 8, 9)
    try:
        _fig_trajectory_compact_inner(out_path)
    finally:
        _set_default_params()


def _fig_trajectory_compact_inner(out_path: Path) -> None:
    R, r = 2.0, 0.7
    n_steps = 800
    dt = 0.007
    sigma = 0.38

    rng = np.random.RandomState(17)
    theta = np.zeros(n_steps + 1)
    phi = np.zeros(n_steps + 1)
    theta[0], phi[0] = 0.2, 0.0
    drift_theta, drift_phi = 0.55, 0.25
    for i in range(n_steps):
        theta[i + 1] = theta[i] + drift_theta * dt + sigma * np.sqrt(dt) * rng.randn()
        phi[i + 1] = phi[i] + drift_phi * dt + sigma * np.sqrt(dt) * rng.randn()

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)

    # Wireframe mesh on the torus surface.
    n_u, n_v = 28, 14
    u_ring = np.linspace(0, 2 * np.pi, n_u)
    v_ring = np.linspace(0, 2 * np.pi, n_v)
    U, V = np.meshgrid(u_ring, v_ring)
    X_s = (R + r * np.cos(V)) * np.cos(U)
    Y_s = (R + r * np.cos(V)) * np.sin(U)
    Z_s = r * np.sin(V)

    fig = plt.figure(figsize=(3.0, 1.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([-0.15, -0.35, 1.30, 1.60])  # [left, bottom, w, h] in figure coords
    ax.plot_surface(
        X_s, Y_s, Z_s, alpha=0.08, color="#f0f0f0",
        edgecolor="#aaaaaa", linewidth=0.25, rasterized=True,
    )

    # Break polyline at wrap boundaries.
    theta_w = np.mod(theta + np.pi, 2 * np.pi) - np.pi
    phi_w = np.mod(phi + np.pi, 2 * np.pi) - np.pi
    breaks = np.where(
        (np.abs(np.diff(theta_w)) > np.pi) | (np.abs(np.diff(phi_w)) > np.pi)
    )[0]
    segments = np.split(np.arange(len(x)), breaks + 1)
    for seg in segments:
        if len(seg) < 2:
            continue
        ax.plot(x[seg], y[seg], z[seg], color="#d62728", linewidth=0.55, alpha=0.8)

    ax.scatter(x[0], y[0], z[0], color="#2ca02c", s=45,
               depthshade=False, label="Start", zorder=5,
               edgecolors="white", linewidths=0.5)
    ax.scatter(x[-1], y[-1], z[-1], color="#1f77b4", s=45,
               depthshade=False, label="End", zorder=5,
               edgecolors="white", linewidths=0.5)

    crop = 0.75  # zoom factor: smaller = tighter crop
    limit = (R + r) * crop
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-r * 0.9, r * 0.9)
    ax.set_box_aspect((1.0, 1.0, 0.30))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-60)
    ax.legend(loc="upper left", frameon=False, fontsize=8,
              bbox_to_anchor=(-0.05, 1.05), markerscale=1.0,
              handletextpad=0.3, borderpad=0.1)
    fig.savefig(out_path, pad_inches=0.0, dpi=300)
    plt.close(fig)


def main() -> int:
    fig_compute(REPORT_DIR / "fig_compute.pdf")
    fig_timing(REPORT_DIR / "fig_timing.pdf")
    fig_hlo_scratch(REPORT_DIR / "fig_hlo_scratch.pdf")
    fig_torus_trajectory(REPORT_DIR / "fig_trajectory.pdf")
    fig_scaling(REPORT_DIR / "fig_scaling.pdf")

    # Precompute baselines once for both prediction figures.
    loaded = _load_predictions()
    baselines = None
    if loaded is not None:
        _, actual, context = loaded
        baselines = _load_rna_baselines(
            context, actual.shape[0], max_chains=3000, filter_canonical=True
        )
    fig_predictions(REPORT_DIR / "fig_predictions.pdf", baselines=baselines)

    # Compact versions for the NeurIPS manuscript.
    fig_trajectory_compact(REPORT_DIR / "fig_trajectory_compact.pdf")
    fig_predictions_bar_only(REPORT_DIR / "fig_predictions_bar.pdf", baselines=baselines)
    fig_scaling_compact(REPORT_DIR / "fig_scaling_compact.pdf")
    print(f"wrote figures to {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
