# RNA torus neural SDE — hyperparameter sweep log

Running log of the fast 8-epoch sweeps we use to pin down one hyperparameter
at a time. The "current champion" config at the top is always the config we
roll forward into the next sweep.

## Current champion config

| parameter | value |
|---|---|
| `max_chains` | 1000 |
| `batch_size` | 64 |
| `hidden_dim` | 256 |
| `ctx_dim` | 128 |
| `n_steps` | 20 |
| `dt` | 0.125 |
| `solver` | CFEES25 |
| `diffusion_scale` | 0.1 |
| `context_length` | 20 |
| `residues_per_state` | 1 |
| `future_bases_window` | 0 |
| `energy_score_samples` | 4 |
| **`learning_rate`** | **3e-4** |
| `warmup_steps` | 1000 |
| `end_lr_frac` | 0.05 |
| `weight_decay` | 1e-4 |
| **`activation`** | **relu** |
| `drift_depth` | 3 (TBD) |
| `diffusion_depth` | 2 (TBD) |

8-epoch best val_mae at this config: **0.4692** at epoch 6.

Committed reference ("full 20-epoch + test-time MC") overall test MAE of the
energy-score model at LR=1e-4 was 0.412 (commit `796563f`).

## Sweep history

### LR peak sweep (end_lr_frac = 0.05, 8 epochs)

| LR | best val_mae | at epoch | note |
|---|---|---|---|
| **3e-4** | **0.4692** | 6 | **winner** |
| 1e-4 | 0.4806 | 5 | previous default |
| 5e-5 | 0.4875 | 6 | too slow |

### LR peak + end_lr_frac sweep (8 epochs)

| peak LR | end frac | best val_mae | note |
|---|---|---|---|
| 3e-4 | **0.05** | **0.4692** | original, still champion |
| 3e-4 | 0.01 | 0.4856 | more aggressive decay hurts |
| 3e-4 | 0.001 | ~0.4846 | tightest decay, still worse |
| 1e-3 | 0.01 | DIVERGED | val=1.57 at epoch 2, never recovers |

**Takeaways**:
- 3e-4 is the sweet spot; 1e-3 diverges catastrophically, 5e-5 is slower.
- The default cosine decay end at 5% of peak is already about right;
  forcing it lower doesn't refine further at 8 epochs.
- "3e-4 diverged at ep22" from an earlier long run is a late-training
  phenomenon, not an immediate instability — fine for short fast iteration.

### Activation sweep (LR=3e-4, 8 epochs)

| activation | best val_mae | ep8 train_loss | note |
|---|---|---|---|
| **relu** | **0.4525** | **2.327** | **winner** — 0.017 below silu |
| silu | 0.4692 | 2.562 | previous default |
| tanh | 0.4784 | 2.473 | bounded but slightly worse |
| gelu | 0.4889 | 2.492 | surprisingly underperforms |
| mish | 0.4929 | 2.523 | worst of the five |

**Takeaway**: relu wins decisively, with a 3.6% lower val MAE and a meaningfully
lower train loss. Not obvious *why* relu beats silu here — possibly the piecewise
linearity helps the drift field stay in a useful regime when the inputs (sin/cos
angles + ctx) hit small magnitudes. Worth noting relu on a diffusion MLP should
in theory produce less smooth diffusion coefficients, but softplus on the output
is doing the smoothing so it doesn't matter.

### auto: drift_depth
Varying `RNA_DRIFT_DEPTH`; every other knob held at the current champion.

| value | best val_mae | note |
|---|---|---|
| 3 | 0.4531 | best=0.4531 (953s) |
| 2 | 0.4543 | best=0.4543 (859s) |
| 4 **winner** | 0.4493 | best=0.4493 (1046s) |
| 5 | 0.4597 | best=0.4597 (1146s) |

**Champion updated**: `RNA_DRIFT_DEPTH` 3 → 4

### auto: diffusion_depth
Varying `RNA_DIFFUSION_DEPTH`; every other knob held at the current champion.

| value | best val_mae | note |
|---|---|---|
| 2 | 0.4742 | best=0.4742 (1053s) |
| 1 | 0.4531 | best=0.4531 (953s) |
| 3 **winner** | 0.4479 | best=0.4479 (1144s) |

**Champion updated**: `RNA_DIFFUSION_DEPTH` 2 → 3

### auto: hidden_dim
Varying `RNA_HIDDEN_DIM`; every other knob held at the current champion.

| value | best val_mae | note |
|---|---|---|
| 256 | 0.4558 | best=0.4558 (1152s) |
| 128 | 0.4546 | best=0.4546 (751s) |
| 192 | 0.4573 | best=0.4573 (914s) |
| 384 **winner** | 0.4461 | best=0.4461 (1744s) |

**Champion updated**: `RNA_HIDDEN_DIM` 256 → 384

### auto: ctx_dim
Varying `RNA_CTX_DIM`; every other knob held at the current champion.

| value | best val_mae | note |
|---|---|---|
| 128 | 0.4604 | best=0.4604 (1735s) |
| 64 **winner** | 0.4511 | best=0.4511 (1669s) |
| 192 | 0.4648 | best=0.4648 (1833s) |

**Champion updated**: `RNA_CTX_DIM` 128 → 64

### auto: diffusion_scale
Varying `RNA_DIFFUSION_SCALE`; every other knob held at the current champion.

| value | best val_mae | note |
|---|---|---|
| 0.1 | 0.4544 | best=0.4544 (1730s) |
| 0.05 **winner** | 0.4445 | best=0.4445 (1673s) |
| 0.2 | 0.4582 | best=0.4582 (1679s) |
| 0.3 | 0.4647 | best=0.4647 (1674s) |

**Champion updated**: `RNA_DIFFUSION_SCALE` 0.1 → 0.05

### auto: warmup_steps
Varying `RNA_WARMUP`; every other knob held at the current champion.

| value | best val_mae | note |
|---|---|---|
| 1000 **winner** | 0.4461 | best=0.4461 (1667s) |
| 500 | 0.4587 | best=0.4587 (1669s) |
| 2000 | 0.4509 | best=0.4509 (1674s) |
| 3000 | 0.4506 | best=0.4506 (1673s) |

**Champion unchanged** (`RNA_WARMUP` stays at 1000).

### auto: n_steps
Varying `RNA_N_STEPS`; every other knob held at the current champion.

| value | best val_mae | note |
|---|---|---|
| 20 **winner** | 0.4638 | best=0.4638 (1666s) |
| 10 | 0.4699 | best=0.4699 (887s) |
| 30 | 0.4835 | best=0.4835 (2411s) |

**Champion unchanged** (`RNA_N_STEPS` stays at 20).

### auto: energy_samples (partial — system rebooted during sweep)

Baseline (N=4) completed at 0.4580. N=8 was at epoch 3/8 tracking at
0.4990 (behind baseline) when the system rebooted. N=4 retained as
champion.

**Champion unchanged** (`RNA_ENERGY_SAMPLES` stays at 4).

### Final champion config (from sweep)

| parameter | value |
|---|---|
| `learning_rate` | 3e-4 |
| `activation` | relu |
| `drift_depth` | 4 |
| `diffusion_depth` | 3 |
| `hidden_dim` | 384 |
| `ctx_dim` | 64 |
| `diffusion_scale` | 0.05 |
| `warmup_steps` | 1000 |
| `n_steps` | 20 |
| `dt` | 0.125 |
| `energy_score_samples` | 4 |
| `batch_size` | 64 |
| `end_lr_frac` | 0.05 |

8-epoch best val_mae at this config: ~0.44–0.45 (noisy across runs).
