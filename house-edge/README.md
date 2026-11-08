# House Edge — TAMU Datathon 2026

Next-spin payout forecasting for simulated slot machines. For each query, predict the probability of a 0×, 1×, 2×, 5× or 20× payout. Submissions are scored by squared error relative to a pooled baseline, averaged over four conditions: `stable`, `short_history`, `drift`, `shared_change`.

**Result:** 56.68 / 60 technical points on validation (pooled baseline ≈ 30), validation error ratio R = 0.0585.

## Quick start

```sh
uv venv --python 3.13 .venv && uv pip install --python .venv numpy==2.5.4
.venv/bin/python entry/predict.py --data kit/public --out submission.csv            # test split
.venv/bin/python entry/predict.py --data kit/public --split validation --out val.csv
(cd kit && python3 check_submission.py ../val.csv --split validation)              # official scorer
```

The entry uses only numpy and is deterministic. The test split takes about 90 s on one CPU core. `--workers N` runs episodes in parallel and gives identical output for any N.

## Layout

| Path | What it is |
|---|---|
| `entry/` | The submitted model: `hemodel.py` (model), `predict.py` (CLI), `calibrate.py` (writes `config.json` priors), `reproduce.json`, `requirements.txt`, and a `README.md` describing the method |
| `kit/` | Organizer participant kit: data in `public/`, the checker, the starter, and the rules |
| `dl/` | The original `kit.zip` download |
| `work/` | Analysis and experiment scripts used to reverse-engineer the simulator (`common.py` mirrors the official metric) |
| `runs/` | Intermediate prediction CSVs |
| `submissions/` | Packaged entries (`main_v3` is current; `main_v1` and `main_v2` are superseded) |

## Simulator structure found

1. **Win probability** depends on whether a machine's own win/loss results at two earlier ticks match (XNOR). Only the lag pairs (1,3), (2,5) and (4,7) occur. Each machine type has a high and a low win level (steady ≈ 0.955/0.148, balanced ≈ 0.839/0.039, jackpot ≈ 0.578/0.011). Which outcome gets the high level is random per machine.
2. **drift:** each machine reverses that direction once, at its own tick. **shared_change:** every machine in the episode reverses at the same tick.
3. **Payout size given a win:** log(p5/p2) and log(p20/p5) are fixed per type. log(p2/p1) = μ + β·[payout at lag l ≥ 2], with l ∈ {1, 2, 4} and β = ±1.2.

## Method

The model is Bayesian model averaging over:
- lag pairs, using a 2-cell (XNOR) model and a 4-cell variant
- change points, on a 4-tick grid, averaging a shared change tick against independent ones
- a type-level prior on win rates

Each candidate model is scored by its Beta-Binomial marginal likelihood. Payout size uses a per-machine grid posterior over (μ, β), averaged over the size lag. EM and empirical Bayes on the public data set the round-level priors. See `entry/README.md` for details.

| Condition | Error vs baseline |
|---|---|
| stable | 0.040 |
| drift | 0.054 |
| shared_change | 0.051 |
| short_history | 0.088 |

The error ratio fell from 1.0 (baseline) through 0.148 (lag-pair model), 0.096 (pair averaging) and 0.0795 (change-point mixture) to 0.0585 (type priors, lag-dependent size, bimodal β).

## Data note

`kit/` and `dl/` contain organizer data. Keep this repo private unless the datathon rules allow sharing it.
