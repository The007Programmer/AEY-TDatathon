# House Edge entry: Bayesian lag-pair change-point forecaster

- Team ID: AEY Inc.
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13) + `numpy==2.5.4` (requirements.txt). No other dependencies.
- Hardware: CPU only, single core, < 1 GiB RAM. Runtime ~100 s CPU for the test split.
- Deterministic: no random numbers are used anywhere. Round-level priors are saved in `config.json` (produced by
  `python calibrate.py --data <public> --out config.json` from the public train/validation/test data); the per-machine
  model is fit from the history in `/data` at prediction time.

## Recreate the forecast

Run from the ZIP root:

```sh
python predict.py --data /data --out /output/submission.csv --workers 1
```

(`--data` is the round's `public/` folder; the script reads `public/test/queries.csv` and `public/test/history.csv`.
Add `--split validation` to produce validation predictions. `--workers N` parallelises over episodes; output is identical for any N.)

## Method

1. **Win probability.** Each machine's win chance is set by an XNOR-style interaction of its own win/loss at two lags
   (mostly (1,3), (2,5), (4,7)). We average over all 28 lag pairs, a 2-cell (XNOR) and a 4-cell variant,
   with Beta-Binomial marginal likelihoods and a prior favouring the common pairs.
2. **Change points.** The interaction can reverse. We put a change-point prior on a 4-tick grid; post-change cell rates get a
   Beta prior centred on the flipped pre-change rate. Two scenarios are averaged by evidence: one change tick shared across
   the episode's machines, or independent change ticks.
   Cell rates get a type-level prior (high/low win levels per machine type, mixed over both directions).
3. **Payout size.** Given a win, log(p5/p2) and log(p20/p5) are fixed per machine type; log(p2/p1) = mu + beta * [lag-l payout >= 2],
   with a per-machine Bayesian grid posterior over (mu, beta) (beta prior: +/-1.2), averaged over lag l (EM-estimated prior: 1, 2, 4).

Constants (type ratios, priors) were chosen on the public train/validation data (and empirical-Bayes on public histories) only.
Validation: 56.68 / 60 technical points (`check_submission.py --split validation`).
