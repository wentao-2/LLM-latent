# Depth Portfolio Allocation

Source code for budget-constrained depth portfolio allocation (DPA), imported unchanged from `dpa_src.zip`.

- `src/dpa_uniform.py`: deterministic Uniform allocation with depth costs, per-depth candidate limits, and a hard budget cap.
- `src/dpa_ppo.py`: PPO training, validation-based configuration selection, and independent test evaluation over cached candidates.

## Uniform baseline

Requires Python 3.10 or later. Print allocations for the supplied budget caps:

```bash
python src/dpa_uniform.py
```

## PPO experiments

Install NumPy and PyTorch in your experiment environment. Prepare candidate caches in `./data`, or set `DPA_DATA_DIR` to their directory. The script reads `<dataset>_train.npz`, `<dataset>_validation.npz`, and `<dataset>_test.npz`, with `qids`, `texts`, `labels`, and `scores` arrays. Hidden-state features can be stored in `<dataset>_<split>_hT.npy` or the NPZ `hT` array.

For the hidden-state tuning configuration:

```bash
python src/dpa_ppo.py --dataset gsm8k --tune --only-hT --gpu 0
```

Dataset choices are `gsm8k`, `svamp`, `mbpp`, and `multiarith`. Checkpoints and JSON results are written to `./runs/<dataset>`; set `DPA_RUN_DIR` to change the output directory. Candidate caches and checkpoints are not included in this source archive.

The imported source passed Python syntax checks. PPO training and evaluation have not been run as part of this upload.
