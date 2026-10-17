# ETTh1 + LSTM example

A **runnable, end-to-end** demonstration of `rundown` on a standard
time-series benchmark (ETTh1) with a standard neural baseline (2-layer LSTM).

## What it shows

1. How to write a **`BatchProvider`** that conforms to the contract rundown
   expects (see `data.py`).
2. How to **drive training via the CLI** — `train.py` is a thin wrapper that
   subprocesses `rundown.cli train` (Phase D). For real projects you don't
   even need this wrapper; just call the CLI directly.
3. That the **artifact** is self-describing: `manifest.yaml`, weights,
   curves, events, and predictions all land in one directory.

## Dataset

**ETTh1** (Electricity Transformer Temperature - hourly), the standard
benchmark for Informer / Autoformer / PatchTST / LSTNet.

- 17420 hourly readings, 2016-07-01 to 2018-06-26
- 6 power-load features (HUFL, HULL, MUFL, MULL, LUFL, LULL) → target OT
  (Oil Temperature, degC, range ~[-4, 46])
- Source: [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) (CC0)
- Bundled at `data/ETTh1.csv` for offline use

## Model

**2-layer LSTM** (`in_dim=6, hidden=64, num_layers=2`). Most ETT papers
include an LSTM baseline alongside the proposed Transformer; this is the
canonical shape. The head is a `Linear(hidden, 1)` applied per time step.

Task: given 24 hours of lookback, predict 24 hours ahead.

## Run it

```bash
cd /path/to/rundown                       # (Windows: cd E:\rundown)
python -m examples.baseline.train          # thin wrapper, equivalent to:
python -m rundown.cli train --dataset etth1 --model lstm
```

Expected output (final `[done]` line plus summary):

```
[done] test_mse(test_best)=0.7xxx  ...
```

Test MSE is in **y-standardized space** (the provider normalizes the OT
column). A reasonable LSTM with `epochs_max=15` (the rundown default) lands
in **[0.5, 1.2]** — far below the naive "predict OT mean" baseline of 1.0
when training kicks in. Anything consistently above 1.2 means the example
regressed; file an issue.

## Files

| File | Role |
|---|---|
| `data/ETTh1.csv` | Bundled benchmark data (17k hourly rows, CC0) |
| `data.py` | `load_etth1()` + `SlidingWindowProvider` (the BatchProvider contract) |
| `model.py` | `SequenceLSTM` — 2-layer LSTM, contract-only on `forward(x, warm_start)` |
| `train.py` | Thin subprocess wrapper around `rundown.cli train` (for test isolation only) |
| `test_baseline.py` | pytest that runs `train.py` as a subprocess and asserts loss decreases |

## Where to read next

If you're wiring `rundown` into a real project, the order to read is:

1. **`rundown/cli.py`** — the 8 subcommands. Real usage is the CLI, not a
   Python script. Override anything with `--epochs-max / --steps-per-epoch
   / --batch-size / --lr / --kwargs KEY=VAL`.
2. **`configs/{datasets,models,experiments}.yaml`** — declare your
   datasets / models / sweeps here. Run `rundown prepare / train /
   experiment run <name>` against them.
3. **`rundown/artifact.py`** — the artifact format. The manifest fields
   you'll read back and the curves / predictions / events on disk.
4. **`data.py`** in this dir — the **BatchProvider** contract. Your real
   loader conforms to it.

## Adapting to your project

The diff to convert this example into your real run is roughly:

- **`data.py`** → replace `load_etth1` with your parquet/SQL/CSV reader;
  replace `SlidingWindowProvider` with your own iterable yielding the same
  `{"x", "y", "warm_start"}` dict.
- **`model.py`** → replace `SequenceLSTM` with your model. Keep the
  `forward(x, warm_start) -> [B, T']` contract.
- **`configs/datasets.yaml`** → register your dataset (alias, source path,
  splits, target column).
- **`configs/models.yaml`** → register your model (factory target with
  `module.attr` form, default kwargs, optional `variants:` presets).
- **Tune `rundown train`** args (`--epochs-max / --lr / --variant small`)
  or write an **`configs/experiments.yaml`** entry to sweep.

That's the entire surface area of `rundown` for a real project.
