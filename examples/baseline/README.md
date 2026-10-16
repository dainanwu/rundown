# ETTh1 + LSTM example

A **runnable, end-to-end** demonstration of `rundown` on a standard
time-series benchmark (ETTh1) with a standard neural baseline (2-layer LSTM).

## What it shows

1. How to write a **`BatchProvider`** that conforms to the contract rundown
   expects (see `data.py`).
2. How to **import `rundown` as a library** and drive training without the
   CLI (`train.py`).
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
cd E:\rundown
python -m examples.baseline.train
```

Expected output (last 4 lines):

```
[result] test MSE   = 5.xxxx
[result] best val   = 5.xxxx
[result] epochs     = 5
[result] run dir    = artifacts\lstm_etth1__ot24h__seed0
```

**`test MSE < 8.0`** means the LSTM learned the OT signal — it's about half
the naive persistence baseline (predict last value, MSE ~15). Anything
above 8.0 means the example regressed; file an issue.

## Files

| File | Role |
|---|---|
| `data/ETTh1.csv` | Bundled benchmark data (17k hourly rows, CC0) |
| `data.py` | `load_etth1()` + `SlidingWindowProvider` (the BatchProvider contract) |
| `model.py` | `SequenceLSTM` — 2-layer LSTM, contract-only on `forward(x, warm_start)` |
| `train.py` | end-to-end script: data → providers → model → manifest → engine → read-back |
| `test_baseline.py` | pytest that runs `train.py` as a subprocess and asserts loss decreases |

## Where to read next

If you're wiring `rundown` into a real project, the order to read is:

1. `train.py` — the **only** file that orchestrates everything end-to-end.
2. `data.py` — the **BatchProvider** contract. Your real loader conforms to it.
3. `rundown/artifact.py` — the artifact format. The manifest fields you'll
   populate and the curves / predictions / events you can read back.

## Adapting to your project

The diff to convert this example into your real run is roughly:

- **`data.py`** → replace `load_etth1` with your parquet/SQL/CSV reader;
  replace `SlidingWindowProvider` with your own iterable yielding the same
  `{"x", "y", "warm_start"}` dict.
- **`model.py`** → replace `SequenceLSTM` with your model. Keep the
  `forward(x, warm_start) -> [B, T']` contract.
- **`train.py`** → change `RUN_ID`, `WARM_START`, `ROLLOUT`, `EPOCHS`,
  `STEPS_PER_EPOCH`, `LR`, `HIDDEN`, `NUM_LAYERS`. Everything else stays.

That's the entire surface area of `rundown` for a real project.