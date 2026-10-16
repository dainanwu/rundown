"""ETTh1 reader + sliding-window batch provider.

This file demonstrates the **BatchProvider contract** rundown expects from your
real dataloader. Replace ``load_etth1`` and ``SlidingWindowProvider`` with your
own code (parquet reader, SQL query, etc.) and the rest of rundown stays
the same.

Dataset
-------
ETTh1 (Electricity Transformer Temperature - hourly). 17420 hourly readings,
6 power-load features (HUFL, HULL, MUFL, MULL, LUFL, LULL) → target OT
(Oil Temperature, degC). Standard benchmark for Informer / Autoformer /
LSTM-based baselines.

Source: zhouhaoyi/ETDataset on GitHub (CC0). The complete CSV is bundled
at ``data/ETTh1.csv`` so the example is self-contained and offline.

Contract recap
--------------
A BatchProvider is any callable that takes a split name and returns a dict:

    provider(split: str) -> {
        "x":          torch.Tensor [B, T, F],
        "y":          torch.Tensor [B, T],    # engine will slice [:, warm:warm+T']
        "warm_start": int,
    }

The engine slices target ``y[:, warm:warm + yhat.shape[1]]`` to match the
model's rollout length, so the provider only needs to return ``y`` covering at
least the warm_start + rollout range.

``build_etth1_providers`` is the registry-facing builder. It loads the CSV,
splits chronologically per the registry's ``splits`` dict, computes X/y
standardization stats from train only, and returns three BatchProviders
with matching scalers. This is what ``rundown.cli train --dataset etth1``
ends up calling.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


# ETT column conventions (multivariate -> univariate OT).
FEATURE_COLS: List[str] = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET_COL: str = "OT"


def load_etth1(csv_path: Path | str) -> pd.DataFrame:
    """Load ETTh1 from the bundled CSV. Parse ``date`` column to datetime."""
    df = pd.read_csv(Path(csv_path))
    df["date"] = pd.to_datetime(df["date"])
    return df


def standard_scaler_stats(df: pd.DataFrame, cols: List[str]):
    """Compute per-column mean / std on the given DataFrame for X-standardization.

    The provider stores these and applies (x - mean) / std at batch time.
    Returning them as numpy arrays (not torch tensors) keeps the provider
    allocation-free of autograd.
    """
    arr = df[cols].to_numpy(dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    # Avoid division by zero for constant columns (rare but defensive).
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def y_scaler_stats(df: pd.DataFrame, col: str):
    """Scalar mean / std of the target column for y-standardization.

    Why we standardize y too: OT (degC, range -4..46) makes MSE live at the
    ~100 scale; Adam with default lr collapses LSTM training on this
    scale. Standardizing to N(0,1) keeps the loss surface sane. The same
    scaler is recorded in the manifest so the artifact remains the ground
    truth.
    """
    arr = df[col].to_numpy(dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


def build_etth1_providers(
    source_path: str,
    splits: Dict[str, float],
    target_col: Optional[str] = None,
    *,
    feature_cols: Optional[List[str]] = None,
    warm_start: int = 24,
    rollout: int = 24,
    batch_size: int = 32,
    **overrides,
) -> Dict[str, "SlidingWindowProvider"]:
    """Registry-facing builder: 拿 source_path -> 3 个 BatchProvider (train/val/test).

    签名跟 DatasetEntry.build_providers 约定一致: builder(source_path, splits,
    target_col, **overrides). feature_cols / warm_start / rollout / batch_size
    通过 overrides 传进来 (典型来源: configs/datasets.yaml 的 feature_cols +
    后续可能加的 provider_kwargs 字段).

    时间顺序切分; X/y 标准化统计量只在 train 上算, 然后应用到 val/test.
    """
    feat_cols = feature_cols if feature_cols else FEATURE_COLS
    tgt_col = target_col if target_col else TARGET_COL
    csv_path = Path(source_path)
    df = load_etth1(csv_path)

    n = len(df)
    train_end = int(n * splits["train"])
    val_end = train_end + int(n * splits["val"])
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    x_mean, x_std = standard_scaler_stats(train_df, feat_cols)
    y_mean, y_std = y_scaler_stats(train_df, tgt_col)

    common = dict(
        feature_cols=feat_cols, target_col=tgt_col,
        warm_start=warm_start, rollout=rollout, batch_size=batch_size,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
    )
    return {
        "train": SlidingWindowProvider(train_df, **common, seed=0),
        "val":   SlidingWindowProvider(val_df,   **common, seed=1),
        "test":  SlidingWindowProvider(test_df,  **common, seed=2),
    }


class SlidingWindowProvider:
    """Yield random mini-batches of sliding windows from one DataFrame split.

    Per the BatchProvider contract, ``__call__`` returns a dict with
    ``x: [B, T, F]``, ``y: [B, T]``, ``warm_start: int``.

    Optional X-standardization: pass ``x_mean`` and ``x_std`` (computed on
    the train split, see ``standard_scaler_stats``) so the LSTM sees inputs
    in roughly N(0, 1) scale — essential for stable RNN training on
    heterogeneous-magnitude features.

    The provider is stateful with respect to its own RNG (seed) but stateless
    with respect to split — the split name is accepted for API symmetry and
    ignored, since the constructor already received a specific DataFrame for
    this split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: List[str],
        target_col: str,
        warm_start: int,
        rollout: int,
        batch_size: int,
        seed: int = 0,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
        y_mean: Optional[float] = None,
        y_std: Optional[float] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.warm_start = warm_start
        self.rollout = rollout
        self.batch_size = batch_size
        self.window_len = warm_start + rollout + 1   # +1 for the last target
        self.rng = np.random.default_rng(seed)
        # Pre-extract as numpy arrays (faster than DataFrame indexing per call).
        X_raw = df[feature_cols].to_numpy(dtype=np.float32)
        if x_mean is not None and x_std is not None:
            # Broadcasting (1, F) over (N, F) keeps shape (N, F).
            self._X = (X_raw - x_mean) / x_std
        else:
            self._X = X_raw
        y_raw = df[target_col].to_numpy(dtype=np.float32)
        if y_mean is not None and y_std is not None:
            self._y = (y_raw - y_mean) / y_std
        else:
            self._y = y_raw
        n = len(df)
        # Valid start indices: 0 .. n - window_len (inclusive)
        self._starts = np.arange(0, max(1, n - self.window_len + 1))

    def __call__(self, split: str = "train") -> Dict[str, torch.Tensor]:  # noqa: ARG002
        idx = self.rng.choice(self._starts, size=self.batch_size, replace=True)
        x = np.stack([self._X[s : s + self.window_len] for s in idx])  # [B, T, F]
        y = np.stack([self._y[s : s + self.window_len] for s in idx])  # [B, T]
        return {
            "x":          torch.from_numpy(x),
            "y":          torch.from_numpy(y),
            "warm_start": self.warm_start,
        }


# Convenience type alias for documentation.
BatchProvider = Callable[[str], Dict[str, torch.Tensor]]