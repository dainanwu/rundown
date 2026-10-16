"""End-to-end test for the ETTh1 + LSTM example.

Runs ``train.py`` as a subprocess in an isolated tmp dir, then verifies:
  1. exit code 0
  2. test MSE is small (the LSTM actually learned the OT signal)
  3. all expected artifact files exist
  4. epoch_completed events in events.jsonl show train_loss decreasing
  5. manifest.yaml is parseable and records the frozen config

OT in ETTh1 spans roughly [-4, 46] degC with hourly std ~8.5. The data is
y-standardized inside the provider, so MSE here is in N(0,1) space:
  - Naive "predict OT mean" baseline MSE = 1.0
  - ETT papers report LSTM OT 24h RMSE ~ 6-9 degC → standardized MSE ~ 0.5-1.1
We assert < 1.2 (still captures regressions to "model didn't learn",
e.g. un-normalized MSE ~ 5+ in standardized space). The manifest records
the y-scaler so the artifact can be denormalized back to degC.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# CLI 默认组成: model_alias__dataset_alias__seed<N>
RUN_ID = "lstm__etth1__seed0"
ENTRY = [sys.executable, "-m", "examples.baseline.train"]


def _run_example(tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["RUNDOWN_OUT_DIR"] = str(tmp_path / "artifacts")
    return subprocess.run(
        ENTRY,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )


def test_example_runs(tmp_path: Path):
    """Full end-to-end: train.py runs the CLI, produces artifact, prints test MSE.

    Phase D 起, examples/baseline/train.py 是 rundown.cli train 的 wrapper;
    端到端覆盖了 CLI 解析 -> registry -> provider -> engine -> artifact.
    """
    result = _run_example(tmp_path)
    assert result.returncode == 0, (
        f"train.py failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # CLI 输出格式: "[done] ... test_mse(test_best)=X.XXXX ..."
    m = re.search(r"test_mse\([\w_]+\)=([\d.]+)", result.stdout)
    assert m, f"test_mse line not found in stdout:\n{result.stdout}"
    test_mse = float(m.group(1))
    # y is standardized in provider; constant-mean baseline MSE = 1.0.
    # LSTM with 15 epochs x 50 steps should land in [0.5, 1.2].
    assert test_mse < 1.2, f"test MSE {test_mse:.4f} too high; LSTM didn't learn"


def test_artifact_files(tmp_path: Path):
    """The run dir contains the expected files (now including test_best)."""
    result = _run_example(tmp_path)
    assert result.returncode == 0, f"failed: {result.stderr}"

    run_dir = tmp_path / "artifacts" / RUN_ID
    expected = [
        "manifest.yaml",
        "weights.safetensors",
        "weights_best.safetensors",
        "curves.safetensors",
        "events.jsonl",
        "predictions/test.safetensors",
        "predictions/test_best.safetensors",
    ]
    for rel in expected:
        path = run_dir / rel
        assert path.exists(), f"missing artifact: {path}"
        assert path.stat().st_size > 0, f"empty artifact: {path}"


def test_train_loss_decreases(tmp_path: Path):
    """Parse events.jsonl; second half of train_loss < first half."""
    result = _run_example(tmp_path)
    assert result.returncode == 0, f"failed: {result.stderr}"

    events_path = tmp_path / "artifacts" / RUN_ID / "events.jsonl"
    events = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_losses = [e["train_loss"] for e in events if e.get("kind") == "epoch_completed"]
    assert len(train_losses) >= 4, f"too few epoch events: {len(train_losses)}"

    mid = len(train_losses) // 2
    first_avg = sum(train_losses[:mid]) / mid
    second_avg = sum(train_losses[mid:]) / (len(train_losses) - mid)
    assert second_avg < first_avg, (
        f"loss did not decrease: first half avg={first_avg:.4f}, "
        f"second half avg={second_avg:.4f}"
    )


def test_manifest_records_frozen_config(tmp_path: Path):
    """manifest.yaml is parseable and contains the fields we wrote."""
    import yaml
    result = _run_example(tmp_path)
    assert result.returncode == 0, f"failed: {result.stderr}"

    manifest_path = tmp_path / "artifacts" / RUN_ID / "manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert data["run_id"] == RUN_ID
    assert data["seed"] == 0
    assert data["dataset"]["alias"] == "etth1"
    assert data["dataset"]["target_col"] == "OT"
    assert data["dataset"]["splits"] == {"train": 0.7, "val": 0.2, "test": 0.1}
    assert data["model"]["alias"] == "lstm"
    assert data["model"]["variant"] is None  # base variant
    assert data["model"]["factory"]["target"] == "examples.baseline.model.SequenceLSTM"
    assert data["config"]["epochs_max"] == 15
    # metrics was populated by the engine at end of fit
    assert "best_val_mse" in data["metrics"]
    # 双产物: primary predictions 由 best weights 算, 与 manifest.metrics 对齐
    assert data["config"]["predictions_primary"] == "test_best"