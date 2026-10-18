"""Tests for rundown.inspect_run — read artifacts → README-style markdown.

Assertions are intentionally narrow: the exact format is allowed to evolve.
What MUST hold:
  - load_run() returns a RunIntel with non-empty manifest, decoded curves, sample predictions
  - render_run_markdown() includes the run id, all key metrics, and a code block
  - Missing run -> FileNotFoundError
  - iter_recent_runs() --all listing respects filters
  - rundown inspect CLI exits 0 with non-empty stdout markdown
  - C4: generic metrics / curves rendering, ``Trainer`` line, multi-model
    curves all work without breaking legacy output.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
import yaml

# bootstrap: tests/ has no conftest, rely on root cwd (pytest already sets it)
from rundown.inspect_run import (
    EpochPoint,
    RunIntel,
    iter_recent_runs,
    load_run,
    render_run_markdown,
)


# ============================================================================
# Real-artifact smoke (against the project's own artifacts/)
# ============================================================================

# Path to the project's live run (created during earlier train smoke tests).
# If absent (fresh checkout), tests still pass on the tmp_path mini-run below.
_LIVE_RUN_DIR = Path("artifacts") / "lstm__varsmall__etth1__seed0"


@pytest.mark.skipif(not _LIVE_RUN_DIR.exists(), reason="no live run on disk")
class TestLiveArtifact:
    def test_load_live_run(self) -> None:
        run = load_run("lstm__varsmall__etth1__seed0", root=Path("artifacts"))
        assert isinstance(run, RunIntel)
        assert run.run_id == "lstm__varsmall__etth1__seed0"
        m = run.manifest
        assert m.get("dataset", {}).get("alias") == "etth1"
        assert m.get("model", {}).get("alias") == "lstm"
        assert run.curves_epochs, "loss curves should decode from curves.safetensors"
        assert any(p.is_best for p in run.curves_epochs), "≥1 epoch must be flagged best"
        assert run.predictions_sample is not None
        assert run.predictions_sample.shape[1] > 0  # horizon
        assert run.event_kinds.get("run_completed", 0) >= 1, \
            "at least 1 run_completed event expected"

    def test_render_markdown_smoke(self) -> None:
        run = load_run("lstm__varsmall__etth1__seed0", root=Path("artifacts"))
        md = render_run_markdown(run)
        assert isinstance(md, str) and len(md) > 200
        for must_have in (
            f"# Run: `{run.run_id}`",
            "## Purpose",
            "## Setup",
            "## Environment",
            "## Results",
            "## How to evaluate",
            "## Artifacts on disk",
            "best_val_mse",
            "manifest.yaml",
        ):
            assert must_have in md, f"missing section in rendered markdown: {must_have!r}"

    def test_md_includes_code_block(self) -> None:
        run = load_run("lstm__varsmall__etth1__seed0", root=Path("artifacts"))
        md = render_run_markdown(run)
        assert "```python" in md, "must include a python code block (re-eval example)"
        assert "```bash" in md, "must include a bash code block (re-train command)"
        assert "git checkout" in md


# ============================================================================
# Synthetic run — built in tmp_path, no reliance on live artifacts
# ============================================================================

def _write_minimal_run(
    root: Path,
    run_id: str = "r__m__v__s0",
    *,
    created_at: str = "2025-01-15T08:30:00",
) -> Path:
    """Write a minimal rundown run dir for testing load_run() + render_run_markdown()."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "seed": 0,
        "dataset": {
            "alias": "toy",
            "source": {"kind": "csv", "path": "data/toy.csv"},
            "target_col": "y",
            "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
        },
        "model": {
            "alias": "m",
            "variant": "s",
            "factory": {
                "target": "tests.fake.MyModel",
                "kwargs": {"in_dim": 4, "hidden": 8},
            },
        },
        "config": {
            "epochs_max": 3,
            "steps_per_epoch": 20,
            "optimizer": {"kind": "adam", "lr": 0.005},
            "lr": 0.005,
        },
        "code": {},
        "env": {
            "rundown_version": "0.1.0",
            "python": "3.10.21",
            "platform": "Windows",
            "hostname": "test-host",
            "argv": ["cli.py", "train", "--dataset", "toy", "--model", "m", "--seed", "0"],
            "cwd": str(root),
            "torch": "2.0.0",
            "numpy": "1.26",
            "git_commit": "1234567890abcdef",
            "git_dirty": False,
            "code_snapshot": {"format": "tar", "sha256_short": "deadbeef", "num_files": 5},
        },
        "metrics": {"best_val_mse": 0.123, "epochs_to_best": 2, "epochs_actual": 3},
    }
    import yaml

    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # events.jsonl — 3 epoch events
    events = [
        {"kind": "run_started", "t_iso": "2025-01-15T08:30:00"},
        {"kind": "epoch_completed", "epoch": 1, "val_mse": 0.20,
         "best_val": 0.20, "best_epoch": 1, "train_loss": 0.5},
        {"kind": "epoch_completed", "epoch": 2, "val_mse": 0.15,
         "best_val": 0.15, "best_epoch": 2, "train_loss": 0.3},
        {"kind": "epoch_completed", "epoch": 3, "val_mse": 0.12,
         "best_val": 0.12, "best_epoch": 3, "train_loss": 0.2},
        {"kind": "run_completed", "epochs_actual": 3, "best_val": 0.12, "early_stopped": False},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )

    # curves + predictions — written via safetensors.numpy (already a runtime dep)
    try:
        from safetensors.numpy import save_file
        import numpy as np

        save_file(
            {"train_loss": np.array([0.5, 0.3, 0.2], dtype=np.float32),
             "val_mse": np.array([0.20, 0.15, 0.12], dtype=np.float32)},
            str(run_dir / "curves.safetensors"),
        )
        save_file(
            {"y_hat": np.zeros((4, 5), dtype=np.float32),
             "y_true": np.zeros((4, 5), dtype=np.float32)},
            str(run_dir / "predictions" / "test_best.safetensors"),
        )
    except ImportError:
        pass  # curves/predictions optional

    # code snapshot meta (synthetic, no actual shared tar/zip)
    shared = root / "_shared"
    shared.mkdir(exist_ok=True)
    meta = {
        "format": "tar",
        "tracked_only": True,
        "sha256": "deadbeef" * 8,
        "sha256_short": "deadbeef",
        "size_bytes": 4096,
        "num_files": 5,
        "archive_path": "_shared/code_deadbeef/",
    }
    (shared / "code_deadbeef.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    return run_dir


class TestSyntheticRun:
    def test_load_run_minimal(self, tmp_path: Path) -> None:
        run_dir = _write_minimal_run(tmp_path)
        run = load_run(run_dir)
        assert run.run_id == "r__m__v__s0"
        assert run.manifest["dataset"]["alias"] == "toy"
        assert run.event_kinds["epoch_completed"] == 3
        assert run.code_snapshot is not None
        assert run.code_snapshot["sha256_short"] == "deadbeef"
        assert run.artifacts_on_disk, "artifacts_on_disk must list at least manifest.yaml"

    def test_load_run_by_id_via_root(self, tmp_path: Path) -> None:
        _write_minimal_run(tmp_path, run_id="r__m__v__s99")
        run = load_run("r__m__v__s99", root=tmp_path)
        assert run.run_id == "r__m__v__s99"

    def test_missing_run_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_run("__does_not_exist__", root=tmp_path)

    def test_render_smoke_keys(self, tmp_path: Path) -> None:
        run_dir = _write_minimal_run(tmp_path)
        run = load_run(run_dir)
        md = render_run_markdown(run)
        for must_have in (
            "# Run: `r__m__v__s0`",
            "## Purpose",
            "## Setup",
            "## Results",
            "## How to evaluate",
            "## Artifacts on disk",
            "0.123",         # best_val_mse rounded
            "tests.fake.MyModel",
        ):
            assert must_have in md, f"missing in synthetic render: {must_have!r}"

    def test_iter_recent_filters_by_dataset(self, tmp_path: Path) -> None:
        _write_minimal_run(tmp_path, "r__a__v__s0")
        _write_minimal_run(tmp_path, "r__b__v__s0")
        # overwrite manifest to give them distinct dataset aliases
        for run_dir in tmp_path.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            import yaml

            m = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
            m["dataset"]["alias"] = "a" if run_dir.name.startswith("r__a") else "b"
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump(m, allow_unicode=True), encoding="utf-8",
            )

        runs_a = iter_recent_runs(tmp_path, dataset="a")
        runs_b = iter_recent_runs(tmp_path, dataset="b")
        assert len(runs_a) == 1
        assert len(runs_b) == 1
        assert runs_a[0].run_id.startswith("r__a")

    def test_iter_recent_skips_underscore_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "_shared").mkdir()  # must NOT appear as a run
        runs = iter_recent_runs(tmp_path)
        assert runs == []

    def test_curves_decode_safetensors(self, tmp_path: Path) -> None:
        pytest.importorskip("safetensors")  # skip if dep missing
        run_dir = _write_minimal_run(tmp_path)
        run = load_run(run_dir)
        assert len(run.curves_epochs) == 3
        assert run.curves_epochs[-1].is_best  # epoch 3 = lowest val_mse
        assert run.curves_epochs[-1].val_mse == pytest.approx(0.12)

    def test_since_days_with_naive_created_at_does_not_crash(self, tmp_path: Path) -> None:
        """Regression: manifest.created_at is naive iso by default (engine.py).
        --since-days must compare against that without raising TypeError."""
        _write_minimal_run(tmp_path, "r__a__v__s0")  # created_at = "2025-01-15T08:30:00" (no tz)
        # The earlier bug was `ts.tzinfo()` (method call on attribute) → TypeError.
        runs = iter_recent_runs(tmp_path, since_days=7)
        assert isinstance(runs, list)
        # Either includes the run (if within window) or excludes it.
        # The contract is only that no exception escapes — the cutoff logic is exercised.

    def test_since_days_excludes_old_runs(self, tmp_path: Path) -> None:
        """Two runs with very different created_at — only the recent one survives --since-days."""
        # Use now-based timestamps so the test is robust to clock advancement.
        from datetime import datetime, timedelta, timezone
        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        old_iso = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        _write_minimal_run(tmp_path, "r__recent__v__s0", created_at=now_iso)
        _write_minimal_run(tmp_path, "r__old__v__s0", created_at=old_iso)

        runs = iter_recent_runs(tmp_path, since_days=7)
        ids = [r.run_id for r in runs]
        assert "r__recent__v__s0" in ids
        assert "r__old__v__s0" not in ids


# ============================================================================
# CLI integration
# ============================================================================

class TestCliInspect:
    """Each test calls `rundown.cli.main(["inspect", ...])` which sets rundown.events
    quiet state to True (via --quiet).  Without cleanup, that bleeds into later tests
    that expect quiet == False (e.g. TestEvents.test_emit_quiet_suppresses_stderr_but_writes_file)."""

    @pytest.fixture(autouse=True)
    def _reset_quiet(self):
        from rundown.events import set_quiet, clear_active_log
        set_quiet(False)
        clear_active_log()
        yield
        set_quiet(False)
        clear_active_log()

    def test_cli_single_run_id_outputs_markdown(self, tmp_path: Path, capsys) -> None:
        from rundown.cli import main

        _write_minimal_run(tmp_path, "r__m__v__s0")
        rc = main(["inspect", "r__m__v__s0", "--root", str(tmp_path), "--quiet"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "# Run:" in out
        assert "## Purpose" in out

    def test_cli_missing_run_exits_2(self, tmp_path: Path) -> None:
        from rundown.cli import main

        rc = main(["inspect", "__nope__", "--root", str(tmp_path), "--quiet"])
        assert rc == 2

    def test_cli_all_with_no_runs_exits_1(self, tmp_path: Path) -> None:
        from rundown.cli import main

        rc = main(["inspect", "--all", "--root", str(tmp_path), "--quiet"])
        assert rc == 1  # (no runs match)

    def test_cli_all_lists_each_run(self, tmp_path: Path, capsys) -> None:
        from rundown.cli import main

        _write_minimal_run(tmp_path, "r__a__v__s0")
        _write_minimal_run(tmp_path, "r__b__v__s0")
        rc = main(["inspect", "--all", "--root", str(tmp_path), "--quiet"])
        assert rc == 0
        out = capsys.readouterr().out
        assert out.count("# Run:") == 2


# ============================================================================
# C4 — generic metrics / curves rendering + Trainer line
# ============================================================================

class TestSyntheticCustomRun:
    """Custom trainer 风格的 manifest (custom metric + 任意 curves + trainer 名)
    渲染: trainer 行出现在 Setup, custom metric 出现在 Results, 通用曲线表
    出现, legacy sections 仍在. 不依赖真 CLI / 真 trainer (写纯 synthetic
    manifest).
    """

    def _write_custom_run(self, root: Path) -> Path:
        """写一份 minimal synthetic custom-trainer run (best_val_loss +
        任意 curves + config.trainer).
        """
        run_dir = root / "r__custom__s0"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "predictions").mkdir(exist_ok=True)

        manifest = {
            "run_id": "r__custom__s0",
            "created_at": "2025-01-15T08:30:00",
            "seed": 0,
            "dataset": {
                "alias": "toy",
                "source": {"kind": "csv", "path": "data/toy.csv"},
                "target_col": "y",
                "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
            },
            "model": {
                "alias": "m",
                "variant": None,
                "factory": {"target": "myproj.X", "kwargs": {"in_dim": 4}},
            },
            "config": {
                "epochs_max": 2,
                "steps_per_epoch": 10,
                "optimizer": {"kind": "adam", "lr": 0.001},
                "lr": 0.001,
                "trainer": "custom",
                "trainer_config": {"huber_delta": 0.5},
                "primary_metric": "best_val_loss",
            },
            "env": {
                "rundown_version": "0.1.0",
                "python": "3.10.21",
                "platform": "Windows",
                "hostname": "test-host",
                "argv": ["cli.py", "train", "--dataset", "toy", "--model", "m"],
                "cwd": str(root),
                "torch": "2.0.0",
                "numpy": "1.26",
            },
            "metrics": {
                "best_val_loss": 0.25,
                "custom_extra": 7,
                "epochs_actual": 2,
            },
        }
        (run_dir / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        # curves: stage1_loss / stage2_loss (任意键名, 没有 train_loss / val_mse
        # 触发 legacy table)
        from safetensors.numpy import save_file
        import numpy as np
        save_file({
            "stage1_loss": np.array([0.5, 0.4], dtype=np.float32),
            "stage2_loss": np.array([0.3, 0.2], dtype=np.float32),
        }, str(run_dir / "curves.safetensors"))
        save_file({
            "y_hat":  np.zeros((4, 5), dtype=np.float32),
            "y_true": np.zeros((4, 5), dtype=np.float32),
        }, str(run_dir / "predictions" / "test_best.safetensors"))

        return run_dir

    def test_render_includes_trainer_and_custom_metrics_and_curves(
        self, tmp_path: Path,
    ) -> None:
        run_dir = self._write_custom_run(tmp_path)
        run = load_run(run_dir)
        md = render_run_markdown(run)

        # legacy sections 都在
        for must in (
            "## Purpose", "## Setup", "## Environment",
            "## Results", "## How to evaluate", "## Artifacts on disk",
        ):
            assert must in md, f"missing legacy section {must!r}"

        # Trainer 行 (C4)
        assert "**Trainer**" in md
        assert "`custom`" in md

        # custom metrics 出现在 Results 表
        assert "best_val_loss" in md
        assert "custom_extra" in md

        # curves — 通用表 (没有 train_loss/val_mse 触发 legacy)
        assert "stage1_loss" in md
        assert "stage2_loss" in md

    def test_curves_dict_loaded_for_custom_run(self, tmp_path: Path) -> None:
        """RunIntel.curves 装载所有 1-D numeric series, 即使没有 train_loss/val_mse."""
        run_dir = self._write_custom_run(tmp_path)
        run = load_run(run_dir)
        assert "stage1_loss" in run.curves
        assert "stage2_loss" in run.curves
        # float32 精度 (写 safetensors 经 numpy round-trip) — 用 approx
        assert len(run.curves["stage1_loss"]) == 2
        assert run.curves["stage1_loss"][0] == pytest.approx(0.5)
        assert run.curves["stage1_loss"][1] == pytest.approx(0.4)
        assert run.curves["stage2_loss"][0] == pytest.approx(0.3)
        assert run.curves["stage2_loss"][1] == pytest.approx(0.2)
        # curves_epochs 空 (没 train_loss/val_mse)
        assert run.curves_epochs == []
