"""C3 — experiment-level trainer selection / trainer config pass-through / metric-agnostic summary.

覆盖:
    1. ``load_experiments`` 读 ``trainer`` / ``trainer_config`` / ``metric``;
       旧 experiments.yaml (无新字段) 仍走 ``None`` 默认.
    2. 无效 ``trainer`` (空 / bad alias / bad dotted) / 非 dict ``trainer_config``
       / 空 ``metric`` → ValueError.
    3. ``run_experiment`` 拼 subprocess 时加 ``--trainer`` 和 ``--trainer-config <json>``
       (monkeypatch + 解析 JSON 验证).
    4. metric 汇总: ``exp.metric is None`` 时读 ``manifest.config.primary_metric``,
       显示 ``custom_metric=0.2500``.
    5. ``exp.metric`` 覆盖 manifest primary metric.
    6. ``TrialResult.best_val_mse`` 兼容属性:
       metric_name="best_val_mse" → 透传 best_metric; metric_name="custom" → None.
    7. ``_read_best_val`` 仍优先 manifest / fallback events; ``_read_metric``
       在 custom metric 名时不 fallback events.

不真跑 subprocess (那需要真实 configs/ + 数据集); 全部走 monkeypatch +
captured cmd. 无网络 / GPU / 大文件.
"""
from __future__ import annotations
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml


# ---------------------------------------------------------------------------
# 共用 helper
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _write_exp_cfg(tmp_path: Path, body: str) -> Path:
    """写一份最小 ``configs/experiments.yaml``, 用 ``etth1`` / ``lstm`` 占位."""
    cfg = tmp_path / "configs"
    cfg.mkdir(exist_ok=True)
    (cfg / "experiments.yaml").write_text(
        "experiments:\n  e1:\n    dataset: etth1\n    model: lstm\n"
        "    epochs_max: 5\n" + body,
        encoding="utf-8",
    )
    return cfg


class _FakeSubprocessResult:
    """``subprocess.run`` 的最小替身, 配 ``_make_fake_subprocess`` 用."""

    def __init__(
        self,
        returncode: int = 0,
        stdout: str = "[run] fake__etth1__seed0\n",
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_fake_subprocess(
    captured: Dict[str, Any],
    manifest_writer: Optional[Any] = None,
):
    """构造 ``subprocess.run`` 替身.

    Args:
        captured: 写入 ``captured["cmds"]`` (list). 多个 trial 时全部记录.
        manifest_writer: 可选 callable(cmd, run_dir, captured) -> None.
            给 fake_run 一个钩子, 让它在 trial_dir 内写 manifest.yaml
            (场景 4 / 5 需要).
    """
    def fake_run(cmd, *args, **kwargs):
        captured["cmds"].append(cmd)

        # 找 --out-dir 的下一个参数, 拿 out_root.
        out_root: Optional[Path] = None
        for i, arg in enumerate(cmd):
            if arg == "--out-dir" and i + 1 < len(cmd):
                out_root = Path(cmd[i + 1])
                break

        # 找 stdout 里 [run] <id>, 拿 run_id.
        import re as _re
        m = _re.search(r"^\[run\]\s+(\S+)", captured.get("forced_stdout", "[run] fake__etth1__seed0\n"))
        run_id = m.group(1) if m else "fake__etth1__seed0"

        run_dir = out_root / run_id if out_root else None
        if manifest_writer is not None and run_dir is not None:
            manifest_writer(cmd, run_dir, captured)

        return _FakeSubprocessResult(
            stdout=captured.get("forced_stdout", "[run] fake__etth1__seed0\n"),
        )

    return fake_run


# ---------------------------------------------------------------------------
# 1. load_experiments 读 trainer / trainer_config / metric
# ---------------------------------------------------------------------------

class TestLoadExperimentsTrainerFields:
    def test_trainer_fields_default_to_none(self, tmp_path):
        """不写 trainer / trainer_config / metric → 全 None (旧 yaml 兼容)."""
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "")
        exps = load_experiments(cfg)
        e = exps["e1"]
        assert e.trainer is None
        assert e.trainer_config is None
        assert e.metric is None

    def test_trainer_alias_loaded(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "    trainer: joint\n")
        exps = load_experiments(cfg)
        assert exps["e1"].trainer == "joint"

    def test_trainer_dotted_loaded(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(
            tmp_path,
            "    trainer: myproj.trainers.JointTrainer\n",
        )
        exps = load_experiments(cfg)
        assert exps["e1"].trainer == "myproj.trainers.JointTrainer"

    def test_trainer_config_loaded(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(
            tmp_path,
            "    trainer: joint\n"
            "    trainer_config:\n"
            "      alpha: 0.5\n"
            "      stages: [warmup, finetune]\n",
        )
        exps = load_experiments(cfg)
        e = exps["e1"]
        assert e.trainer == "joint"
        assert e.trainer_config == {
            "alpha": 0.5, "stages": ["warmup", "finetune"],
        }

    def test_metric_loaded(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "    metric: best_val_loss\n")
        exps = load_experiments(cfg)
        assert exps["e1"].metric == "best_val_loss"


# ---------------------------------------------------------------------------
# 2. 校验: 无效 trainer / trainer_config / metric → ValueError
# ---------------------------------------------------------------------------

class TestLoadExperimentsTrainerValidation:
    def test_trainer_empty_string_rejected(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "    trainer: ''\n")
        with pytest.raises(ValueError, match="non-empty str"):
            load_experiments(cfg)

    def test_trainer_non_string_rejected(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "    trainer: 42\n")
        with pytest.raises(ValueError, match="non-empty str"):
            load_experiments(cfg)

    def test_trainer_bad_alias_rejected(self, tmp_path):
        """alias 不满足 trainer_alias 正则 (e.g. 大写)."""
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "    trainer: Has-Cap\n")
        with pytest.raises(ValueError, match="invalid trainer_alias"):
            load_experiments(cfg)

    def test_trainer_bad_dotted_rejected(self, tmp_path):
        """dotted path 但 attr 为空 ("mod.") → ValueError."""
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "    trainer: mod.\n")
        with pytest.raises(ValueError, match="dotted path"):
            load_experiments(cfg)

        cfg = _write_exp_cfg(tmp_path, "    trainer: '.attr'\n")
        with pytest.raises(ValueError, match="dotted path"):
            load_experiments(cfg)

    def test_trainer_config_not_dict_rejected(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(
            tmp_path,
            "    trainer: joint\n"
            "    trainer_config: [alpha, beta]\n",  # list, not dict
        )
        with pytest.raises(ValueError, match="must be a mapping"):
            load_experiments(cfg)

    def test_metric_empty_string_rejected(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "    metric: ''\n")
        with pytest.raises(ValueError, match="non-empty str"):
            load_experiments(cfg)

    def test_metric_non_string_rejected(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = _write_exp_cfg(tmp_path, "    metric: 1\n")
        with pytest.raises(ValueError, match="non-empty str"):
            load_experiments(cfg)


# ---------------------------------------------------------------------------
# 3. run_experiment 透传 --trainer / --trainer-config <json> 给 subprocess
# ---------------------------------------------------------------------------

class TestRunExperimentTrainerPassThrough:
    def _capture_run(
        self,
        monkeypatch,
        manifest_writer=None,
    ) -> Dict[str, Any]:
        captured: Dict[str, Any] = {"cmds": []}
        import rundown.experiment as exp_mod
        monkeypatch.setattr(
            exp_mod.subprocess, "run",
            _make_fake_subprocess(captured, manifest_writer=manifest_writer),
        )
        # 干掉 snapshot, 不写 _shared/
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)
        return captured

    def test_passes_trainer_to_subprocess(self, tmp_path, monkeypatch):
        from rundown.experiment import Experiment
        captured = self._capture_run(monkeypatch)

        exp = Experiment(
            name="t", dataset="etth1", model="lstm", epochs_max=1,
            variants=["small"], trainer="myproj.trainers.JointTrainer",
        )
        run_dir_root = tmp_path / "artifacts"
        # 让 fake_run 把 manifest 写到 trial_dir, 避免 _read_metric 兜底 None
        def _write_manifest(cmd, run_dir, cap):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump({"metrics": {"best_val_mse": 0.1},
                                "config": {"primary_metric": "best_val_mse"}}),
                encoding="utf-8",
            )
        # 把 manifest_writer 重新挂上 (上面 _capture_run 没传)
        import rundown.experiment as exp_mod
        captured["cmds"] = []
        monkeypatch.setattr(
            exp_mod.subprocess, "run",
            _make_fake_subprocess(captured, manifest_writer=_write_manifest),
        )

        from rundown.experiment import run_experiment
        run_experiment(exp, out_root=run_dir_root)

        assert len(captured["cmds"]) == 1
        cmd = captured["cmds"][0]
        assert "--trainer" in cmd
        idx = cmd.index("--trainer")
        assert cmd[idx + 1] == "myproj.trainers.JointTrainer"

    def test_passes_trainer_config_as_json(self, tmp_path, monkeypatch):
        """--trainer-config <json> 必须出现; json 解析要等于 trainer_config."""
        from rundown.experiment import Experiment, run_experiment
        import rundown.experiment as exp_mod

        captured: Dict[str, Any] = {"cmds": []}

        def _write_manifest(cmd, run_dir, cap):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump({"metrics": {"best_val_mse": 0.1}}),
                encoding="utf-8",
            )

        monkeypatch.setattr(
            exp_mod.subprocess, "run",
            _make_fake_subprocess(captured, manifest_writer=_write_manifest),
        )
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        exp = Experiment(
            name="t", dataset="etth1", model="lstm", epochs_max=1,
            variants=["small"],
            trainer="myproj.trainers.JointTrainer",
            trainer_config={"alpha": 0.5, "stages": ["warmup", "finetune"]},
        )
        run_experiment(exp, out_root=tmp_path / "artifacts")

        cmd = captured["cmds"][0]
        assert "--trainer-config" in cmd
        idx = cmd.index("--trainer-config")
        json_blob = cmd[idx + 1]
        # ASCII-safe, sorted keys (L10 / L15)
        assert json_blob == json.dumps(
            {"alpha": 0.5, "stages": ["warmup", "finetune"]},
            ensure_ascii=True, sort_keys=True,
        )
        parsed = json.loads(json_blob)
        assert parsed == {"alpha": 0.5, "stages": ["warmup", "finetune"]}

    def test_no_trainer_no_trainer_flag_in_subprocess(self, tmp_path, monkeypatch):
        """不设 trainer → cmd 不含 --trainer / --trainer-config."""
        from rundown.experiment import Experiment, run_experiment
        import rundown.experiment as exp_mod

        captured: Dict[str, Any] = {"cmds": []}

        def _write_manifest(cmd, run_dir, cap):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump({"metrics": {"best_val_mse": 0.1}}),
                encoding="utf-8",
            )

        monkeypatch.setattr(
            exp_mod.subprocess, "run",
            _make_fake_subprocess(captured, manifest_writer=_write_manifest),
        )
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        exp = Experiment(
            name="t", dataset="etth1", model="lstm", epochs_max=1,
            variants=["small"],
        )
        run_experiment(exp, out_root=tmp_path / "artifacts")

        cmd = captured["cmds"][0]
        assert "--trainer" not in cmd
        assert "--trainer-config" not in cmd

    def test_non_json_serializable_trainer_config_raises_value_error(
        self, tmp_path, monkeypatch,
    ):
        """trainer_config 含非 JSON 可序列化对象 → 启动前 ValueError.

        不进入 trial 循环. JSON 校验早于 subprocess 调用, 让 CLI 提前
        报清晰错误 (CLI 在 _cmd_experiment_run 接住, rc=2).
        """
        from rundown.experiment import Experiment, run_experiment
        import rundown.experiment as exp_mod

        called = {"n": 0}
        def fake_run(*args, **kwargs):
            called["n"] += 1
            return _FakeSubprocessResult()
        monkeypatch.setattr(exp_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        exp = Experiment(
            name="t", dataset="etth1", model="lstm", epochs_max=1,
            variants=["small"],
            trainer_config={"unsupported": {1, 2, 3}},  # set 不是 JSON
        )
        with pytest.raises(ValueError, match="not JSON serializable"):
            run_experiment(exp, out_root=tmp_path / "artifacts")
        assert called["n"] == 0, "JSON 校验必须在 trial 循环之前"


# ---------------------------------------------------------------------------
# 4. metric summary: exp.metric is None → 读 manifest.config.primary_metric
# ---------------------------------------------------------------------------

class TestMetricSummaryFromManifest:
    def test_metric_from_manifest_primary_metric(self, tmp_path, monkeypatch, capsys):
        """exp.metric 为 None → 读 manifest.config.primary_metric = "custom_metric"."""
        from rundown.experiment import Experiment, run_experiment
        import rundown.experiment as exp_mod

        captured: Dict[str, Any] = {"cmds": []}

        def _write_manifest(cmd, run_dir, cap):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump({
                    "metrics": {"custom_metric": 0.25, "best_val_mse": 0.99},
                    "config": {"primary_metric": "custom_metric"},
                }),
                encoding="utf-8",
            )

        monkeypatch.setattr(
            exp_mod.subprocess, "run",
            _make_fake_subprocess(captured, manifest_writer=_write_manifest),
        )
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        exp = Experiment(
            name="t", dataset="etth1", model="lstm", epochs_max=1,
            variants=["small"],
            # exp.metric 不传 → None
        )
        run_experiment(exp, out_root=tmp_path / "artifacts")

        out = capsys.readouterr().out
        # trial 行 + summary 行都应显示 custom_metric
        assert "custom_metric=0.2500" in out, (
            f"expected 'custom_metric=0.2500' in stdout; got:\n{out}"
        )
        # summary 行
        assert "[summary] best:" in out
        assert "custom_metric=0.2500" in out.split("[summary]")[1]


# ---------------------------------------------------------------------------
# 5. exp.metric 覆盖 manifest primary metric
# ---------------------------------------------------------------------------

class TestExperimentMetricOverrides:
    def test_exp_metric_overrides_primary(self, tmp_path, monkeypatch, capsys):
        """manifest.primary_metric = "A", metrics = {A: 1.0, B: 0.5};
        exp.metric = "B" → summary 用 B=0.5000, 不是 A=1.0.
        """
        from rundown.experiment import Experiment, run_experiment
        import rundown.experiment as exp_mod

        captured: Dict[str, Any] = {"cmds": []}

        def _write_manifest(cmd, run_dir, cap):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump({
                    "metrics": {"A": 1.0, "B": 0.5},
                    "config": {"primary_metric": "A"},
                }),
                encoding="utf-8",
            )

        monkeypatch.setattr(
            exp_mod.subprocess, "run",
            _make_fake_subprocess(captured, manifest_writer=_write_manifest),
        )
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        exp = Experiment(
            name="t", dataset="etth1", model="lstm", epochs_max=1,
            variants=["small"],
            metric="B",
        )
        run_experiment(exp, out_root=tmp_path / "artifacts")

        out = capsys.readouterr().out
        summary = out.split("[summary]")[1] if "[summary]" in out else ""
        assert "B=0.5000" in summary, (
            f"expected 'B=0.5000' in summary; got:\n{summary}"
        )
        # 不应误把 A 写成 best
        assert "best_val_mse" not in summary
        assert "A=1.0000" not in summary


# ---------------------------------------------------------------------------
# 6. TrialResult.best_val_mse 兼容属性
# ---------------------------------------------------------------------------

class TestTrialResultBestValMseCompat:
    def test_compat_when_metric_name_is_best_val_mse(self):
        """metric_name='best_val_mse' → best_val_mse 属性返回 best_metric."""
        from rundown.experiment import TrialResult
        r = TrialResult(
            variant="small", seed=0, best_metric=0.5,
            run_dir=Path("/tmp/r"), run_id="x__y__seed0",
            duration_s=1.0, returncode=0,
            metric_name="best_val_mse",
        )
        assert r.best_val_mse == 0.5

    def test_compat_when_metric_name_is_none(self):
        """metric_name=None → 旧构造 (没传 metric_name), 仍返回 best_metric."""
        from rundown.experiment import TrialResult
        r = TrialResult(
            variant=None, seed=0, best_metric=0.7,
            run_dir=Path("/tmp/r"), run_id="x__y__seed0",
            duration_s=1.0, returncode=0,
        )
        assert r.best_val_mse == 0.7

    def test_compat_returns_none_when_metric_name_is_custom(self):
        """metric_name='custom_metric' → best_val_mse 属性 None, 防止误读."""
        from rundown.experiment import TrialResult
        r = TrialResult(
            variant=None, seed=0, best_metric=0.5,
            run_dir=Path("/tmp/r"), run_id="x__y__seed0",
            duration_s=1.0, returncode=0,
            metric_name="custom_metric",
        )
        assert r.best_val_mse is None
        # best_metric 仍可读 (新字段)
        assert r.best_metric == 0.5
        assert r.metric_name == "custom_metric"


# ---------------------------------------------------------------------------
# 7. _read_best_val 仍兼容 (Fix D 回归) + _read_metric 不为 custom metric 兜底 events
# ---------------------------------------------------------------------------

class TestReadMetricAndReadBestVal:
    def test_read_best_val_prefers_manifest(self, tmp_path):
        """Fix D 回归: manifest.metrics.best_val_mse 优先, events 不被采用."""
        from rundown.experiment import _read_best_val
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "manifest.yaml").write_text(
            yaml.safe_dump({"metrics": {"best_val_mse": 0.123}}),
            encoding="utf-8",
        )
        (run_dir / "events.jsonl").write_text(
            json.dumps({
                "kind": "run_completed",
                    "best_val": 9.9,
                    "run_id": "x", "epochs_actual": 1, "early_stopped": False,
                    "t": 0.0, "t_iso": "2026-10-18T00:00:00",
                }) + "\n",
            encoding="utf-8",
        )
        assert _read_best_val(run_dir) == 0.123

    def test_read_best_val_falls_back_to_events(self, tmp_path):
        """Fix D 回归: manifest 缺失 → events.jsonl 兜底."""
        from rundown.experiment import _read_best_val
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "events.jsonl").write_text(
            json.dumps({
                "kind": "run_completed",
                "best_val": 5.5,
                "run_id": "y", "epochs_actual": 2, "early_stopped": False,
                "t": 0.0, "t_iso": "2026-10-18T00:00:00",
            }) + "\n",
            encoding="utf-8",
        )
        assert _read_best_val(run_dir) == 5.5

    def test_read_metric_returns_custom_metric_value(self, tmp_path):
        """_read_metric(run_dir, 'custom') 从 manifest 读, 返回 (value, name)."""
        from rundown.experiment import _read_metric
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "manifest.yaml").write_text(
            yaml.safe_dump({
                "metrics": {"custom_metric": 0.42, "best_val_mse": 0.99},
                "config": {"primary_metric": "best_val_mse"},
            }),
            encoding="utf-8",
        )
        v, name = _read_metric(run_dir, "custom_metric")
        assert v == pytest.approx(0.42)
        assert name == "custom_metric"

    def test_read_metric_no_fallback_to_events_for_custom_name(self, tmp_path):
        """_read_metric 收到 custom metric 但 manifest 没 → (None, name);
        不应 fallback 到 events.jsonl (那只有 best_val 字段, 不是 custom).
        """
        from rundown.experiment import _read_metric
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "manifest.yaml").write_text(
            yaml.safe_dump({
                "metrics": {"best_val_mse": 9.9},  # 没有 custom_metric
                "config": {"primary_metric": "best_val_mse"},
            }),
            encoding="utf-8",
        )
        # events.jsonl 里有 best_val=5.5, 但 custom_metric 不应 fallback.
        (run_dir / "events.jsonl").write_text(
            json.dumps({
                "kind": "run_completed", "best_val": 5.5,
                "run_id": "x", "epochs_actual": 1, "early_stopped": False,
                "t": 0.0, "t_iso": "2026-10-18T00:00:00",
            }) + "\n",
            encoding="utf-8",
        )
        v, name = _read_metric(run_dir, "custom_metric")
        assert v is None
        assert name == "custom_metric"

    def test_read_metric_resolves_primary_metric_when_name_none(self, tmp_path):
        """metric_name=None → 走 manifest.config.primary_metric."""
        from rundown.experiment import _read_metric
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        (run_dir / "manifest.yaml").write_text(
            yaml.safe_dump({
                "metrics": {"custom_metric": 0.7, "best_val_mse": 9.9},
                "config": {"primary_metric": "custom_metric"},
            }),
            encoding="utf-8",
        )
        v, name = _read_metric(run_dir, None)
        assert v == pytest.approx(0.7)
        assert name == "custom_metric"

    def test_read_metric_falls_back_to_best_val_mse_default(self, tmp_path):
        """metric_name=None + manifest 没 primary_metric → 走默认 best_val_mse,
        events.jsonl 仍可 fallback (旧 run 兼容).
        """
        from rundown.experiment import _read_metric
        run_dir = tmp_path / "r"
        run_dir.mkdir()
        # 没有 manifest, 只有 events
        (run_dir / "events.jsonl").write_text(
            json.dumps({
                "kind": "run_completed", "best_val": 4.2,
                "run_id": "x", "epochs_actual": 1, "early_stopped": False,
                "t": 0.0, "t_iso": "2026-10-18T00:00:00",
            }) + "\n",
            encoding="utf-8",
        )
        v, name = _read_metric(run_dir, None)
        assert v == pytest.approx(4.2)
        assert name == "best_val_mse"


# ---------------------------------------------------------------------------
# 8. Experiment dataclass 字段顺序 / 默认值
# ---------------------------------------------------------------------------

class TestExperimentDataclass:
    def test_field_order_preserved(self):
        """C3 新字段必须在 optimizer 之后, 不动旧字段顺序.

        旧 caller 可能按位置传参, 不能让 trainer 落到 batch_size 槽.
        """
        from rundown.experiment import Experiment
        from dataclasses import fields
        names = [f.name for f in fields(Experiment)]
        # 旧字段
        assert names.index("optimizer") < names.index("trainer")
        assert names.index("trainer") < names.index("trainer_config")
        assert names.index("trainer_config") < names.index("metric")

    def test_defaults_are_none(self):
        from rundown.experiment import Experiment
        e = Experiment(name="x", dataset="d", model="m", epochs_max=1)
        assert e.trainer is None
        assert e.trainer_config is None
        assert e.metric is None


# ---------------------------------------------------------------------------
# 9. CLI experiment run --trainer override
# ---------------------------------------------------------------------------

class TestCliExperimentTrainerOverride:
    def test_experiment_run_has_trainer_flag(self):
        """CLI experiment run 解析必须有 --trainer, 默认 None."""
        from rundown.cli import build_parser
        parser = build_parser()
        ns = parser.parse_args(["experiment", "run", "e1"])
        assert ns.trainer is None
        ns = parser.parse_args(["experiment", "run", "e1", "--trainer", "foo.bar.Baz"])
        assert ns.trainer == "foo.bar.Baz"

    def test_cli_experiment_run_trainer_overrides_yaml(
        self, tmp_path, monkeypatch,
    ):
        """CLI --trainer 覆盖 experiments.yaml 里的 trainer, subprocess 应收到 CLI 的值."""
        cfg = _write_exp_cfg(
            tmp_path,
            "    trainer: yaml_trainer_alias\n",  # YAML 里有一个 trainer
        )

        # 写 configs/{datasets,models}.yaml 占位, 让 load_experiments 不报错;
        # 实际不真跑 train, 只检查 sub command.
        (cfg / "datasets.yaml").write_text(
            "datasets:\n  etth1: {}\n", encoding="utf-8",
        )
        (cfg / "models.yaml").write_text(
            "models:\n  lstm: {factory: {target: x}}\n", encoding="utf-8",
        )

        monkeypatch.chdir(tmp_path)
        from rundown.cli import main
        import rundown.experiment as exp_mod

        captured: Dict[str, Any] = {"cmds": []}

        def _write_manifest(cmd, run_dir, cap):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump({"metrics": {"best_val_mse": 0.1}}),
                encoding="utf-8",
            )

        monkeypatch.setattr(
            exp_mod.subprocess, "run",
            _make_fake_subprocess(captured, manifest_writer=_write_manifest),
        )
        # snapshot 跳过 (避免写 _shared/)
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        rc = main([
            "experiment", "run", "e1",
            "--trainer", "cli_overridden_trainer",
        ])
        assert rc == 0, f"CLI 失败: rc={rc}"

        # 至少一个 trial, cmd 的 --trainer 后应是 CLI 覆盖值
        assert len(captured["cmds"]) >= 1
        cmd = captured["cmds"][0]
        idx = cmd.index("--trainer")
        assert cmd[idx + 1] == "cli_overridden_trainer"