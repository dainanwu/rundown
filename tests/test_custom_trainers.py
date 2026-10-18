"""C4 — Custom trainer 模板集成测试.

覆盖 (per plan §6 / §E):

  1. CLI 跑通三个模板, 产物齐全 (manifest / weights / curves / predictions /
     events), config.trainer 写入, 关键 metric 出现.
  2. CustomLossTrainer: primary_metric=best_val_loss, metrics.best_val_loss 存在,
     curves.safetensors 含 val_loss.
  3. StagedTrainer: events 含 stage_started / stage_completed, curves 含
     stage1_train_loss / stage2_train_loss.
  4. JointTrainer: weights.safetensors keys 以 main./aux. 开头,
     metrics.aux_val_mse 存在, predictions/test_best.safetensors 存在.
  5. inspect_run.load_run + render_run_markdown 在三个 custom run 上不崩;
     custom metric / curve 名出现在 markdown; legacy sections 仍在.
  6. list_runs(root=..., metric="best_val_loss") 表格列展示该 metric 值;
     JSON 输出 unchanged.
  7. Experiment + run_experiment 跑通 CustomLossTrainer:
     trainer=examples.custom_trainers.custom_loss.CustomLossTrainer,
     trainer_config={"huber_delta": 0.5}, metric="best_val_loss",
     variants=["small"], steps_per_epoch=1, epochs_max=1.

跑得动 etth1 + lstm 是真 subprocess / 真训练, 不写 mock. 但 epochs_max=1 +
steps_per_epoch=2 让单次训练 < 2s, 总测试 < 30s.

inspect synthetic 测试在另一个文件 (tests/test_inspect_run.py).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

# bootstrap: tests/ has no conftest, rely on root cwd (pytest already sets it)
from rundown.inspect_run import (
    load_run,
    render_run_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
# `python -m` mode puts cwd on sys.path, so `examples.custom_trainers.X` resolves.
CMD_BASE = [PYTHON, "-m", "rundown.cli"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_cli_train(
    *,
    trainer_dotted: str,
    trainer_config: Any = None,
    out_dir: Path,
    epochs_max: int = 1,
    steps_per_epoch: int = 2,
    seed: int = 0,
    variant: str = "small",
) -> Dict[str, Any]:
    """跑一次 ``rundown train`` (subprocess), 返回 rc / stdout / stderr.

    cwd=PROJECT_ROOT 让 cwd 在 sys.path, `examples.custom_trainers.X` 可 import.
    使用 ``--out-dir`` 隔离到 tmp_path, 不污染 artifacts/.
    """
    import subprocess

    cmd = list(CMD_BASE) + [
        "train",
        "--dataset", "etth1",
        "--model",   "lstm",
        "--variant", variant,
        "--seed",    str(seed),
        "--epochs-max",        str(epochs_max),
        "--steps-per-epoch",   str(steps_per_epoch),
        "--trainer", trainer_dotted,
        "--out-dir", str(out_dir),
    ]
    if trainer_config is not None:
        cmd += ["--trainer-config", json.dumps(trainer_config)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), timeout=300,
    )
    return {
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _read_manifest(run_dir: Path) -> Dict[str, Any]:
    return yaml.safe_load(
        (run_dir / "manifest.yaml").read_text(encoding="utf-8")
    )


def _read_events(run_dir: Path) -> list:
    p = run_dir / "events.jsonl"
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_curves_keys(run_dir: Path) -> list:
    from safetensors import safe_open
    with safe_open(str(run_dir / "curves.safetensors"), framework="numpy") as f:
        return list(f.keys())


def _read_weights_keys(run_dir: Path) -> list:
    from safetensors import safe_open
    with safe_open(str(run_dir / "weights.safetensors"), framework="pt") as f:
        return list(f.keys())


def _run_id_for(out_dir: Path) -> Path:
    """Find the run dir under ``out_dir``. CLI 默认 run_id 格式:
    ``{model}[__var{variant}]__{dataset}__seed<N>`` — variant="small" → lstm__varsmall__etth1__seed0.
    """
    runs = [p for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    assert len(runs) == 1, f"expected 1 run, got {[r.name for r in runs]}"
    return runs[0]


# ---------------------------------------------------------------------------
# 1. CLI 跑通三个模板
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "trainer_dotted",
    [
        "examples.custom_trainers.custom_loss.CustomLossTrainer",
        "examples.custom_trainers.staged.StagedTrainer",
        "examples.custom_trainers.joint.JointTrainer",
    ],
)
class TestCliEndToEnd:
    """每个 trainer 跑通一次, 产物 + manifest 字段齐."""

    def test_cli_runs_and_writes_artifacts(self, tmp_path: Path, trainer_dotted: str) -> None:
        result = _run_cli_train(
            trainer_dotted=trainer_dotted,
            out_dir=tmp_path,
            epochs_max=1, steps_per_epoch=2,
        )
        assert result["rc"] == 0, (
            f"CLI 失败 (rc={result['rc']}): stdout={result['stdout']!r}, "
            f"stderr={result['stderr']!r}"
        )
        run_dir = _run_id_for(tmp_path)
        # 产物齐全
        assert (run_dir / "manifest.yaml").exists()
        assert (run_dir / "weights.safetensors").exists()
        assert (run_dir / "weights_best.safetensors").exists()
        assert (run_dir / "curves.safetensors").exists()
        assert (run_dir / "events.jsonl").exists()
        assert (run_dir / "predictions" / "test_best.safetensors").exists()

        # manifest.config.trainer / primary_metric 写入
        m = _read_manifest(run_dir)
        cfg = m["config"]
        assert cfg["trainer"] == trainer_dotted
        # required_provider_keys 应进 manifest
        assert cfg["trainer_contract"] == ["x", "y", "warm_start"]
        # trainer_config 默认空 (没传 --trainer-config)
        assert cfg["trainer_config"] == {}

        # primary_metric 三选一
        assert cfg["primary_metric"] in ("best_val_mse", "best_val_loss")


# ---------------------------------------------------------------------------
# 2. CustomLossTrainer: primary_metric=best_val_loss, curves 字段
# ---------------------------------------------------------------------------

class TestCustomLossTrainer:
    def test_primary_metric_and_curves(self, tmp_path: Path) -> None:
        result = _run_cli_train(
            trainer_dotted="examples.custom_trainers.custom_loss.CustomLossTrainer",
            out_dir=tmp_path,
            epochs_max=1, steps_per_epoch=2,
        )
        assert result["rc"] == 0, result["stderr"]
        run_dir = _run_id_for(tmp_path)

        m = _read_manifest(run_dir)
        # primary_metric = best_val_loss (这个 trainer 自己声明)
        assert m["config"]["primary_metric"] == "best_val_loss"

        # metrics 必备键
        metrics = m["metrics"]
        assert "best_val_loss" in metrics
        assert "best_val_mse" in metrics
        assert "epochs_to_best" in metrics
        assert "epochs_actual" in metrics

        # curves 字段
        curve_keys = _read_curves_keys(run_dir)
        assert "val_loss" in curve_keys
        assert "train_loss" in curve_keys

        # epoch_completed 事件带 val_loss
        events = _read_events(run_dir)
        ep = [e for e in events if e.get("kind") == "epoch_completed"]
        assert ep, "epoch_completed events missing"
        assert "val_loss" in ep[-1]
        assert "best_val_loss" in ep[-1]

    def test_trainer_config_override(self, tmp_path: Path) -> None:
        """--trainer-config 传 huber_delta=0.5 → manifest.config.trainer_config 落盘."""
        result = _run_cli_train(
            trainer_dotted="examples.custom_trainers.custom_loss.CustomLossTrainer",
            trainer_config={"huber_delta": 0.5, "l1_weight": 0.002},
            out_dir=tmp_path,
            epochs_max=1, steps_per_epoch=2,
        )
        assert result["rc"] == 0, result["stderr"]
        run_dir = _run_id_for(tmp_path)
        m = _read_manifest(run_dir)
        cfg_block = m["config"]
        assert cfg_block["trainer_config"] == {"huber_delta": 0.5, "l1_weight": 0.002}


# ---------------------------------------------------------------------------
# 3. StagedTrainer: events + curves
# ---------------------------------------------------------------------------

class TestStagedTrainer:
    def test_stage_events_and_curves(self, tmp_path: Path) -> None:
        result = _run_cli_train(
            trainer_dotted="examples.custom_trainers.staged.StagedTrainer",
            out_dir=tmp_path,
            epochs_max=1, steps_per_epoch=2,
        )
        assert result["rc"] == 0, result["stderr"]
        run_dir = _run_id_for(tmp_path)

        # events
        events = _read_events(run_dir)
        kinds = [e["kind"] for e in events]
        assert "stage_started" in kinds
        assert "stage_completed" in kinds
        # 默认 stages 是 2 个 → 各 2 个
        assert kinds.count("stage_started") == 2
        assert kinds.count("stage_completed") == 2

        # curves
        curve_keys = _read_curves_keys(run_dir)
        assert "stage1_train_loss" in curve_keys
        assert "stage2_train_loss" in curve_keys
        assert "val_mse" in curve_keys

        # metrics 含 stages_completed / best_val_mse
        m = _read_manifest(run_dir)
        assert m["config"]["primary_metric"] == "best_val_mse"
        assert "best_val_mse" in m["metrics"]
        assert m["metrics"]["stages_completed"] == 2


# ---------------------------------------------------------------------------
# 4. JointTrainer: 前缀权重 + 多 model
# ---------------------------------------------------------------------------

class TestJointTrainer:
    def test_prefix_weights_and_aux_metric(self, tmp_path: Path) -> None:
        result = _run_cli_train(
            trainer_dotted="examples.custom_trainers.joint.JointTrainer",
            out_dir=tmp_path,
            epochs_max=1, steps_per_epoch=2,
        )
        assert result["rc"] == 0, result["stderr"]
        run_dir = _run_id_for(tmp_path)

        # weights 前缀
        w_keys = _read_weights_keys(run_dir)
        assert any(k.startswith("main.") for k in w_keys), w_keys
        assert any(k.startswith("aux.") for k in w_keys), w_keys
        # 不存在无前缀的 key
        assert not any("." not in k for k in w_keys), f"un-prefixed keys: {w_keys}"

        # weights_best 同款前缀 (joint trainer 在 val 改善时落 best)
        from safetensors import safe_open
        with safe_open(str(run_dir / "weights_best.safetensors"), framework="pt") as f:
            wb_keys = list(f.keys())
        if wb_keys:  # 若训练 1 epoch 内 val 改善了 (常见), 应该非空
            assert any(k.startswith("main.") for k in wb_keys)
            assert any(k.startswith("aux.") for k in wb_keys)

        # metrics
        m = _read_manifest(run_dir)
        assert "aux_val_mse" in m["metrics"]
        assert "best_val_mse" in m["metrics"]

        # predictions
        assert (run_dir / "predictions" / "test_best.safetensors").exists()

        # curves 含 main / aux 分离
        curve_keys = _read_curves_keys(run_dir)
        assert "main_train_loss" in curve_keys
        assert "aux_train_loss" in curve_keys
        assert "val_mse" in curve_keys

    def test_missing_build_model_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ctx.build_model 缺 → ValueError (不静默).

        直接 import trainer 调 fit, 构造不带 build_model 的 ctx; 应抛 ValueError.
        """
        from examples.custom_trainers.joint import JointTrainer
        from rundown.trainers import TrainResult

        # 用 tests.fixtures.ToyNet 当主模型
        from tests.fixtures import RandomProvider, ToyNet
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig
        from rundown.events import clear_active_log, set_active_log
        from rundown.session import RunSession

        run_dir = tmp_path = PROJECT_ROOT / ".scratch" / "_joint_no_build_model"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = "x__y__seed0"
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-18T00:00:00", seed=0,
                dataset={"alias": "y"}, model={"alias": "x"},
                config={"epochs_max": 1},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            set_active_log(run_dir, run_id)
            model = ToyNet(in_dim=7, hidden=16)
            providers = {
                "train": RandomProvider(in_dim=7, batch_size=4, seed=0),
                "val":   RandomProvider(in_dim=7, batch_size=4, seed=1),
                "test":  RandomProvider(in_dim=7, batch_size=4, seed=2),
            }
            ctx = EngineConfig(
                run_id=run_id, seed=0, epochs_max=1, batch_size=4,
                steps_per_epoch=1, build_model=None,
            )
            session = RunSession(artifact=artifact, ctx=ctx,
                                 trainer_name="joint", primary_metric="best_val_mse")
            session.start(run_info={"seed": 0})

            with pytest.raises(ValueError, match="ctx.build_model"):
                JointTrainer().fit(model=model, providers=providers, ctx=ctx, session=session)
        finally:
            clear_active_log()
            set_quiet = __import__("rundown.events", fromlist=["set_quiet"]).set_quiet
            set_quiet(False)


# ---------------------------------------------------------------------------
# 5. inspect_run.load_run + render_run_markdown 在三个 custom run 上
# ---------------------------------------------------------------------------

def _run_one_trainer(out_dir: Path, trainer_dotted: str) -> Path:
    """跑一次 CLI, 返回 run_dir."""
    result = _run_cli_train(
        trainer_dotted=trainer_dotted,
        out_dir=out_dir,
        epochs_max=1, steps_per_epoch=2,
    )
    assert result["rc"] == 0, result["stderr"]
    return _run_id_for(out_dir)


@pytest.fixture
def custom_loss_run(tmp_path: Path) -> Path:
    out = tmp_path / "custom_loss"
    out.mkdir()
    return _run_one_trainer(
        out, "examples.custom_trainers.custom_loss.CustomLossTrainer",
    )


@pytest.fixture
def staged_run(tmp_path: Path) -> Path:
    out = tmp_path / "staged"
    out.mkdir()
    return _run_one_trainer(
        out, "examples.custom_trainers.staged.StagedTrainer",
    )


@pytest.fixture
def joint_run(tmp_path: Path) -> Path:
    out = tmp_path / "joint"
    out.mkdir()
    return _run_one_trainer(
        out, "examples.custom_trainers.joint.JointTrainer",
    )


class TestInspectCustomRuns:
    def test_render_custom_loss(self, custom_loss_run: Path) -> None:
        run = load_run(custom_loss_run)
        md = render_run_markdown(run)
        # 必须字段 (legacy sections 仍在)
        for must in (
            "## Purpose", "## Setup", "## Environment",
            "## Results", "## How to evaluate", "## Artifacts on disk",
        ):
            assert must in md, f"missing {must!r}"
        # trainer 在 Setup 段
        assert "**Trainer**" in md
        assert "custom_loss" in md
        # custom metric
        assert "best_val_loss" in md
        # curves — 应该用通用表 (curves_epochs 为空 — curves 有 train_loss + val_loss 但没 val_mse)
        assert "val_loss" in md

    def test_render_staged(self, staged_run: Path) -> None:
        run = load_run(staged_run)
        md = render_run_markdown(run)
        for must in (
            "## Setup", "## Results",
            "**Trainer**", "staged",
        ):
            assert must in md, f"missing {must!r}"
        # stage curves 出现在通用表里
        assert "stage1_train_loss" in md
        assert "stage2_train_loss" in md

    def test_render_joint(self, joint_run: Path) -> None:
        run = load_run(joint_run)
        md = render_run_markdown(run)
        for must in (
            "## Setup", "## Results",
            "**Trainer**", "joint",
            "best_val_mse", "aux_val_mse",
            "main_train_loss", "aux_train_loss",
        ):
            assert must in md, f"missing {must!r}"

    def test_curves_dict_populated_for_all(self, custom_loss_run: Path,
                                           staged_run: Path,
                                           joint_run: Path) -> None:
        # RunIntel.curves: 通用 series (不论 curves_epochs 是否解析成功)
        assert "train_loss" in load_run(custom_loss_run).curves
        assert "val_loss"   in load_run(custom_loss_run).curves
        assert "stage1_train_loss" in load_run(staged_run).curves
        assert "stage2_train_loss" in load_run(staged_run).curves
        assert "main_train_loss"   in load_run(joint_run).curves
        assert "aux_train_loss"    in load_run(joint_run).curves


# ---------------------------------------------------------------------------
# 6. list_runs(...metric="best_val_loss")
# ---------------------------------------------------------------------------

class TestListRunsMetric:
    def test_table_header_uses_metric(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # 跑一个 custom_loss run (它的 primary_metric = best_val_loss)
        result = _run_cli_train(
            trainer_dotted="examples.custom_trainers.custom_loss.CustomLossTrainer",
            out_dir=tmp_path,
            epochs_max=1, steps_per_epoch=2,
        )
        assert result["rc"] == 0, result["stderr"]

        from rundown.cli_query import list_runs

        rc = list_runs(
            root=tmp_path,
            metric="best_val_loss",
            fmt="table",
        )
        assert rc == 0
        out = capsys.readouterr().out
        # header 含 metric 名
        assert "best_val_loss" in out
        # 数值行 (4 位小数)
        run_id = _run_id_for(tmp_path).name
        assert run_id in out
        # 数值是 numeric (e.g. "0.xxxx"), 不是 "-"
        # 找一行含 run_id + numeric value
        has_value = False
        for line in out.splitlines():
            if run_id in line:
                tokens = line.split()
                # 第 5 个 token 是 metric value (run_id, dataset, model, variant, value, ...)
                if len(tokens) >= 5 and "." in tokens[4]:
                    has_value = True
                    break
        assert has_value, f"no numeric best_val_loss in output:\n{out}"

    def test_json_output_unchanged_with_metric_arg(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        result = _run_cli_train(
            trainer_dotted="examples.custom_trainers.custom_loss.CustomLossTrainer",
            out_dir=tmp_path,
            epochs_max=1, steps_per_epoch=2,
        )
        assert result["rc"] == 0, result["stderr"]

        from rundown.cli_query import list_runs

        # 用 --format json + --metric — JSON 含 metrics 全 dict, 跟 metric arg 无关.
        list_runs(root=tmp_path, metric="best_val_loss", fmt="json")
        out = capsys.readouterr().out
        rows = json.loads(out)
        assert len(rows) == 1
        r = rows[0]
        assert "best_val_loss" in r["metrics"]
        assert "best_val_mse"  in r["metrics"]


# ---------------------------------------------------------------------------
# 7. Experiment + run_experiment 跑 CustomLossTrainer
# ---------------------------------------------------------------------------

class TestExperimentCustomLoss:
    def test_experiment_runs_custom_loss_trainer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Experiment(trainer=CustomLossTrainer, metric=best_val_loss) 跑通一次 trial.

        用 monkeypatch 把 subprocess.run 替成: 写一个最小 manifest (含
        best_val_loss) 到 out_dir / <run_id>, 假装 CLI 跑通. 验证:
          1. ``--trainer`` 和 ``--trainer-config`` 透传给 CLI cmd
          2. ``metric`` 解析: ``exp.metric == "best_val_loss"`` 优先
          3. summary 的 reproduce 命令含 ``--trainer``
        """
        from rundown.experiment import Experiment, run_experiment

        captured: Dict[str, Any] = {"cmds": []}

        def fake_subprocess(cmd, *args, **kwargs):
            captured["cmds"].append(list(cmd))
            # 解析 --out-dir / --seed / --variant / --model / --dataset
            out_dir = None
            seed = 0
            variant = None
            model = "x"
            ds = "etth1"
            for i, tok in enumerate(cmd):
                if tok == "--out-dir" and i + 1 < len(cmd):
                    out_dir = Path(cmd[i + 1])
                elif tok == "--seed" and i + 1 < len(cmd):
                    seed = int(cmd[i + 1])
                elif tok == "--variant" and i + 1 < len(cmd):
                    variant = cmd[i + 1]
                elif tok == "--model" and i + 1 < len(cmd):
                    model = cmd[i + 1]
                elif tok == "--dataset" and i + 1 < len(cmd):
                    ds = cmd[i + 1]
            if out_dir is None:
                out_dir = tmp_path / "artifacts"
            var_suffix = f"__var{variant}" if variant else ""
            run_id = f"{model}{var_suffix}__{ds}__seed{seed}"
            run_dir = out_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.yaml").write_text(
                yaml.safe_dump({
                    "run_id": run_id,
                    "config": {
                        "trainer": "examples.custom_trainers.custom_loss.CustomLossTrainer",
                        "primary_metric": "best_val_loss",
                    },
                    "metrics": {"best_val_loss": 0.42, "best_val_mse": 0.50,
                                "epochs_to_best": 1, "epochs_actual": 1},
                }),
                encoding="utf-8",
            )
            return _FakeResult(rc=0, stdout=f"[run] {run_id}\n")

        import rundown.experiment as exp_mod
        monkeypatch.setattr(exp_mod.subprocess, "run", fake_subprocess)
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        exp = Experiment(
            name="custom_loss_sweep",
            dataset="etth1",
            model="lstm",
            epochs_max=1,
            variants=["small"],
            steps_per_epoch=1,
            trainer="examples.custom_trainers.custom_loss.CustomLossTrainer",
            trainer_config={"huber_delta": 0.5},
            metric="best_val_loss",
        )

        results = run_experiment(exp, out_root=tmp_path / "artifacts")
        assert len(results) == 1, results
        r = results[0]
        assert r.returncode == 0
        assert r.run_id == "lstm__varsmall__etth1__seed0"

        # metric 解析走 exp.metric → manifest.metrics["best_val_loss"]
        assert r.metric_name == "best_val_loss"
        assert r.best_metric is not None and r.best_metric == pytest.approx(0.42)

        # cmd 透传 trainer / trainer-config
        assert len(captured["cmds"]) == 1
        cmd = captured["cmds"][0]
        assert "--trainer" in cmd
        assert cmd[cmd.index("--trainer") + 1] == \
            "examples.custom_trainers.custom_loss.CustomLossTrainer"
        assert "--trainer-config" in cmd
        tc_json = cmd[cmd.index("--trainer-config") + 1]
        assert json.loads(tc_json) == {"huber_delta": 0.5}


class _FakeResult:
    def __init__(self, rc: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


# ---------------------------------------------------------------------------
# Syntactic helpers
# ---------------------------------------------------------------------------

def _run_id_for(out_dir: Path) -> Path:
    runs = [p for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    assert len(runs) == 1, f"expected 1 run, got {[r.name for r in runs]}"
    return runs[0]
