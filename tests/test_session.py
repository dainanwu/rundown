"""RunSession 单元测试 — start / finalize / fail / 幂等.

C1 范围: ``rundown/session.py`` 暴露的 RunSession 是 run 生命周期 owner.
engine.fit() 内部走 session; 这里直接测 session 的契约.

测试契约:
    - start() 保存初始 manifest, 激活 events (清空旧 events.jsonl),
      捕获 env 到内存, emit run_started (含 trainer + run_info 字段).
    - finalize() 写 metrics / predictions_primary / primary_metric /
      env, 保存 manifest, emit run_completed, clear_active_log.
    - finalize() 幂等 — 第二次 no-op (不重复 emit run_completed).
    - fail(exc) emit run_failed (含 error_type / error), 不 emit run_completed,
      保存 env, clear_active_log.
    - fail() 在 finalize 之后 no-op (L13).
    - checkpoint_best(state) 写 weights_best.safetensors.

与 test_smoke.py 同款的 fixture 模式: set_quiet(False) + try/finally +
clear_active_log() 收尾 — 不污染模块全局.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn as nn
import yaml

from rundown.artifact import RunArtifact, RunManifest
from rundown.engine import EngineConfig
from rundown.events import (
    clear_active_log,
    is_quiet,
    set_active_log,
    set_quiet,
)
from rundown.session import RunSession
from tests.fixtures import RandomProvider, ToyNet


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_run(
    tmp_path: Path,
    *,
    run_id: str = "toy__smoke__seed0",
    epochs_max: int = 1,
    dataset_alias: str = "smoke",
    model_alias: str = "toy",
    variant: Optional[str] = None,
    seed: int = 0,
    **engine_overrides: Any,
):
    """构造 (RunArtifact, EngineConfig) 给 RunSession 测试用.

    不调 set_active_log (session.start 自己激活).
    """
    run_dir = tmp_path / "artifacts" / run_id
    manifest = RunManifest(
        run_id=run_id,
        created_at="2026-10-18T00:00:00",
        seed=seed,
        dataset={"alias": dataset_alias, "source": "random"},
        model={"alias": model_alias, "variant": variant,
               "factory": {"target": "tests.fixtures.ToyNet"}},
        config={"epochs_max": epochs_max, "warm_start": 4, "rollout": 16},
    )
    artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
    ctx = EngineConfig(
        run_id=run_id, seed=seed,
        epochs_max=epochs_max, batch_size=4, steps_per_epoch=1,
        **engine_overrides,
    )
    return artifact, ctx


def _read_events(run_dir: Path) -> list:
    p = run_dir / "events.jsonl"
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(ln) for ln in text.split("\n") if ln.strip()]


# ---------------------------------------------------------------------------
# happy path: start → write → finalize
# ---------------------------------------------------------------------------

class TestRunSessionLifecycle:
    def test_start_finalize_writes_manifest_and_events(self, tmp_path):
        """start() → 写产物 (权重/曲线/预测) → finalize():

        - manifest 含 metrics / config.predictions_primary / config.primary_metric / env
        - events 含 run_started + run_completed (各 1 次)
        - active log 已清
        """
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)

            session = RunSession(artifact=artifact, ctx=ctx, trainer_name="gradient")
            session.start(run_info={
                "params": sum(p.numel() for p in ToyNet().parameters()),
                "epochs_max": ctx.epochs_max,
                "steps_per_epoch": ctx.steps_per_epoch,
                "lr": ctx.lr,
                "seed": ctx.seed,
            })

            # 写最小产物
            model = ToyNet(in_dim=7, hidden=16)
            state = {k: v.detach().cpu().contiguous().clone()
                     for k, v in model.state_dict().items()}
            session.checkpoint_best(state)
            session.write_weights(state, best=False)
            session.write_curves({
                "train_loss": torch.tensor([0.5], dtype=torch.float32),
                "val_mse":    torch.tensor([0.7], dtype=torch.float32),
            })
            session.write_predictions("test_best", {
                "y_hat":  torch.zeros(1, 4),
                "y_true": torch.zeros(1, 4),
            })

            session.finalize(
                metrics={"best_val_mse": 0.7, "epochs_to_best": 1, "epochs_actual": 1},
                predictions_primary="test_best",
                epochs_actual=1, early_stopped=False, best_val=0.7,
            )
        finally:
            clear_active_log()
            set_quiet(False)

        # 1) manifest 落地且字段齐
        m = yaml.safe_load((artifact.run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        assert m["metrics"]["best_val_mse"] == 0.7
        assert m["metrics"]["epochs_actual"] == 1
        assert m["config"]["predictions_primary"] == "test_best"
        # primary_metric 默认 best_val_mse (GradientStrategy 默认值, 未显式设)
        assert m["config"]["primary_metric"] == "best_val_mse"
        # env 由 start() 捕获, finalize() 落盘 — 含 runddown_version
        assert "rundown_version" in m["env"]
        assert m["env"]["python"]

        # 2) events: run_started (session.start) + run_completed (session.finalize), 各 1
        events = _read_events(artifact.run_dir)
        kinds = [e["kind"] for e in events]
        assert kinds.count("run_started") == 1
        assert kinds.count("run_completed") == 1
        # run_completed 必须包含 metrics (新字段, 但兼容旧 key 不丢)
        rc = next(e for e in events if e["kind"] == "run_completed")
        assert rc["epochs_actual"] == 1
        assert rc["best_val"] == 0.7
        assert rc["early_stopped"] is False
        assert rc["metrics"]["best_val_mse"] == 0.7

        # 3) active log 已清
        from rundown.events import active_run_id
        assert active_run_id() is None

        # 4) 物理产物齐全
        assert (artifact.run_dir / "weights.safetensors").exists()
        assert (artifact.run_dir / "weights_best.safetensors").exists()
        assert (artifact.run_dir / "curves.safetensors").exists()
        assert (artifact.run_dir / "predictions" / "test_best.safetensors").exists()

    def test_checkpoint_best_writes_weights_best_safetensors(self, tmp_path):
        """checkpoint_best(state) 写 weights_best.safetensors."""
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)
            session = RunSession(artifact=artifact, ctx=ctx, trainer_name="gradient")
            session.start(run_info={"seed": ctx.seed})

            model = ToyNet(in_dim=7, hidden=16)
            state = {k: v.detach().cpu().contiguous().clone()
                     for k, v in model.state_dict().items()}
            path = session.checkpoint_best(state)
            assert path == artifact.run_dir / "weights_best.safetensors"
            assert path.exists()

            session.finalize(
                metrics={"best_val_mse": 1.0, "epochs_actual": 1},
                predictions_primary="test_best",
                epochs_actual=1, early_stopped=False, best_val=1.0,
            )
        finally:
            clear_active_log()
            set_quiet(False)

    def test_run_started_payload_preserves_run_info_keys(self, tmp_path):
        """run_started 必须保留 run_info 提供的 params / epochs_max /
        steps_per_epoch / lr (现行 engine 契约, 不可丢).

        即使 session 默认会注入 dataset / model / variant / seed / trainer,
        caller 提供的 run_info 仍要原样出现在 events.jsonl 里.
        """
        set_quiet(False)
        try:
            artifact, ctx = _make_run(
                tmp_path, epochs_max=3, dataset_alias="my_data",
                model_alias="my_net", variant="small",
            )
            session = RunSession(artifact=artifact, ctx=ctx, trainer_name="gradient")
            session.start(run_info={
                "params": 12345,
                "epochs_max": 3,
                "steps_per_epoch": 7,
                "lr": 0.0005,
                "seed": 0,
            })

            session.finalize(
                metrics={"best_val_mse": 0.5, "epochs_actual": 3},
                predictions_primary="test_best",
                epochs_actual=3, early_stopped=False, best_val=0.5,
            )
        finally:
            clear_active_log()
            set_quiet(False)

        events = _read_events(artifact.run_dir)
        rs = next(e for e in events if e["kind"] == "run_started")
        # 必带默认字段
        assert rs["run_id"] == "toy__smoke__seed0"
        assert rs["dataset"] == "my_data"
        assert rs["model"] == "my_net"
        assert rs["variant"] == "small"
        assert rs["seed"] == 0
        assert rs["trainer"] == "gradient"
        # 必带 run_info 字段 (保留)
        assert rs["params"] == 12345
        assert rs["epochs_max"] == 3
        assert rs["steps_per_epoch"] == 7
        assert rs["lr"] == 0.0005


# ---------------------------------------------------------------------------
# finalize idempotence
# ---------------------------------------------------------------------------

class TestFinalizeIdempotence:
    def test_finalize_called_twice_is_noop(self, tmp_path):
        """二次 finalize → no-op. 不重复 emit run_completed, 不二次 save_manifest."""
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)
            session = RunSession(artifact=artifact, ctx=ctx, trainer_name="gradient")
            session.start(run_info={"seed": ctx.seed})

            session.finalize(
                metrics={"best_val_mse": 0.1, "epochs_actual": 1},
                predictions_primary="test_best",
                epochs_actual=1, early_stopped=False, best_val=0.1,
            )
            # 第二次 — 应被吞掉
            session.finalize(
                metrics={"best_val_mse": 999.0, "epochs_actual": 999},
                predictions_primary="test",
                epochs_actual=999, early_stopped=True, best_val=999.0,
            )
        finally:
            clear_active_log()
            set_quiet(False)

        events = _read_events(artifact.run_dir)
        rc_events = [e for e in events if e["kind"] == "run_completed"]
        # 只 1 次 run_completed (第一次的; 第二次的 metrics/epochs 没生效)
        assert len(rc_events) == 1
        assert rc_events[0]["best_val"] == 0.1

        m = yaml.safe_load((artifact.run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        # 第一次 finalize 写的 metrics 在
        assert m["metrics"]["best_val_mse"] == 0.1


# ---------------------------------------------------------------------------
# failure path
# ---------------------------------------------------------------------------

class TestFailurePath:
    def test_fail_emits_run_failed_no_run_completed(self, tmp_path):
        """session.start() 后 trainer 抛错 → session.fail(exc):

        - emit run_failed (含 error_type / error / run_id)
        - 不 emit run_completed
        - manifest 落地 (start 时已落初始, fail() 再 save 含 env)
        - active log 清掉
        """
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)
            session = RunSession(artifact=artifact, ctx=ctx, trainer_name="gradient")
            session.start(run_info={"seed": ctx.seed})

            # 模拟训练抛错
            session.fail(ValueError("synthetic boom"))
        finally:
            clear_active_log()
            set_quiet(False)

        # events
        events = _read_events(artifact.run_dir)
        kinds = [e["kind"] for e in events]
        assert kinds.count("run_failed") == 1
        assert "run_completed" not in kinds, f"failure run 不能出 run_completed: {kinds}"

        rf = next(e for e in events if e["kind"] == "run_failed")
        assert rf["run_id"] == "toy__smoke__seed0"
        assert rf["error_type"] == "ValueError"
        assert rf["error"] == "synthetic boom"

        # manifest 落地 (含 env — start() 捕获)
        m = yaml.safe_load((artifact.run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        assert "rundown_version" in m["env"]

        # active log 清掉
        from rundown.events import active_run_id
        assert active_run_id() is None

    def test_fail_after_finalize_is_noop(self, tmp_path):
        """finalize 之后 fail() → no-op (L13: 不要「看似成功」地复活失败 run).

        run_failed 不应出现, run_completed 保留, manifest 不被覆盖.
        """
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)
            session = RunSession(artifact=artifact, ctx=ctx, trainer_name="gradient")
            session.start(run_info={"seed": ctx.seed})

            session.finalize(
                metrics={"best_val_mse": 0.2, "epochs_actual": 1},
                predictions_primary="test_best",
                epochs_actual=1, early_stopped=False, best_val=0.2,
            )

            # 二次 finalize 后再 fail — 都应 no-op
            session.fail(RuntimeError("late boom"))
        finally:
            clear_active_log()
            set_quiet(False)

        events = _read_events(artifact.run_dir)
        kinds = [e["kind"] for e in events]
        assert "run_failed" not in kinds
        assert kinds.count("run_completed") == 1


# ---------------------------------------------------------------------------
# context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_with_statement_calls_fail_on_exception(self, tmp_path):
        """with RunSession(...) as session: 块内抛错 → session.fail() 自动调.

        注意: 内层 ``pytest.raises`` 不能吞异常, 否则 outer __exit__ 收到的是
        (None, None, None) — __exit__ 只在异常真从块内抛出时调 fail().
        直接 raise 让 outer with 接到. 用 RuntimeError 而非 KeyError, 因为
        ``str(KeyError("kaboom"))`` 会带单引号 (e.g. ``"'kaboom'"``),
        影响断言. RuntimeError 直接返回原始字符串.
        """
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)
            with pytest.raises(RuntimeError):
                with RunSession(artifact, ctx, trainer_name="gradient") as session:
                    session.start(run_info={"seed": ctx.seed})
                    raise RuntimeError("kaboom")
        finally:
            clear_active_log()
            set_quiet(False)

        events = _read_events(artifact.run_dir)
        kinds = [e["kind"] for e in events]
        assert kinds.count("run_failed") == 1
        rf = next(e for e in events if e["kind"] == "run_failed")
        assert rf["error_type"] == "RuntimeError"
        assert rf["error"] == "kaboom"
        # __exit__ 必须返回 False → RuntimeError 真的从 outer with 抛出 (上面 pytest.raises 验证)

    def test_with_statement_normal_exit_does_not_emit_run_failed(self, tmp_path):
        """with 块正常退出 → 不 emit run_failed (留给 caller 自己 finalize)."""
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)
            with RunSession(artifact, ctx, trainer_name="gradient") as session:
                session.start(run_info={"seed": ctx.seed})
                # 显式 finalize
                session.finalize(
                    metrics={"best_val_mse": 0.3, "epochs_actual": 1},
                    predictions_primary="test_best",
                    epochs_actual=1, early_stopped=False, best_val=0.3,
                )
        finally:
            clear_active_log()
            set_quiet(False)

        events = _read_events(artifact.run_dir)
        kinds = [e["kind"] for e in events]
        assert "run_failed" not in kinds
        assert kinds.count("run_completed") == 1


# ---------------------------------------------------------------------------
# env capture timing — start 捕获, finalize 写入
# ---------------------------------------------------------------------------

class TestEnvCaptureTiming:
    def test_env_written_at_finalize_not_at_start(self, tmp_path):
        """env 在 start() 时捕获, finalize() 才写盘到 manifest.env.

        锁住 L3 风险: 与现行 engine 行为对齐 (env 在 fit 末尾写).
        """
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)
            session = RunSession(artifact=artifact, ctx=ctx, trainer_name="gradient")
            session.start(run_info={"seed": ctx.seed})

            # start() 之后立刻读 manifest — 此时 env 应尚未被覆写
            # (save_manifest 写的是 start 时刻空 env, 不是 captured_env)
            m_after_start = yaml.safe_load(
                (artifact.run_dir / "manifest.yaml").read_text(encoding="utf-8")
            )
            # start 时 save_manifest 落的是初始 manifest, env 默认空 dict
            # (RunManifest.env = field(default_factory=dict)).
            assert m_after_start["env"] == {}, (
                f"start 后 env 应为空 (captured_env 在内存, finalize 才写), "
                f"got {m_after_start['env']}"
            )

            session.finalize(
                metrics={"best_val_mse": 0.4, "epochs_actual": 1},
                predictions_primary="test_best",
                epochs_actual=1, early_stopped=False, best_val=0.4,
            )
        finally:
            clear_active_log()
            set_quiet(False)

        # finalize 之后 — env 字段齐
        m_final = yaml.safe_load(
            (artifact.run_dir / "manifest.yaml").read_text(encoding="utf-8")
        )
        assert "rundown_version" in m_final["env"]
        assert "python" in m_final["env"]


# ---------------------------------------------------------------------------
# manifest config preservation
# ---------------------------------------------------------------------------

class TestPrimaryMetricOverride:
    def test_primary_metric_in_config_preserved_if_pre_set(self, tmp_path):
        """trainer / caller 在 finalize 前已写 manifest.config.primary_metric
        → session.finalize 不强行覆盖 (让 trainer 决定 priority).
        """
        set_quiet(False)
        try:
            artifact, ctx = _make_run(tmp_path)
            # 模拟 trainer 在 finalize 前先写 primary_metric
            artifact.manifest.config["primary_metric"] = "custom_loss_v3"

            session = RunSession(artifact=artifact, ctx=ctx,
                                 trainer_name="custom", primary_metric="best_val_mse")
            session.start(run_info={"seed": ctx.seed})
            session.finalize(
                metrics={"custom_loss_v3": 0.5, "epochs_actual": 1},
                predictions_primary="test_best",
                epochs_actual=1, early_stopped=False, best_val=0.5,
            )
        finally:
            clear_active_log()
            set_quiet(False)

        m = yaml.safe_load((artifact.run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        # 已存在的 primary_metric → 保留 (不覆盖)
        assert m["config"]["primary_metric"] == "custom_loss_v3"
        # predictions_primary 仍按 finalize 参数写
        assert m["config"]["predictions_primary"] == "test_best"