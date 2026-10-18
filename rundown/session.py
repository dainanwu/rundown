"""Run lifecycle — 把 run 生命周期从 GradientStrategy 抽出.

职责: 一次 run 的「开始 / 写产物 / 收尾」统一收口, 不参与训练语义.
训练循环 / loss / optimizer 选择仍由 trainer 决定.

生命周期 (4 步):
    start(run_info)   # 初始 manifest + 激活 events + 捕获 env + emit run_started
    during            # emit / write_weights / write_curves / write_predictions
    finalize(metrics) # 写 manifest.metrics / primary_metric / env + 保存 + emit run_completed
    fail(exc)         # emit run_failed + 保存 manifest (含 captured env) + clear_active_log

硬规则:
    - finalize 幂等; 第二次调用 no-op.
    - fail 在 finalize 之后 → no-op (不要「看似成功」地复活, L13).
    - 不吞异常; context manager 的 __exit__ 返回 False (让 caller 决定 reraise).
    - 不调 set_quiet; quiet 由 CLI 控制.
    - 不支持同进程内嵌套/并发 session: events 是模块全局 (_ACTIVE_DIR /
      _ACTIVE_RUN_ID), 多 session 会串. 需要并发 → 多进程.
    - session 只 import artifact / _env / events; 不 import engine /
      trainers / registry / cli.

调用方契约: start 之后不要手动调 set_active_log / clear_active_log /
artifact.save_manifest — session 拥有. 否则会破坏单例生命周期.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from .artifact import RunArtifact
from ._env import capture_env
from .events import clear_active_log, emit, set_active_log


class RunSession:
    """一次 run 的生命周期 owner. 不参与训练语义.

    trainer_name 字段 (默认 "gradient") 让 emit 出来的 run_started 自描述
    这次走的是哪条训练路径 — 后续 C2 的 trainer 注册表引入自定义 trainer 时,
    trainer_name 由 CLI/registry 注入; GradientStrategy 在 C1 默认仍写 "gradient".
    """

    def __init__(
        self,
        artifact: RunArtifact,
        ctx: Any,  # EngineConfig — 不强 import, 避免循环依赖
        *,
        trainer_name: str = "gradient",
        primary_metric: str = "best_val_mse",
    ) -> None:
        self.artifact = artifact
        self.ctx = ctx
        self.trainer_name = trainer_name
        self.primary_metric = primary_metric
        self._started: bool = False
        self._finalized: bool = False
        self._failed: bool = False
        # env 在 start() 时捕获; finalize() / fail() 才写进 manifest.env.
        # 这样 env.code_snapshot 物理落盘时机保持一致 (与现行实现对齐).
        self._captured_env: Optional[Dict[str, Any]] = None

    @property
    def finalized(self) -> bool:
        """session 是否已 finalize. C2 给 CLI 用: trainer.fit() 返
        回后检查 ``session.finalized``; 若 trainer 自己没调 finalize
        (违反契约), 抛 RuntimeError, 让 CLI 报错而不是写一个看似成功
        但 ``run_completed`` 缺失的 run_dir.

        ``fail()`` 之后也是 no-finalize (L13) — 这时 ``finalized`` 仍
        为 False, 但 ``_failed`` 为 True. 调用方通常用 ``session.finalized``
        区分"成功 finalize" vs "没成功" 两种情况.
        """
        return self._finalized

    # ---- 公开 API ---------------------------------------------------------

    def start(self, run_info: Optional[Dict[str, Any]] = None) -> None:
        """启动一次 run.

        顺序 (重要, 与现行实现对齐):
            1. 保存初始 manifest — 写盘即合同, 跑崩了也有空壳 manifest.
               C2: setdefault trainer / primary_metric 到 manifest.config
               (caller 已 set → 保留, 不覆盖). C1 测试依赖 preservation.
            2. set_active_log(run_dir, run_id) — 激活 events 流; 这一步会
               unlink 残留 events.jsonl, 让重激活 (中断重跑 / agent 重试)
               从干净流开始. 不依赖 caller 是否预先激活.
            3. 捕获 env + code snapshot 到 self._captured_env (暂不写盘到 manifest).
            4. emit run_started — run_id / dataset / model / variant / seed /
               trainer (self.trainer_name) + run_info 的所有字段.
        """
        if self._started:
            # 二次 start → no-op. session 设计为单次 run 单生命周期, 调用方
            # 不要重复触发 (CLI 不该触发; C2 的 GradientTrainer 适配器复用
            # 时也只在第一次调用). 防御性 no-op 比抛错更稳.
            return

        run_info = dict(run_info) if run_info else {}
        artifact = self.artifact

        # 0) C2 — manifest.config 写 trainer / primary_metric. caller
        #    在 start 前可能已经 (trainer 自己或 CLI 写完 train 字段再
        #    调 start) → setdefault 不覆盖, 保留 caller 决定. C1 测试
        #    不依赖这两个字段, 不破坏.
        artifact.manifest.config.setdefault("trainer", self.trainer_name)
        artifact.manifest.config.setdefault("primary_metric", self.primary_metric)

        # 1) 初始 manifest 落盘
        artifact.save_manifest()

        # 2) 激活 events 流
        set_active_log(artifact.run_dir, artifact.manifest.run_id)

        # 3) 捕获 env / code snapshot (snapshot 落盘时机与现行对齐: 单次
        #    train → artifact.run_dir 或 out_root/_shared, 取决于 ctx).
        ctx = self.ctx
        snapshot_dir = getattr(ctx, "snapshot_target_dir", None) or artifact.run_dir
        self._captured_env = capture_env(
            snapshot_dir=snapshot_dir,
            snapshot=getattr(ctx, "snapshot", True),
            external_snapshot=getattr(ctx, "external_snapshot", None),
            archive_rel=getattr(ctx, "snapshot_archive_rel", ""),
        )

        # 4) emit run_started.
        # 默认字段从 manifest / run_info 推导; run_info 提供额外字段
        # (params / epochs_max / steps_per_epoch / lr / ...). 后合并时
        # run_info 覆盖默认 — 与 README「kwargs 合并: 默认 < run_info」一致.
        manifest = artifact.manifest
        default_fields: Dict[str, Any] = {
            "run_id": manifest.run_id,
            "dataset": run_info.pop("dataset", manifest.dataset.get("alias")),
            "model": run_info.pop("model", manifest.model.get("alias")),
            "variant": run_info.pop("variant", manifest.model.get("variant")),
            "seed": run_info.pop("seed", manifest.seed),
            "trainer": self.trainer_name,
        }
        # 保留 run_info 全部键 (含 params / epochs_max / steps_per_epoch / lr
        # 等等). 顺序: 默认字段在前, run_info 在后, 后者覆盖前者.
        fields: Dict[str, Any] = {}
        fields.update(default_fields)
        fields.update(run_info)
        emit("run_started", **fields)

        self._started = True

    def emit(self, kind: str, **fields: Any) -> None:
        """训练循环里的事件都走这里; 委托给 rundown.events.emit."""
        emit(kind, **fields)

    def checkpoint_best(self, state: Dict[str, Any]) -> Path:
        """val_mse 改善时落 weights_best. 委托给 RunArtifact.write_weights(best=True)."""
        return self.artifact.write_weights(state, best=True)

    def write_weights(self, state: Dict[str, Any], *, best: bool = False) -> Path:
        """最终权重 / best 权重都走这里. best=True 与 checkpoint_best 等价."""
        return self.artifact.write_weights(state, best=best)

    def write_curves(self, curves: Dict[str, Any]) -> Path:
        """写 curves.safetensors."""
        return self.artifact.write_curves(curves)

    def write_predictions(self, split: str, tensors: Dict[str, Any]) -> Path:
        """写 predictions/<split>.safetensors (test / test_best / val / train)."""
        return self.artifact.write_predictions(split, tensors)

    def finalize(
        self,
        metrics: Dict[str, Any],
        *,
        predictions_primary: str = "test_best",
        epochs_actual: Optional[int] = None,
        early_stopped: bool = False,
        best_val: Optional[float] = None,
    ) -> None:
        """成功收尾. 幂等 — 第二次 no-op.

        顺序:
            1. 写 manifest.metrics (覆盖 trainer 计算结果)
            2. 写 manifest.config.primary_metric (若 trainer / caller 没显式
               设置; 默认 best_val_mse). 已存在 → 保留.
            3. 写 manifest.config.predictions_primary = predictions_primary
            4. 写 manifest.env = self._captured_env (start 时捕获的)
            5. save_manifest 落盘
            6. emit run_completed (run_id / epochs_actual / best_val /
               early_stopped / metrics — 全部保留, 不丢旧 key)
            7. clear_active_log
        """
        if self._finalized:
            return  # 幂等
        if self._failed:
            # fail 后再调 finalize → no-op. L13: 不要「看似成功」地复活一个
            # 已经标记为 failed 的 run.
            return

        artifact = self.artifact
        manifest = artifact.manifest

        # 1) metrics
        manifest.metrics = dict(metrics) if metrics else {}

        # 2) primary_metric — 已有 (trainer 在 fit 末尾或 caller 已设) → 保留
        if "primary_metric" not in manifest.config:
            manifest.config["primary_metric"] = self.primary_metric

        # 3) predictions_primary
        manifest.config["predictions_primary"] = predictions_primary

        # 4) env — start 时捕获的 (与现行实现对齐, L3 风险已锁定)
        if self._captured_env is not None:
            manifest.env = self._captured_env

        # 5) 落盘
        artifact.save_manifest()

        # 6) run_completed — 保留旧 payload keys (run_id / epochs_actual /
        #    best_val / early_stopped); 增加 metrics 方便 reader 直接读.
        rc_fields: Dict[str, Any] = {
            "run_id": manifest.run_id,
            "epochs_actual": epochs_actual,
            "best_val": best_val,
            "early_stopped": early_stopped,
            "metrics": dict(metrics) if metrics else {},
        }
        emit("run_completed", **rc_fields)

        # 7) 清掉全局指针
        clear_active_log()

        self._finalized = True

    def fail(self, exc: BaseException) -> None:
        """失败收尾. finalize 之后 → no-op (L13).

        顺序:
            1. emit run_failed (run_id / error_type / error). 不 emit run_completed.
            2. 若 start 时已捕获 env → 写进 manifest.env 并落盘 (让失败 run
               也保留 env / code snapshot 痕迹).
            3. clear_active_log
        """
        if self._finalized or self._failed:
            return  # 幂等

        manifest = self.artifact.manifest
        emit("run_failed",
             run_id=manifest.run_id,
             error_type=type(exc).__name__,
             error=str(exc))

        # 落 env (若有). start 时未触发 → 跳过 (例如 start 前就抛).
        if self._captured_env is not None:
            manifest.env = self._captured_env
            try:
                self.artifact.save_manifest()
            except OSError:
                # 仅 IO 写盘失败时 best-effort: 不能让它掩盖原始训练异常.
                # 非 IO 错误 (schema / 序列化) 继续抛出, 不静默吞.
                pass

        clear_active_log()
        self._failed = True

    # ---- Context manager (可选, 不强制) ---------------------------------

    def __enter__(self) -> "RunSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # 返回 False — 不吞异常, 让 caller 决定 reraise.
        if exc is not None:
            self.fail(exc)
        return False


__all__ = ["RunSession"]