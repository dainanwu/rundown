"""训练引擎. MVP 阶段只有 GradientStrategy.

GradientStrategy 走 BatchProvider 契约: train_data / val_data / test_data
必须是 callable ``f(split: str) -> {"x": Tensor[B,T,F], "y": Tensor[B,T], "warm_start": int}``.
Engine 不再支持合成数据 fallback (有 #2 修复). 真实数据接入是 library
user 或 CLI registry 的责任.

Phase 2 会加:
  - AnalyticStrategy: 无可训练参数 (对应 PST 那个 98 行 Persistence 分叉).
  - FrozenStrategy:   载入预训练权重, 可选部分解冻.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import numpy as np
import torch
import torch.nn as nn

from .artifact import RunArtifact
from ._env import capture_env
from .defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS_MAX,
    DEFAULT_GRAD_NORM_THRESHOLD,
    DEFAULT_LR,
    DEFAULT_STEP_EVENT_EVERY_N,
    DEFAULT_STEPS_PER_EPOCH,
)
from .events import emit


@dataclass
class EngineConfig:
    run_id: str
    seed: int
    epochs_max: int = DEFAULT_EPOCHS_MAX
    batch_size: int = DEFAULT_BATCH_SIZE
    steps_per_epoch: int = DEFAULT_STEPS_PER_EPOCH
    grad_clip_max_norm: Optional[float] = 1.0
    lr: float = DEFAULT_LR
    # Per-step event throttling: write ``step_completed`` every N steps
    # (default 10). The last step of every epoch is always written so
    # reader can pair step_completed with epoch_completed.
    # ``gradient_norm_exceeded`` is emitted the moment a raw grad norm
    # crosses ``grad_norm_threshold`` (default 100.0) — independent of
    # step_event_every_n, so a single spike is never lost.
    step_event_every_n: int = DEFAULT_STEP_EVENT_EVERY_N
    grad_norm_threshold: float = DEFAULT_GRAD_NORM_THRESHOLD
    # code snapshot — 默认开; CLI --no-snapshot 关闭. 跟 rundown 哲学对齐:
    # 产物比代码活得久 → 跑一次就把代码 snapshot 进 artifact.
    snapshot: bool = True
    # experiment sweep 共享 snapshot 池场景: caller 已建好, 直接引用.
    # dict 形式 (snapshot_code() 返回值). None → 自己建 (单次 train 走这条).
    external_snapshot: Optional[Dict[str, Any]] = None
    # snapshot 落盘目录 (默认 None → fit() 用 artifact.run_dir).
    # 普通 train CLI 设为 out_root/_shared, 让多次 train 共享 1 份 (artifact 边界是 out_root).
    # --run-dir 模式: 留 None, snapshot 落 run_dir (self-contained).
    snapshot_target_dir: Optional[Path] = None
    # archive_path 相对 run_dir 的前缀 (e.g. "../_shared"). snapshot 落 _shared/
    # 时, manifest.env.code_snapshot.archive_path 应该是 "../_shared/code_xxx.zip"
    # 而不是 "code_xxx.zip" (因为 snapshot 不在 run_dir 里).
    snapshot_archive_rel: str = ""


class Strategy(Protocol):
    """策略接口. 任何训练/推理逻辑都通过 fit() 走.

    train_data / val_data / test_data 可以是 None (synthetic fallback) 或一个
    BatchProvider callable ``f(split: str) -> {"x": Tensor[B,T,F], "y": Tensor[B,T], "warm_start": int}``.
    """
    def fit(
        self,
        model: nn.Module,
        train_data: Any,
        val_data: Any,
        ctx: EngineConfig,
        artifact: RunArtifact,
        test_data: Any = None,
    ) -> Dict[str, Any]: ...


def _set_seed(seed: int) -> None:
    """Seed PyTorch + numpy + cuDNN for reproducible runs.

    Without these, LSTM val_mse drifts ~5% across runs of the same seed
    (verified empirically: epoch 6 best_val=0.87 vs 0.93 across two runs
    with identical CLI args). The drift is non-trivial for paper-quality
    claims like "best_val_mse = X" — readers expect to be able to reproduce.
    """
    s = int(seed)
    torch.manual_seed(s)
    np.random.seed(s)
    # Single-thread: reduces intra-op float accumulation order differences
    # (a known source of CPU non-determinism in LSTM forward).
    torch.set_num_threads(1)
    # cuDNN: pick deterministic algo, disable benchmark autotuner.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Some ops don't have deterministic kernels; warn instead of crash.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_batch(data: Any, *, split: str) -> Dict[str, torch.Tensor]:
    """BatchProvider 契约:
        ``f(split: str) -> {"x": Tensor[B,T,F], "y": Tensor[B,T], "warm_start": int}``

    Engine 现在不再支持合成数据 fallback — BatchProvider 是必需的.
    任何训练 / 验证 / 测试数据来源都由调用方提供 (CLI registry / library user).
    """
    if not callable(data):
        raise TypeError(
            f"BatchProvider required for split={split!r}; "
            f"got {type(data).__name__}. Engine no longer supports "
            "synthetic-data fallback. Pass a callable "
            "f(split) -> {'x': Tensor[B,T,F], 'y': Tensor[B,T], 'warm_start': int}."
        )
    return data(split)


def _forward(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """兼容 forward(x, warm_start=) 签名. Phase 2 会按 dataset schema 适配."""
    yhat = model(batch["x"], warm_start=int(batch["warm_start"]))
    # 模型输出可能是 [B, T, 1] 或 [B, T]; 统一 squeeze 最后一维
    if yhat.dim() == 3 and yhat.shape[-1] == 1:
        yhat = yhat.squeeze(-1)
    return yhat


class GradientStrategy:
    """常规监督训练. Adam + 全局梯度裁剪 + 早停.

    这是 MVP 唯一实现的 strategy. AnalyticStrategy / FrozenStrategy Phase 2 写.
    """

    def __init__(self, early_stop_patience: int = 10) -> None:
        self.early_stop_patience = early_stop_patience

    def fit(
        self,
        model: nn.Module,
        train_data: Any,
        val_data: Any,
        ctx: EngineConfig,
        artifact: RunArtifact,
        test_data: Any = None,
    ) -> Dict[str, Any]:
        _set_seed(int(ctx.seed))

        # run_started: 一次 run 摘要事件, 让 events.jsonl 自描述.
        # agent 读 events.jsonl 第一行就知道这是哪个 dataset/model 的 run.
        n_params = sum(p.numel() for p in model.parameters())
        emit("run_started",
             run_id=ctx.run_id,
             dataset=artifact.manifest.dataset.get("alias"),
             model=artifact.manifest.model.get("alias"),
             variant=artifact.manifest.model.get("variant"),
             params=n_params,
             epochs_max=ctx.epochs_max,
             steps_per_epoch=ctx.steps_per_epoch,
             lr=ctx.lr,
             seed=ctx.seed)
        emit("strategy_started", run_id=ctx.run_id, strategy="gradient", epochs_max=ctx.epochs_max)

        optimizer = torch.optim.Adam(model.parameters(), lr=ctx.lr)
        loss_fn = nn.MSELoss()

        train_losses, val_mses = [], []
        best_val = float("inf")
        best_epoch = 0
        epochs_actual = 0
        early_stopped = False

        for epoch in range(1, ctx.epochs_max + 1):
            epochs_actual = epoch
            model.train()

            # ---- train steps (steps_per_epoch 决定一个 epoch 跑多少 batch) ----
            losses_in_epoch = []
            for _step in range(ctx.steps_per_epoch):
                train_batch = _resolve_batch(train_data, split="train")
                warm = int(train_batch["warm_start"])
                yhat = _forward(model, train_batch)                       # [B, T']
                target = train_batch["y"][:, warm:warm + yhat.shape[1]]   # [B, T']
                loss = loss_fn(yhat, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Raw grad norm BEFORE clipping (max_norm=inf = no clip).
                # This is the value the ``gradient_norm_exceeded`` event uses;
                # the actual clip is applied a few lines below.
                raw_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).item()
                )
                if raw_grad_norm > ctx.grad_norm_threshold:
                    emit("gradient_norm_exceeded",
                         run_id=ctx.run_id, epoch=epoch, step=_step + 1,
                         grad_norm=raw_grad_norm, threshold=ctx.grad_norm_threshold)
                if ctx.grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ctx.grad_clip_max_norm)
                optimizer.step()
                losses_in_epoch.append(float(loss.item()))
                # Per-step event (throttled by step_event_every_n, but the
                # last step of every epoch is always written so reader can
                # pair step_completed with the surrounding epoch_completed).
                if ((_step + 1) % ctx.step_event_every_n == 0
                        or _step == ctx.steps_per_epoch - 1):
                    emit("step_completed",
                         run_id=ctx.run_id, epoch=epoch, step=_step + 1,
                         train_loss=float(loss.item()),
                         grad_norm=raw_grad_norm, lr=ctx.lr)
            # epoch 平均损失 (覆盖所有 steps_per_epoch 个 batch, 不只是最后一个)
            train_losses.append(sum(losses_in_epoch) / len(losses_in_epoch))

            # ---- val step ----
            model.eval()
            with torch.no_grad():
                val_batch = _resolve_batch(val_data, split="val")
                v = _forward(model, val_batch)
                v_warm = int(val_batch["warm_start"])
                val_target = val_batch["y"][:, v_warm:v_warm + v.shape[1]]
                val_mse = float(loss_fn(v, val_target).item())
            val_mses.append(val_mse)

            if val_mse < best_val:
                best_val = val_mse
                best_epoch = epoch
                artifact.write_weights(model.state_dict(), best=True)
                patience = 0
            else:
                patience = (epoch - best_epoch)
                if patience >= self.early_stop_patience:
                    early_stopped = True
                    emit("early_stopped", run_id=ctx.run_id, epoch=epoch, best_epoch=best_epoch, best_val=best_val)
                    break

            emit("epoch_completed", run_id=ctx.run_id, epoch=epoch,
                 train_loss=train_losses[-1], val_mse=val_mse, lr=ctx.lr,
                 best_val=best_val, best_epoch=best_epoch)

        # ---- 写最终产物 ----
        # 保存 final state (CPU contiguous, 不依赖 model 当前 device)
        final_state = {k: v.detach().cpu().contiguous().clone()
                       for k, v in model.state_dict().items()}
        artifact.write_weights(final_state, best=False)
        artifact.write_curves({
            "train_loss": torch.tensor(train_losses, dtype=torch.float32),
            "val_mse":    torch.tensor(val_mses,    dtype=torch.float32),
        })

        # ---- 写测试预测: 双产物 (test_best + test) ----
        # 关键修复: manifest.metrics.best_val_mse 来自 weights_best,
        # 所以 primary predictions 必须用 weights_best 算, 否则指标和
        # 产物来自不同 epoch 的 weights.
        # 同时保留 final predictions 作为「过拟合后漂到哪」的对比.
        # 同一组 test_batch 同时给两个权重用, 保证 y_true 完全一致,
        # y_hat 差异直接反映 best vs final 的预测分歧.
        with torch.no_grad():
            test_batch = _resolve_batch(test_data, split="test")
            t_warm = int(test_batch["warm_start"])

        # 1) test_best: 用 best weights (manifest.metrics 对应的权重)
        # 注: weights_best 在 artifact 里是 CPU tensor; model 假设在 CPU.
        # 如果用户在 GPU 跑, 需自己 model.to(device) 后 load_state_dict;
        # 不在 engine 处理 (避免静默 device 转换, 不撞「拒绝 3: 不做隐式回落」).
        model.load_state_dict(artifact.weights_best)
        with torch.no_grad():
            test_yhat_best = _forward(model, test_batch)
        artifact.write_predictions("test_best", {
            "y_hat":  test_yhat_best.detach().cpu().contiguous(),
            "y_true": test_batch["y"][:, t_warm:t_warm + test_yhat_best.shape[1]].cpu().contiguous(),
        })

        # 2) test: 用 final weights (过拟合后的预测, 用于对比诊断)
        model.load_state_dict(final_state)
        with torch.no_grad():
            test_yhat_final = _forward(model, test_batch)
        artifact.write_predictions("test", {
            "y_hat":  test_yhat_final.detach().cpu().contiguous(),
            "y_true": test_batch["y"][:, t_warm:t_warm + test_yhat_final.shape[1]].cpu().contiguous(),
        })

        artifact.manifest.metrics = {
            "best_val_mse": best_val,
            "epochs_to_best": best_epoch,
            "epochs_actual": epochs_actual,
        }
        # reader (read-run / analyzer) 应该读哪个 predictions?
        # 标记 primary 让 artifact 自描述.
        artifact.manifest.config["predictions_primary"] = "test_best"
        # Record step-event granularity so reader can parse events.jsonl
        # without guessing.
        artifact.manifest.config["step_event_every_n"] = ctx.step_event_every_n
        artifact.manifest.config["grad_norm_threshold"] = ctx.grad_norm_threshold
        # autolog env — git commit / python / platform / hostname / code snapshot
        # 让 artifact 是 self-describing ground truth.
        # code snapshot 默认开: 跟 rundown 哲学对齐 — 产物比代码活得久.
        # snapshot_target_dir=None → snapshot 落 run_dir (self-contained 模式, --run-dir)
        # snapshot_target_dir=out_root/_shared → 普通 train CLI 默认, 多次 train 共享 1 份
        # external_snapshot 不为空 → sweep subprocess 直接引用, 跳过自建.
        artifact.manifest.env = capture_env(
            snapshot_dir=ctx.snapshot_target_dir or artifact.run_dir,
            snapshot=ctx.snapshot,
            external_snapshot=ctx.external_snapshot,
            archive_rel=ctx.snapshot_archive_rel,
        )
        artifact.save_manifest()
        emit("run_completed", run_id=ctx.run_id,
             epochs_actual=epochs_actual, best_val=best_val, early_stopped=early_stopped)
        return {
            "epochs_actual": epochs_actual,
            "best_val": best_val,
            "early_stopped": early_stopped,
        }


__all__ = ["EngineConfig", "Strategy", "GradientStrategy"]
