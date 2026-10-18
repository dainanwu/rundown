"""StagedTrainer — 分阶段训练模板 (Stage 1 全训练 / Stage 2 部分冻结).

场景: 预训练 → 微调, 或 LR schedule 切换, 或某一阶段冻结 backbone. 常见:
  * Vision/NLP: backbone 训完后, freeze backbone 只训 head;
  * Time-series: SequenceLSTM 训 1 个 stage 后, freeze LSTM 只优化 head.

模板示例配置::

    stages:
      - epochs: 1
        lr: 0.001
        freeze: false        # Stage 1: 训全部
      - epochs: 1
        lr: 0.0005
        freeze: true         # Stage 2: 冻住 LSTM, 训 head / 其它

实现要点:
  * 冻结通过 ``param.requires_grad = False`` + 过滤 ``optimizer`` 参数;
  * 跨 stage 重新构造 optimizer (新 LR / 新参数集) — PyTorch 惯例.
  * curves 写 ``stage1_train_loss`` / ``stage2_train_loss`` /
    ``stage_<i>_train_loss`` + ``val_mse`` (跨 stage 累积).
  * events 写 ``stage_started`` / ``stage_completed``.
  * primary_metric = ``best_val_mse``, 跨 stage 选 best.

前提: 模型有 ``.lstm`` 属性 (SequenceLSTM 模式). 其它结构请改 trainer.py
或者改 ``stages[i].freeze_module = "encoder"`` 这类自定义. 当前模板对
``SequenceLSTM`` 是 hard-coded — README 里说明.

参考: examples/custom_trainers/README.md.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from . import common as C


def _freeze_module(model: nn.Module, attr: str) -> List[nn.Parameter]:
    """冻结 ``model.<attr>``, 返回需要优化的剩余参数列表.

    异常:
      * ``model.<attr>`` 不存在 → ValueError (例是 SequenceLSTM-specific).
      * 否则 ``<attr>`` 所有参数 ``requires_grad=False``.
    """
    if not hasattr(model, attr):
        raise ValueError(
            f"staged trainer: model has no attribute {attr!r}; "
            f"this template assumes SequenceLSTM-style architecture "
            f"(.lstm / .head). Adapt the trainer or change stages[*].freeze."
        )
    module = getattr(model, attr)
    frozen_params = []
    for p in module.parameters():
        p.requires_grad_(False)
        frozen_params.append(p)
    # 剩余 trainable 参数
    trainable = [p for p in model.parameters() if p.requires_grad]
    return trainable


class StagedTrainer:
    """Staged trainer — no-arg 构造, 配置走 ``ctx.trainer_config['stages']``.

    协议字段:
      * ``name = "staged"``
      * ``required_provider_keys = ("x", "y", "warm_start")``
      * ``primary_metric = "best_val_mse"``

    默认 stages (当 ``ctx.trainer_config`` 没写时):
        [
          {"epochs": 1, "lr": 0.001, "freeze": false},
          {"epochs": 1, "lr": 0.0005, "freeze": true},
        ]
    """

    name = "staged"
    required_provider_keys = ("x", "y", "warm_start")
    primary_metric = "best_val_mse"

    DEFAULT_STAGES: List[Dict[str, Any]] = [
        {"epochs": 1, "lr": 0.001, "freeze": False},
        {"epochs": 1, "lr": 0.0005, "freeze": True},
    ]

    def __init__(self) -> None:
        pass

    # ---- Trainer protocol ----------------------------------------------

    def validate(self, providers: Dict[str, Any]) -> None:
        """No-op. 不消耗 batch."""
        return None

    def fit(
        self,
        *,
        model: nn.Module,
        providers: Dict[str, Callable[[str], Dict[str, torch.Tensor]]],
        ctx: Any,
        session: Any,
    ) -> Any:
        from rundown.trainers import TrainResult

        # ---- 解析 stages ----
        cfg = dict(ctx.trainer_config or {})
        raw_stages = cfg.get("stages", self.DEFAULT_STAGES)
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError(
                f"staged trainer: trainer_config.stages must be non-empty list, "
                f"got {raw_stages!r}"
            )

        stages: List[Dict[str, Any]] = []
        for i, s in enumerate(raw_stages):
            if not isinstance(s, dict):
                raise ValueError(
                    f"staged trainer: stages[{i}] must be dict, got {type(s).__name__}"
                )
            stages.append({
                "epochs":  int(s.get("epochs", 1)),
                "lr":      float(s.get("lr", ctx.lr)),
                "freeze":  bool(s.get("freeze", False)),
                "module":  s.get("module", "lstm"),  # frozen module attr 名
            })

        C.seed_everything(int(ctx.seed))

        train_p = providers["train"]
        val_p = providers["val"]
        grad_clip = float(getattr(ctx, "grad_clip_max_norm", 1.0) or 1.0)

        # ---- 跨 stage 累积 ----
        val_mses_all: List[float] = []
        best_val_mse = float("inf")
        best_epoch = 0
        epochs_actual = 0
        stage_train_curves: Dict[int, List[float]] = {}
        stage_epochs_completed: Dict[int, int] = {}

        # Strategy: 每个 stage 独立循环; optimizer 在 stage 边界重建
        # (新 LR / 新参数集). 用 param.requires_grad 控制冻结.
        for stage_idx, stage in enumerate(stages, start=1):
            stage_train_losses: List[float] = []
            lr = stage["lr"]
            freeze = stage["freeze"]
            freeze_attr = stage["module"]

            session.emit(
                "stage_started",
                run_id=ctx.run_id, stage=stage_idx,
                epochs=stage["epochs"], lr=lr, freeze=freeze,
                freeze_module=freeze_attr,
            )

            # 冻结: 先全部 requires_grad=True, 然后 freeze module.
            for p in model.parameters():
                p.requires_grad_(True)
            if freeze:
                trainable_params = _freeze_module(model, freeze_attr)
            else:
                trainable_params = list(model.parameters())

            if not trainable_params:
                # freeze=True 且所有参数都被 freeze → 没人训, 报错让 user 知道
                raise ValueError(
                    f"staged trainer: stage {stage_idx} (freeze={freeze}) "
                    f"has zero trainable parameters; check stage.module or model."
                )

            optimizer = torch.optim.Adam(trainable_params, lr=lr)

            for epoch in range(1, stage["epochs"] + 1):
                epochs_actual += 1
                global_epoch = epochs_actual
                model.train()
                losses_in_epoch = []

                for _step in range(int(ctx.steps_per_epoch)):
                    batch = train_p("train")
                    yhat = C.forward(model, batch)
                    target = C.target_for(batch, yhat)
                    loss = ((yhat - target) ** 2).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    raw_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=float("inf")
                        ).item()
                    )
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=grad_clip
                        )
                    optimizer.step()
                    losses_in_epoch.append(float(loss.item()))
                    session.emit(
                        "step_completed",
                        run_id=ctx.run_id, stage=stage_idx,
                        epoch=global_epoch, step=_step + 1,
                        train_loss=float(loss.item()),
                        grad_norm=raw_grad_norm, lr=lr,
                    )

                ep_loss = sum(losses_in_epoch) / len(losses_in_epoch)
                stage_train_losses.append(ep_loss)
                val_mse = C.evaluate_mse(model, val_p)
                val_mses_all.append(val_mse)

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_epoch = global_epoch
                    session.checkpoint_best(C.state_cpu(model))

                session.emit(
                    "epoch_completed",
                    run_id=ctx.run_id, stage=stage_idx,
                    epoch=global_epoch,
                    train_loss=ep_loss, val_mse=val_mse, lr=lr,
                    best_val_mse=best_val_mse, best_epoch=best_epoch,
                )

            stage_train_curves[stage_idx] = stage_train_losses
            stage_epochs_completed[stage_idx] = len(stage_train_losses)

            session.emit(
                "stage_completed",
                run_id=ctx.run_id, stage=stage_idx,
                epochs_completed=len(stage_train_losses),
                best_val_mse=best_val_mse,
                best_epoch=best_epoch,
            )

        # ---- 写产物 ----
        # final weights (Stage 2 结束后模型当前 state)
        session.write_weights(C.state_cpu(model), best=False)

        # curves: per-stage train_loss + 跨 stage val_mse.
        curves: Dict[str, torch.Tensor] = {}
        for stage_idx, losses in stage_train_curves.items():
            curves[f"stage{stage_idx}_train_loss"] = torch.tensor(
                losses, dtype=torch.float32,
            )
        curves["val_mse"] = torch.tensor(val_mses_all, dtype=torch.float32)
        session.write_curves(curves)

        # test predictions — best weights
        C.write_test_predictions(session, model, providers)

        # ---- finalize ----
        metrics: Dict[str, Any] = {
            "best_val_mse":   best_val_mse,
            "epochs_to_best": best_epoch,
            "epochs_actual":  epochs_actual,
            "stages_completed": len(stages),
        }
        # 把每个 stage 实际跑的 epochs 也写进 metrics (便于 experiment 汇总看)
        for stage_idx, n in stage_epochs_completed.items():
            metrics[f"stage{stage_idx}_epochs"] = n

        session.finalize(
            metrics=metrics,
            predictions_primary="test_best",
            epochs_actual=epochs_actual,
            early_stopped=False,
            best_val=best_val_mse,
        )

        return TrainResult(
            metrics=metrics,
            predictions_primary="test_best",
            epochs_actual=epochs_actual,
            early_stopped=False,
        )
