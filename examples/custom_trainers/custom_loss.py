"""CustomLossTrainer — 自定义 loss 模板 (Huber + L1 magnitude penalty).

场景: 不想只用 MSE. 例如 ——
  * 用 HuberLoss (delta 控制 outlier 敏感度) 替 MSE 提升鲁棒性;
  * 加 ``l1_weight * |yhat|.mean()`` 鼓励预测趋近 0 (sparse / conservative
    forecasting 常见手法);
  * ``primary_metric`` 不是 best_val_mse 而是 best_val_loss — 让 experiment
    汇总用自定义指标选 trial.

与 ``GradientStrategy`` 区别:
  * Loss: ``HuberLoss(delta) + l1_weight * |yhat|.mean()`` (不是纯 MSE).
  * Val 同样用这个 custom loss (不是 MSE), 让 ``best_val_loss`` 与 train loss
    同尺度.
  * primary_metric = ``best_val_loss``, ``best_val_mse`` 作为附带指标.
  * ``weights_best`` 在 ``val_loss`` 改善时落盘, curves 同时记 train_loss /
    val_loss.
  * fit() 末尾用 best weights 算 MSE (与 metrics 错峰), 写进
    ``metrics.best_val_mse``.

参考: examples/custom_trainers/README.md.
"""
from __future__ import annotations
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from . import common as C


class CustomLossTrainer:
    """Custom loss trainer — no-arg 构造, config 走 ``ctx.trainer_config``.

    协议字段:
      * ``name = "custom_loss"``
      * ``required_provider_keys = ("x", "y", "warm_start")``
      * ``primary_metric = "best_val_loss"`` — experiment 汇总用这个
        选 best trial, ``list-runs --metric best_val_loss`` 显示它.
    """

    name = "custom_loss"
    required_provider_keys = ("x", "y", "warm_start")
    primary_metric = "best_val_loss"

    # 默认超参. trainer 应当暴露可配置的字段, 让 ``--trainer-config`` 改;
    # 用户改 trainer.py 也是合法路径.
    DEFAULT_HUBER_DELTA: float = 1.0
    DEFAULT_L1_WEIGHT: float = 0.001

    def __init__(self) -> None:
        # 内部状态: 让 ``--trainer-config`` 的 dict 在 fit() 里读, 避免 no-arg
        # 构造后忘了调 ``configure``.
        pass

    # ---- Trainer protocol ----------------------------------------------

    def validate(self, providers: Dict[str, Any]) -> None:
        """No-op. 不消耗 batch.

        跟 ``GradientTrainer.validate`` 对齐 — 默认 CLI 路径的 RNG 消费
        契约不被打破.
        """
        return None

    def fit(
        self,
        *,
        model: nn.Module,
        providers: Dict[str, Callable[[str], Dict[str, torch.Tensor]]],
        ctx: Any,
        session: Any,
    ) -> Any:
        """跑训练循环, 写产物, finalize, 返回 TrainResult."""
        # 函数内 import 避开 trainer 模块 -> rundown 的反向依赖.
        from rundown.trainers import TrainResult

        # ---- 读 trainer_config ----
        cfg = dict(ctx.trainer_config or {})
        huber_delta = float(cfg.get("huber_delta", self.DEFAULT_HUBER_DELTA))
        l1_weight = float(cfg.get("l1_weight", self.DEFAULT_L1_WEIGHT))
        # lr 也允许 trainer-config 覆盖; 不传 → 走 ctx.lr (defaults.DEFAULT_LR)
        lr = float(cfg.get("lr", ctx.lr))

        # RNG: 跟 GradientStrategy 同款
        C.seed_everything(int(ctx.seed))

        # ---- 模型 / 优化器 / loss ----
        train_p = providers["train"]
        val_p = providers["val"]
        # test 留给 write_test_predictions
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        huber = nn.HuberLoss(delta=huber_delta)

        # ---- 训练循环 ----
        train_losses: list = []
        val_losses: list = []
        best_val_loss = float("inf")
        best_val_mse_at_best = float("inf")
        best_epoch = 0
        epochs_actual = 0
        grad_clip = float(getattr(ctx, "grad_clip_max_norm", 1.0) or 1.0)

        def _custom_loss(yhat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Huber + L1 magnitude penalty.

            L1 是 ``|yhat|.mean()`` — 只压 yhat 的绝对值, 不压 target 差;
            鼓励模型「保守预测」, 跟 MSE 形成 L2 vs L1 的混合.
            """
            return huber(yhat, target) + l1_weight * yhat.abs().mean()

        for epoch in range(1, int(ctx.epochs_max) + 1):
            epochs_actual = epoch
            model.train()
            losses_in_epoch = []

            for _step in range(int(ctx.steps_per_epoch)):
                batch = train_p("train")
                yhat = C.forward(model, batch)
                target = C.target_for(batch, yhat)
                loss = _custom_loss(yhat, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 跟 GradientStrategy 同款: 先 raw grad norm 再 clip
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
                    run_id=ctx.run_id, epoch=epoch, step=_step + 1,
                    train_loss=float(loss.item()),
                    grad_norm=raw_grad_norm, lr=lr,
                )

            epoch_train_loss = sum(losses_in_epoch) / len(losses_in_epoch)
            train_losses.append(epoch_train_loss)

            # ---- val: 用同一个 custom loss ----
            val_loss = C.evaluate_loss(model, val_p, _custom_loss)
            val_losses.append(val_loss)
            # 同时算 val MSE (用 best weights 时也会算, 这里先存一份
            # per-epoch val MSE 给 metrics.best_val_mse 兜底 — 实际
            # best_val_mse 用的是 best epoch 的 weights).
            val_mse = C.evaluate_mse(model, val_p)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # val loss 改善 → 存 weights_best
                session.checkpoint_best(C.state_cpu(model))
                # 在 best weights 上算一次 MSE, 给 metrics 附带 best_val_mse
                best_val_mse_at_best = C.evaluate_mse(model, val_p)

            session.emit(
                "epoch_completed",
                run_id=ctx.run_id, epoch=epoch,
                train_loss=epoch_train_loss,
                val_loss=val_loss,
                val_mse=val_mse,
                lr=lr,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )

        # ---- 写产物 ----
        # final weights
        session.write_weights(C.state_cpu(model), best=False)

        # curves — train_loss / val_loss 两条线; 不写 train_loss=val_mse 这种
        # 错峰混合的 key, 让 inspect 通用表清楚展示.
        session.write_curves({
            "train_loss": torch.tensor(train_losses, dtype=torch.float32),
            "val_loss":   torch.tensor(val_losses,   dtype=torch.float32),
        })

        # test predictions — 用 best weights (与 best_val_loss 对齐)
        C.write_test_predictions(session, model, providers)

        # ---- finalize ----
        metrics = {
            "best_val_loss": best_val_loss,
            "best_val_mse":  best_val_mse_at_best,
            "epochs_to_best": best_epoch,
            "epochs_actual": epochs_actual,
        }
        session.finalize(
            metrics=metrics,
            predictions_primary="test_best",
            epochs_actual=epochs_actual,
            early_stopped=False,
            best_val=best_val_loss,
        )

        return TrainResult(
            metrics=metrics,
            predictions_primary="test_best",
            epochs_actual=epochs_actual,
            early_stopped=False,
        )
