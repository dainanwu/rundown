"""JointTrainer — 主模型 + 辅助模型, 多个 optimizer, 前缀权重.

场景: 多任务学习 / 自监督辅助 loss / 蒸馏. 例 ——
  * 主模型 SequenceLSTM 预测 OT;
  * 辅助模型 SequenceLSTM 预测 OT 的滑动均值 (smoothed target);
  * loss = MSE(main) + alpha * MSE(aux) — 鼓励主模型学 auxiliary signal.

与 staged 区别:
  * 同时训两个模型, 不切换冻结;
  * 两个 optimizer 各管一个模型;
  * 权重合并到同一个 ``weights.safetensors``, 用前缀区分 (``main.`` / ``aux.``).

trainer_config 字段:
    aux_model    str  — auxiliary model alias (e.g. ``"lstm"``). default ``"lstm"``.
    aux_variant  str  — auxiliary model variant (e.g. ``"small"``). default ``"small"``.
    alpha        float — auxiliary loss 权重. default ``0.1``.

主模型用 ctx 提供的 ``model`` 参数; 辅助模型通过 ``ctx.build_model(alias, variant=...)``
拿 — 跟 CLI 构造主模型的注册表同源, 不依赖全局状态.

权重前缀约定 (D8):
    * ``weights.safetensors``  顶层 key: ``main.<param>``, ``aux.<param>``.
    * ``weights_best.safetensors`` 同款前缀.
    * 读取方按前缀过滤 / 拆回: ``state = {k[len("main."):]: v for k, v in state.items() if k.startswith("main.")}``

参考: examples/custom_trainers/README.md.
"""
from __future__ import annotations
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from . import common as C


class JointTrainer:
    """Joint multi-model trainer — no-arg 构造, config 走 ctx.trainer_config.

    协议字段:
      * ``name = "joint"``
      * ``required_provider_keys = ("x", "y", "warm_start")``
      * ``primary_metric = "best_val_mse"`` — 主模型的 val MSE.
    """

    name = "joint"
    required_provider_keys = ("x", "y", "warm_start")
    primary_metric = "best_val_mse"

    # Defaults
    DEFAULT_AUX_MODEL: str = "lstm"
    DEFAULT_AUX_VARIANT: str = "small"
    DEFAULT_ALPHA: float = 0.1

    def __init__(self) -> None:
        pass

    # ---- Trainer protocol ----------------------------------------------

    def validate(self, providers: Dict[str, Any]) -> None:
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

        # ---- 解析 trainer_config ----
        cfg = dict(ctx.trainer_config or {})
        aux_model_alias = str(cfg.get("aux_model", self.DEFAULT_AUX_MODEL))
        aux_variant = cfg.get("aux_variant", self.DEFAULT_AUX_VARIANT)
        alpha = float(cfg.get("alpha", self.DEFAULT_ALPHA))

        # ---- 拿辅助模型 (ctx.build_model 是 CLI 注入的回调) ----
        build_model = getattr(ctx, "build_model", None)
        if build_model is None or not callable(build_model):
            raise ValueError(
                "JointTrainer: ctx.build_model is missing or not callable. "
                "Joint trainer requires the CLI / registry to provide a model "
                "factory callback (see rundown.cli._cmd_train:_build_model)."
            )
        aux_model = build_model(aux_model_alias, variant=aux_variant)

        C.seed_everything(int(ctx.seed))

        train_p = providers["train"]
        val_p = providers["val"]
        # lr: 主模型走 ctx.lr, 辅助模型走 trainer_config.aux_lr (默认同 lr)
        main_lr = float(ctx.lr)
        aux_lr = float(cfg.get("aux_lr", main_lr))
        grad_clip = float(getattr(ctx, "grad_clip_max_norm", 1.0) or 1.0)

        # 两个 Adam optimizer, 各管各的 params (避免两个模型都更新到对方的梯度)
        main_opt = torch.optim.Adam(model.parameters(), lr=main_lr)
        aux_opt = torch.optim.Adam(aux_model.parameters(), lr=aux_lr)

        main_train_losses: list = []
        aux_train_losses: list = []
        val_mses: list = []
        best_val_mse = float("inf")
        best_epoch = 0
        epochs_actual = 0
        aux_val_mses: list = []

        for epoch in range(1, int(ctx.epochs_max) + 1):
            epochs_actual = epoch
            model.train()
            aux_model.train()
            main_loss_sum = 0.0
            aux_loss_sum = 0.0

            for _step in range(int(ctx.steps_per_epoch)):
                batch = train_p("train")
                # main forward / loss
                yhat_main = C.forward(model, batch)
                target = C.target_for(batch, yhat_main)
                main_loss = ((yhat_main - target) ** 2).mean()
                # aux forward / loss — 辅助目标同样用 y; 多任务场景可改
                yhat_aux = C.forward(aux_model, batch)
                aux_loss = ((yhat_aux - target) ** 2).mean()
                joint_loss = main_loss + alpha * aux_loss

                main_opt.zero_grad(set_to_none=True)
                aux_opt.zero_grad(set_to_none=True)
                joint_loss.backward()

                raw_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(aux_model.parameters()),
                        max_norm=float("inf"),
                    ).item()
                )
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(aux_model.parameters()),
                        max_norm=grad_clip,
                    )
                main_opt.step()
                aux_opt.step()

                main_loss_sum += float(main_loss.item())
                aux_loss_sum += float(aux_loss.item())

                session.emit(
                    "step_completed",
                    run_id=ctx.run_id, epoch=epoch, step=_step + 1,
                    train_loss=float(joint_loss.item()),
                    main_loss=float(main_loss.item()),
                    aux_loss=float(aux_loss.item()),
                    grad_norm=raw_grad_norm, lr=main_lr,
                )

            n_steps = max(1, int(ctx.steps_per_epoch))
            ep_main = main_loss_sum / n_steps
            ep_aux = aux_loss_sum / n_steps
            main_train_losses.append(ep_main)
            aux_train_losses.append(ep_aux)

            # val: 主 + 辅
            main_val_mse = C.evaluate_mse(model, val_p)
            aux_val_mse = C.evaluate_mse(aux_model, val_p)
            val_mses.append(main_val_mse)
            aux_val_mses.append(aux_val_mse)

            if main_val_mse < best_val_mse:
                best_val_mse = main_val_mse
                best_epoch = epoch
                # best 权重: 两个模型 state 加前缀合并写进 weights_best
                combined = {}
                combined.update(C.prefix_state(C.state_cpu(model), "main"))
                combined.update(C.prefix_state(C.state_cpu(aux_model), "aux"))
                session.checkpoint_best(combined)

            session.emit(
                "epoch_completed",
                run_id=ctx.run_id, epoch=epoch,
                train_loss=ep_main, aux_train_loss=ep_aux,
                val_mse=main_val_mse, aux_val_mse=aux_val_mse,
                lr=main_lr, alpha=alpha,
                best_val_mse=best_val_mse, best_epoch=best_epoch,
            )

        # ---- 写产物 ----
        # final weights (前缀合并)
        final_combined = {}
        final_combined.update(C.prefix_state(C.state_cpu(model), "main"))
        final_combined.update(C.prefix_state(C.state_cpu(aux_model), "aux"))
        session.write_weights(final_combined, best=False)

        session.write_curves({
            "main_train_loss": torch.tensor(main_train_losses, dtype=torch.float32),
            "aux_train_loss":  torch.tensor(aux_train_losses,  dtype=torch.float32),
            "val_mse":         torch.tensor(val_mses,         dtype=torch.float32),
        })

        # test predictions — 只对主模型写 (aux 是辅助, 不参与 inspect 默认读 test_best)
        # 用 best weights (prefixed), 还原 main.* 子集再 load
        weights_best = session.artifact.weights_best
        if weights_best:
            main_only = {
                k[len("main."):]: v
                for k, v in weights_best.items()
                if k.startswith("main.")
            }
            aux_only = {
                k[len("aux."):]: v
                for k, v in weights_best.items()
                if k.startswith("aux.")
            }
            model.load_state_dict(main_only)
            if aux_only:
                aux_model.load_state_dict(aux_only)
        with torch.no_grad():
            test_batch = providers["test"]("test")
            t_warm = int(test_batch["warm_start"])
            yhat_test = C.forward(model, test_batch)
        session.write_predictions("test_best", {
            "y_hat":  yhat_test.detach().cpu().contiguous(),
            "y_true": test_batch["y"][:, t_warm:t_warm + yhat_test.shape[1]].cpu().contiguous(),
        })

        # ---- finalize ----
        aux_val_mse_final = aux_val_mses[-1] if aux_val_mses else None
        metrics = {
            "best_val_mse":   best_val_mse,
            "aux_val_mse":    aux_val_mse_final,
            "epochs_to_best": best_epoch,
            "epochs_actual":  epochs_actual,
        }
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
