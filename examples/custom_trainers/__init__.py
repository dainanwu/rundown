"""Custom trainer 模板包 (C4).

三个 runnable 模板, 都满足 ``rundown.trainers.Trainer`` 协议:

  * ``custom_loss.CustomLossTrainer`` — 自定义 loss (Huber + L1 weight)
  * ``staged.StagedTrainer``         — 分阶段 (Stage 1 全训练 / Stage 2 解冻优化器)
  * ``joint.JointTrainer``           — 主模型 + 辅助模型, 多优化器, 前缀权重

调用方式::

    uv run python -m rundown.cli train \\
        --dataset etth1 --model lstm \\
        --trainer examples.custom_trainers.custom_loss.CustomLossTrainer \\
        --trainer-config examples/custom_trainers/custom_loss.yaml \\
        --epochs-max 1

设计要点:
  * 三个 trainer 都用 ``RunSession`` 写产物; 永远不调
    ``save_manifest`` / ``set_active_log`` / ``clear_active_log``.
  * 训练配置走 ``ctx.trainer_config`` (``--trainer-config`` 注入),
    trainer 自己读, 不假设 CLI flag.
  * ``no-arg`` 可构造 (resolve_trainer 反射 ``cls()``), 配置从 ctx 拿.
"""
from __future__ import annotations

__all__: list = []
