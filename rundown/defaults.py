"""rundown 全局默认值 — 单点真值.

修改这里一处即影响 EngineConfig / CLI / Experiment 三处的 fallback.
"""
from __future__ import annotations
from typing import Any, Dict

# EngineConfig / GradientStrategy 默认超参
DEFAULT_STEPS_PER_EPOCH: int = 50
DEFAULT_EPOCHS_MAX: int = 100
DEFAULT_LR: float = 1.0e-3
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_SEED: int = 0

# 单次 train (无 experiment) 用的默认 optimizer 配置.
# sweep 实验里通过 experiments.yaml.optimizer 字段覆盖, 不读这里.
DEFAULT_OPTIMIZER: Dict[str, Any] = {"kind": "adam", "lr": 1.0e-3}

# Per-step events
DEFAULT_STEP_EVENT_EVERY_N: int = 10
DEFAULT_GRAD_NORM_THRESHOLD: float = 100.0


__all__ = [
    "DEFAULT_STEPS_PER_EPOCH",
    "DEFAULT_EPOCHS_MAX",
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_SEED",
    "DEFAULT_OPTIMIZER",
    "DEFAULT_STEP_EVENT_EVERY_N",
    "DEFAULT_GRAD_NORM_THRESHOLD",
]
