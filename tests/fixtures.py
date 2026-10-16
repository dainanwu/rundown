"""测试夹具.

提供最小可 import 的模型 + BatchProvider, 供 ModelEntry.build() 与
engine 反射调用. 不放进 rundown 包, 避免循环 import.
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class ToyNet(nn.Module):
    """最简单的 per-time-step 线性回归. 适配 rundown engine 的 forward(x, warm_start=) 签名.

    forward 返回 [B, T'] 其中 T' = T - warm_start. Engine 负责把 target 切片到同样长度.
    """

    def __init__(self, in_dim: int = 7, hidden: int = 16) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, warm_start: int = 0) -> torch.Tensor:
        # x: [B, T, in_dim]; 只取 rollout 段; 输出每步一个标量
        h = self.lin(x[:, warm_start:])  # [B, T', hidden]
        return self.out(h).squeeze(-1)   # [B, T']


class RandomProvider:
    """最小 BatchProvider: 生成随机 tensor 给 ToyNet 之类的 smoke 模型.

    Per BatchProvider contract:
        f(split: str) -> {"x": Tensor[B,T,F], "y": Tensor[B,T], "warm_start": int}

    与 examples/baseline/data.py 的 SlidingWindowProvider 区别: 这个不接
    DataFrame, 直接随机生成. 只用于 smoke 测试, 不替代真实 data loader.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        warm_start: int = 4,
        rollout: int = 16,
        batch_size: int = 32,
        seed: int = 0,
    ) -> None:
        self.in_dim = in_dim
        self.warm_start = warm_start
        self.seq_len = warm_start + rollout + 1   # +1 for the last target
        self.batch_size = batch_size
        self.rng = torch.Generator().manual_seed(seed)

    def __call__(self, split: str = "train") -> Dict[str, torch.Tensor]:  # noqa: ARG002
        x = torch.randn(self.batch_size, self.seq_len, self.in_dim, generator=self.rng)
        y = torch.randn(self.batch_size, self.seq_len, generator=self.rng)
        return {"x": x, "y": y, "warm_start": self.warm_start}


__all__ = ["ToyNet", "RandomProvider"]
