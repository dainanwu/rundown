"""测试夹具.

提供最小可 import 的模型 + BatchProvider, 供 ModelEntry.build() 与
engine 反射调用. 不放进 rundown 包, 避免循环 import.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

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


def build_random_providers(
    source_path: str = "",
    splits: Optional[Dict[str, float]] = None,
    target_col: Optional[str] = None,
    *,
    in_dim: int = 7,
    warm_start: int = 4,
    rollout: int = 8,
    batch_size: int = 4,
    **overrides: Any,
) -> Dict[str, "RandomProvider"]:
    """Registry-facing builder for RandomProvider — 给 CLI 集成测试用.

    签名跟 ``DatasetEntry.build_providers`` 约定一致: builder(source_path,
    splits, target_col, **overrides). source_path / splits / target_col 在
    RandomProvider 这里没意义 (我们不读 DataFrame), 但参数列表必须对齐,
    不然 ``registry.DatasetEntry.build_providers`` 反射时 TypeError.

    **overrides 接受 ``in_dim`` / ``warm_start`` / ``rollout`` /
    ``batch_size`` — 让 configs/datasets.yaml 的写法更真实.
    """
    return {
        "train": RandomProvider(in_dim=in_dim, warm_start=warm_start,
                                rollout=rollout, batch_size=batch_size, seed=0),
        "val":   RandomProvider(in_dim=in_dim, warm_start=warm_start,
                                rollout=rollout, batch_size=batch_size, seed=1),
        "test":  RandomProvider(in_dim=in_dim, warm_start=warm_start,
                                rollout=rollout, batch_size=batch_size, seed=2),
    }


class DummyTrainer:
    """C2 用的最小自定义 trainer — 跑通 ``Trainer`` 协议 (写产物 + finalize).

    行为:
        * no-arg 构造 (registry ``resolve_trainer`` 直接 ``DummyTrainer()``).
        * ``name = "dummy"`` / ``required_provider_keys = ("x","y","warm_start")`` /
          ``primary_metric = "best_val_mse"`` (与 GradientTrainer 对齐, 让 test
          走同一份 manifest 字段集).
        * ``validate(providers)`` 是 no-op (与 GradientTrainer 一致).
        * ``fit()``: 把 model 当前的 state_dict 写到 weights / weights_best,
          写最小 train_loss / val_mse 曲线, 写 ``test_best`` / ``test`` 预测
          (零 tensor — 不算 MSE), 调 ``session.finalize(...)`` 收尾, 返回
          ``TrainResult(metrics={"best_val_mse": 0.5, "epochs_actual": 1})``.

    注: fit 内部不 import ``rundown.trainers.TrainResult`` 模块顶层 — 改成
    函数内 import, 避免 ``tests.fixtures`` -> ``rundown.trainers`` 反向
    import (虽然循环 import 在 py3 上不会崩, 但函数内 import 让依赖单向).
    """
    name = "dummy"
    required_provider_keys = ("x", "y", "warm_start")
    primary_metric = "best_val_mse"

    def __init__(self) -> None:
        # 用来让 test 验证 trainer 内部状态隔离
        self.fit_call_count = 0

    def validate(self, providers: Dict[str, Any]) -> None:
        """no-op. 保留默认 trainer 的 RNG 消费契约."""
        return None

    def fit(
        self,
        *,
        model: Any,
        providers: Dict[str, Any],
        ctx: Any,
        session: Any,
    ) -> Any:
        # 函数内 import, 避开 fixtures -> rundown 反向依赖.
        from rundown.trainers import TrainResult

        self.fit_call_count += 1

        # 写 weights / weights_best (用 model 当前 state_dict)
        state = {k: v.detach().cpu().contiguous().clone()
                 for k, v in model.state_dict().items()}
        session.write_weights(state, best=False)
        session.checkpoint_best(state)

        # 写曲线 (一个 epoch 的占位)
        session.write_curves({
            "train_loss": torch.zeros(1, dtype=torch.float32),
            "val_mse":    torch.zeros(1, dtype=torch.float32),
        })

        # 写 test_best / test 预测 (zeros, 不真算)
        # 用一个 batch 的 shape 估一下; 不调 providers, 避免 RNG 消费.
        test_provider = providers.get("test")
        if test_provider is not None and callable(test_provider):
            sample = test_provider("test")
            Tprime = sample["x"].shape[1] - int(sample["warm_start"])
        else:
            Tprime = 8
        y_hat = torch.zeros(1, Tprime, dtype=torch.float32)
        y_true = torch.zeros(1, Tprime, dtype=torch.float32)
        session.write_predictions("test_best", {"y_hat": y_hat, "y_true": y_true})
        session.write_predictions("test",     {"y_hat": y_hat, "y_true": y_true})

        # finalize
        session.finalize(
            metrics={"best_val_mse": 0.5, "epochs_actual": 1},
            predictions_primary="test_best",
            epochs_actual=1,
            early_stopped=False,
            best_val=0.5,
        )

        return TrainResult(
            metrics={"best_val_mse": 0.5, "epochs_actual": 1},
            predictions_primary="test_best",
            epochs_actual=1,
        )


class IncompleteTrainer:
    """C2 反例: 只有 fit, 缺 validate / primary_metric / required_provider_keys.

    给 ``resolve_trainer`` 的协议字段校验用 (应抛 ValueError, 而不是让 CLI
    在训练中途 AttributeError).
    """

    def fit(self, **kwargs: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError("IncompleteTrainer.fit should not be called")


__all__ = [
    "ToyNet", "RandomProvider", "build_random_providers",
    "DummyTrainer", "IncompleteTrainer",
]
