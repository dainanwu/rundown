"""Custom trainer 公共 helpers — 三个模板都从这里抽.

约束:
  * 不 import ``rundown`` (用户 trainer 文件不许 import rundown — L1/L11).
    唯一例外: 函数内 ``from rundown.trainers import TrainResult`` — 这是
    trainer 协议返回值, 必须 import.
  * torch + numpy + stdlib. 不引入新运行时依赖.
  * 没有「框架魔法」: helpers 都是普通函数 / closure, 三个 trainer 看一眼就能
    改 — 鼓励拷贝改写, 不藏抽象.

注: 模型工厂的 ``forward(x, warm_start=...)`` 签名是 rundown 默认契约 (见
``rundown.engine._forward``); ``forward()`` 在这里统一包一层, 自动 squeeze
``[B, T, 1]`` 输出, 让任意 ``[B, T]`` / ``[B, T, 1]`` 模型都能直接用.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed torch + numpy, 设单线程 + cuDNN 确定性.

    与 ``rundown.engine._set_seed`` 行为对齐; 这里再暴露一份, 让 trainer
    在自己调 ``_set_seed`` 之前就能拿到一致的 RNG 状态.

    为什么 trainer 也要 seed: ``GradientStrategy._fit_with_session`` 自己
    seed; 但 custom trainer (e.g. ``CustomLossTrainer``) 不一定走
    ``_fit_with_session``, 需要显式 seed 才能保证 manifest 的 ``env.seed``
    与训练 RNG 对得上.
    """
    s = int(seed)
    torch.manual_seed(s)
    np.random.seed(s)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Forward / target 切片
# ---------------------------------------------------------------------------

def forward(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """调 ``model(x, warm_start=...)`` 并 squeeze 末尾 1 维.

    与 ``rundown.engine._forward`` 行为一致 — 这里重新暴露让 trainer 不 import
    ``rundown``. 输出 shape ``[B, T']`` (T' = T - warm_start).
    """
    yhat = model(batch["x"], warm_start=int(batch["warm_start"]))
    if yhat.dim() == 3 and yhat.shape[-1] == 1:
        yhat = yhat.squeeze(-1)
    return yhat


def target_for(batch: Dict[str, torch.Tensor], yhat: torch.Tensor) -> torch.Tensor:
    """把 ``y`` 切成与 ``yhat`` 同样长度 ``[B, T']``.

    contract: ``batch["y"]`` 全长 ``T``, warm_start 决定起点; ``yhat.shape[1]``
    决定长度. 与 engine._fit_with_session 内的 target 切片同款.
    """
    warm = int(batch["warm_start"])
    return batch["y"][:, warm:warm + yhat.shape[1]]


# ---------------------------------------------------------------------------
# State dict helpers
# ---------------------------------------------------------------------------

def state_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """``state_dict()`` 拷成 CPU contiguous, 准备写 weights/weights_best.

    safetensors 写入前 detach + cpu + contiguous + clone 是必须的 —
    engine 里也有同样模式, 这里集中一份.
    """
    return {
        k: v.detach().cpu().contiguous().clone()
        for k, v in model.state_dict().items()
    }


def prefix_state(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """把 state dict 加 key 前缀 (``main.`` / ``aux.`` 等).

    多模型 trainer (e.g. joint) 共享一个 ``weights.safetensors`` 文件 —
    加前缀防止参数名冲突, 读取方按前缀还原.
    """
    if not prefix or not isinstance(prefix, str):
        raise ValueError(f"prefix must be non-empty str, got {prefix!r}")
    return {f"{prefix}.{k}": v for k, v in state.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mse(
    model: nn.Module,
    provider: Callable[[str], Dict[str, torch.Tensor]],
    *,
    split: str = "val",
) -> float:
    """在 ``provider(split)`` 上跑一轮 forward, 算 MSE.

    单 batch eval — 不为 val/test 跑多 batch. 简单够用; 真要 k-fold / 多 batch
    请在 trainer 内自己实现.
    """
    model.eval()
    batch = provider(split)
    yhat = forward(model, batch)
    target = target_for(batch, yhat)
    return float(((yhat - target) ** 2).mean().item())


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    provider: Callable[[str], Dict[str, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    split: str = "val",
) -> float:
    """Single-batch eval on ``provider(split)`` with given loss_fn.

    公共 API: trainer 调 ``evaluate_loss(model, providers["val"], custom_loss_fn)``.
    """
    model.eval()
    batch = provider(split)
    yhat = forward(model, batch)
    target = target_for(batch, yhat)
    return float(loss_fn(yhat, target).item())


# ---------------------------------------------------------------------------
# Test predictions
# ---------------------------------------------------------------------------

def write_test_predictions(
    session: Any,
    model: nn.Module,
    providers: Dict[str, Callable[[str], Dict[str, torch.Tensor]]],
    *,
    primary_split: str = "test_best",
) -> Any:
    """写 ``predictions/test_best.safetensors``.

    用 ``weights_best`` (primary 评估指标对应权重) forward, 与 manifest.metrics
    对齐. 失败 (没有 test provider / 没有 weights_best) 让 ``RuntimeError``
    自然传上去 — 不静默.

    返回 ``RunArtifact.write_predictions`` 的路径.
    """
    test_provider = providers.get("test")
    if test_provider is None or not callable(test_provider):
        raise ValueError(
            "write_test_predictions: providers['test'] missing or not callable"
        )
    weights_best = session.artifact.weights_best
    if not weights_best:
        raise RuntimeError(
            "write_test_predictions: weights_best not written yet; "
            "call session.checkpoint_best() first."
        )

    # 用 best weights 算 primary predictions, 与 manifest.metrics 对齐
    model.load_state_dict(weights_best)
    model.eval()
    batch = test_provider("test")
    t_warm = int(batch["warm_start"])
    yhat = forward(model, batch)
    return session.write_predictions(primary_split, {
        "y_hat":  yhat.detach().cpu().contiguous(),
        "y_true": batch["y"][:, t_warm:t_warm + yhat.shape[1]].cpu().contiguous(),
    })


# ---------------------------------------------------------------------------
# Tiny utils
# ---------------------------------------------------------------------------

def safe_metric(value: Any) -> Optional[float]:
    """``None`` / 非数字 → ``None``; 否则 ``float()``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
