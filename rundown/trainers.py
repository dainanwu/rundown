"""Trainer 协议 + 内置 GradientTrainer 适配器 + 别名/导入解析.

C2 范围: 把"训练范式"从 CLI 写死的 GradientStrategy 抽出成可注册的 Trainer.

设计要点:
  * ``Trainer`` 是 ``Protocol`` (结构子类型), 任何满足 name / required_provider_keys /
    primary_metric / validate / fit 的类都视为合法 trainer.
  * ``GradientTrainer`` 是 ``GradientStrategy`` 的薄适配器, 把
    ``providers`` / ``ctx`` / ``session`` 喂给 ``_fit_with_session``,
    保留 C1 的所有 RNG / manifest 行为.
  * registry 三种来源:
      1. 内置 ``gradient``  ->  ``GradientTrainer`` (不可被覆盖, L16).
      2. 可选 ``configs/trainers.yaml`` (alias -> dotted target).
      3. 直接写 dotted path (e.g. ``myproj.trainers.JointTrainer``).
  * 选择规则:
        含 ``.``  -> 视为 dotted path, canonical = 路径本身.
        不含 ``.`` -> 视为 alias, 未注册 -> ``ValueError`` (列出可用 alias).

调用方契约:
  * trainer 类必须 no-arg 可构造 (factory.target 反射 ``cls()``).
  * 训练配置走 ``ctx.trainer_config`` (CLI ``--trainer-config`` 注入), trainer
    自己读. trainer 不知道有 ``--trainer-config`` 这个 CLI flag.
  * 一次 run 的产物 (manifest / weights / curves / predictions / events)
    一律走 ``RunSession`` 写; trainer 不直接调 ``save_manifest`` /
    ``set_active_log`` / ``clear_active_log``.
"""
from __future__ import annotations
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

from . import ids
from .engine import GradientStrategy


# ---------------------------------------------------------------------------
# TrainResult — trainer.fit() 的返回值
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """trainer.fit() 返回. session.finalize() 把它落进 manifest.

    字段:
        metrics:             写进 manifest.metrics
        predictions_primary: 写进 manifest.config.predictions_primary
        epochs_actual:       写进 manifest.metrics (键) + run_completed 事件
        early_stopped:       写进 run_completed 事件
    """
    metrics: Dict[str, Any]
    predictions_primary: str = "test_best"
    epochs_actual: Optional[int] = None
    early_stopped: bool = False


# ---------------------------------------------------------------------------
# Trainer Protocol — 结构子类型
# ---------------------------------------------------------------------------

class Trainer(Protocol):
    """trainer 协议. 不强制继承, 只要满足 shape 即可被 registry 接受."""
    name: str
    required_provider_keys: tuple  # e.g. ("x", "y", "warm_start")
    primary_metric: str            # e.g. "best_val_mse"

    def validate(self, providers: Dict[str, Any]) -> None:
        """训练前对 providers 做契约校验. 失败 ValueError.

        注: GradientTrainer.validate() 是 no-op — 不消耗 batch, 以保住默认
        CLI 路径的 RNG 消费一致 (与 C1 完全相同的训练循环).
        """
        ...

    def fit(
        self,
        *,
        model: Any,                  # torch.nn.Module — 不强 import
        providers: Dict[str, Any],   # {"train": ..., "val": ..., "test": ...}
        ctx: Any,                    # EngineConfig — 不强 import
        session: Any,                # RunSession — 不强 import
    ) -> TrainResult:
        """训练 + 写产物 + session.finalize()."""
        ...


# ---------------------------------------------------------------------------
# GradientTrainer — GradientStrategy 的 C2 适配器
# ---------------------------------------------------------------------------

class GradientTrainer:
    """内置 trainer: 包 ``GradientStrategy._fit_with_session``.

    ``name = "gradient"``; ``required_provider_keys = ("x", "y", "warm_start")``
    (与 BatchProvider 契约对齐). ``primary_metric = "best_val_mse"``.

    ``validate()`` 是 no-op: 不消费 providers. 这与 C1 默认行为一致 — C1 的
    GradientStrategy.fit() 不做 batch 契约校验, 训练循环里第一次 ``_resolve_batch``
    失败才抛错. CLI 默认路径 RNG 消费要与 C1 一致, 所以这里不能
    ``providers["train"]("train")`` 那样预热.

    ``fit()`` 走 ``GradientStrategy()._fit_with_session(...)`` (C1 内部方法),
    拿到 session 直接复用 — 不重做 set_active_log / save_manifest.
    """
    name = "gradient"
    required_provider_keys = ("x", "y", "warm_start")
    primary_metric = "best_val_mse"

    def validate(self, providers: Dict[str, Any]) -> None:
        """no-op. 保留默认 CLI 路径的 RNG 消费."""
        return None

    def fit(
        self,
        *,
        model: Any,
        providers: Dict[str, Any],
        ctx: Any,
        session: Any,
    ) -> TrainResult:
        """调 ``GradientStrategy()._fit_with_session`` 然后把返回 dict 包成 TrainResult.

        ``_fit_with_session`` 内部:
          * 用 ``ctx.seed`` 调 ``_set_seed`` (与 C1 完全一致).
          * 走完整训练循环 (strategy_started / step_completed /
            epoch_completed / early_stopped).
          * 写 weights / weights_best / curves / test / test_best 预测.
          * 调 ``session.finalize(...)`` 写 metrics / env / run_completed.

        它返回 ``{epochs_actual, best_val, early_stopped}`` — 我们再补 metrics
        (从 ``session.artifact.manifest.metrics`` 读回) 和 predictions_primary
        (从 ``session.artifact.manifest.config["predictions_primary"]`` 读回).
        """
        strat = GradientStrategy()
        info = strat._fit_with_session(model, providers, ctx, session)
        manifest = session.artifact.manifest
        return TrainResult(
            metrics=dict(manifest.metrics),
            predictions_primary=str(
                manifest.config.get("predictions_primary", "test_best")
            ),
            epochs_actual=info.get("epochs_actual"),
            early_stopped=bool(info.get("early_stopped", False)),
        )


# ---------------------------------------------------------------------------
# Registry — builtin + configs/trainers.yaml + dotted path
# ---------------------------------------------------------------------------

_BUILTIN_TRAINERS: Dict[str, type] = {
    "gradient": GradientTrainer,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    """读 yaml.

    缺文件 -> {} (可选配置). 文件存在但读不了 / YAML 损坏 / 顶层不是
    mapping -> ValueError (不静默把配置当空; 跟「不擅自替你修」一致).
    """
    if not path.exists():
        return {}
    import yaml
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as e:
        raise ValueError(f"{path}: cannot read trainer registry: {e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"{path}: invalid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level must be a mapping, got {type(data).__name__}"
        )
    return data


def load_trainer_aliases(
    configs_dir: Path = Path("configs"),
) -> Dict[str, str]:
    """读 ``configs/trainers.yaml``, 返回 alias -> dotted target 映射.

    校验:
        * alias 必须合法 ``trainer_alias`` (走 ``ids.assert_valid``).
        * alias 不能 shadow built-in (``gradient``) — L16.
        * target 必须 ``module.attr`` 形式 (含 ``.``); 我们不 import, 只
          校验 shape; 真正的 import 在 ``resolve_trainer`` 里.

    文件缺 / 没 ``trainers:`` 块 / 块非 dict / 任何 entry 不合法 -> ValueError.
    """
    path = Path(configs_dir) / "trainers.yaml"
    raw = _load_yaml(path)
    if not raw:
        return {}

    body = raw.get("trainers")
    if body is None:
        return {}
    if not isinstance(body, dict):
        raise ValueError(
            f"{path}: top-level 'trainers' must be a mapping, "
            f"got {type(body).__name__}"
        )

    out: Dict[str, str] = {}
    for alias, spec in body.items():
        if alias in _BUILTIN_TRAINERS:
            raise ValueError(
                f"{path}: trainer alias {alias!r} shadows built-in "
                f"{list(_BUILTIN_TRAINERS)}"
            )
        ids.assert_valid("trainer_alias", alias)
        if not isinstance(spec, dict):
            raise ValueError(
                f"{path}: trainers.{alias!r} must be a mapping, "
                f"got {type(spec).__name__}"
            )
        target = spec.get("target")
        if not target or not isinstance(target, str) or "." not in target:
            raise ValueError(
                f"{path}: trainers.{alias!r}.target must be a dotted path "
                f"'module.attr', got {target!r}"
            )
        out[alias] = target
    return out


def _import_trainer_target(target: str) -> Any:
    """import ``module.attr`` (e.g. ``myproj.trainers.JointTrainer``).

    失败抛 ValueError 含上下文 (module path / 错误原因).
    """
    module_path, _, attr = target.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"trainer target must be 'module.attr', got {target!r}"
        )
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"trainer: cannot import {module_path!r} "
            f"from target={target!r}: {e}"
        ) from e
    cls_or_obj = getattr(mod, attr, None)
    if cls_or_obj is None:
        raise ValueError(
            f"trainer: {module_path!r} has no attribute {attr!r} "
            f"(target={target!r})"
        )
    return cls_or_obj


def resolve_trainer(
    name: str,
    configs_dir: Path = Path("configs"),
) -> Tuple[Any, str]:
    """解析 trainer 名称 -> (instance, canonical_name).

    规则 (与 L16 锁定):
        * ``name == "gradient"`` -> 内置 ``GradientTrainer``,
          canonical = ``"gradient"``.
        * ``name`` 含 ``.`` -> 视为 dotted path, canonical = name 本身
          (直接 import, 不看 alias).
        * 否则 -> 在 ``configs/trainers.yaml`` 的 alias 表里查; 找不到
          -> ``ValueError`` 列出可用 alias (含 built-in).
        * 找到 target 后: import (importlib + getattr), no-arg 实例化,
          要求实例有 ``fit`` 属性. 任何失败 -> ValueError 含上下文.

    不会修改全局状态; 每次调用都返回新实例 (trainer 内部可能存 self.*,
    实例之间互不影响).
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"trainer name must be non-empty str, got {name!r}")

    if name == "gradient":
        return _BUILTIN_TRAINERS["gradient"](), "gradient"

    if "." in name:
        target = name
        canonical = name
    else:
        # 查 alias 表
        aliases = load_trainer_aliases(configs_dir=configs_dir)
        if name not in aliases:
            available = sorted(set(_BUILTIN_TRAINERS) | set(aliases))
            raise ValueError(
                f"unknown trainer alias {name!r}; available: {available}. "
                f"Or use a dotted path (e.g. 'myproj.trainers.MyTrainer')."
            )
        target = aliases[name]
        canonical = name

    # import + instantiate
    try:
        cls_or_obj = _import_trainer_target(target)
    except ValueError:
        raise

    # no-arg 实例化 (trainer 必须是类; 实例化后才能校验协议字段)
    try:
        instance = cls_or_obj()
    except TypeError as e:
        raise ValueError(
            f"trainer {target!r}: must be no-arg constructible, "
            f"got TypeError: {e}"
        ) from e
    missing = [
        attr for attr in ("fit", "validate", "primary_metric", "required_provider_keys")
        if not hasattr(instance, attr)
    ]
    if missing:
        raise ValueError(
            f"trainer {target!r}: instance of {type(instance).__name__} "
            f"is missing required Trainer attribute(s): {missing}"
        )
    return instance, canonical


__all__ = [
    "TrainResult",
    "Trainer",
    "GradientTrainer",
    "load_trainer_aliases",
    "resolve_trainer",
]
