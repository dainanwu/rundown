"""YAML 注册表. 一条记录 -> 一个对象. 反射绑定.

设计要点:
  * ``factory.target`` 形如 ``myproj.models.MyNet``. 注册时立即 import, 写回 yaml.
  * 不支持隐式回落 / 隐式继承 (extends 留 Phase 2).
  * 找不到 factory.target 即 ValueError 含模块路径 + 原因.
  * dataset / model / experiment 各自一个入口, 不共享解析路径.
  * dataset 自带 ``provider_factory`` (dotted path to builder function) +
    ``splits`` (train/val/test 比例), 让 CLI 通过 ``build_providers()`` 一步拿
    到 {train, val, test} 三个 BatchProvider, 无需 user 自己写 build script.
  * model 自带 ``variants`` (variant_name -> kwargs override), 同一 alias 可
    有多个超参组合 (small / large 等), CLI ``--variant large`` 切换.
"""
from __future__ import annotations
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .ids import assert_valid


# Default split ratios used when a dataset entry doesn't specify its own.
_DEFAULT_SPLITS: Dict[str, float] = {"train": 0.7, "val": 0.1, "test": 0.2}


@dataclass
class DatasetEntry:
    alias: str
    source_kind: str
    source_path: str
    target_col: Optional[str] = None
    provider_factory: Optional[str] = None   # dotted path to builder function
    splits: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_SPLITS))

    def __post_init__(self) -> None:
        assert_valid("dataset_alias", self.alias)
        # Validate splits: ratios in (0, 1), sum ≈ 1
        if not self.splits:
            return
        for k, v in self.splits.items():
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"dataset {self.alias!r}: split {k!r}={v} must be in [0, 1]"
                )
        total = sum(self.splits.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"dataset {self.alias!r}: splits must sum to 1.0, got {total:.4f} "
                f"({self.splits})"
            )

    def build_providers(self, **factory_overrides: Any) -> Dict[str, Any]:
        """调用 provider_factory 拿 {train, val, test} 三组 BatchProvider.

        provider_factory 签名约定:
            builder(source_path: str, splits: Dict[str, float], target_col: Optional[str], **overrides)
                -> Dict[str, BatchProvider]

        Returns dict with keys "train", "val", "test".
        Raises ValueError if provider_factory missing / unimportable.
        """
        if not self.provider_factory:
            raise ValueError(
                f"dataset {self.alias!r}: provider_factory missing. "
                f"Add it to configs/datasets.yaml (dotted path to a builder function)."
            )
        module_path, _, attr = self.provider_factory.rpartition(".")
        if not module_path or not attr:
            raise ValueError(
                f"dataset {self.alias!r}: provider_factory must be 'module.attr', "
                f"got {self.provider_factory!r}"
            )
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(
                f"dataset {self.alias!r}: cannot import {module_path!r} "
                f"from provider_factory={self.provider_factory!r}: {e}"
            ) from e
        builder = getattr(mod, attr, None)
        if builder is None:
            raise ValueError(
                f"dataset {self.alias!r}: {module_path!r} has no attribute {attr!r}"
            )
        providers = builder(
            source_path=self.source_path,
            splits=self.splits,
            target_col=self.target_col,
            **factory_overrides,
        )
        if not isinstance(providers, dict) or not all(
            k in providers for k in ("train", "val", "test")
        ):
            raise ValueError(
                f"dataset {self.alias!r}: provider_factory must return "
                f"dict with keys train/val/test, got {type(providers).__name__}"
            )
        return providers


@dataclass
class ModelEntry:
    """一个模型 = PyTorch 类 + 默认 kwargs + 便捷 variants.

    设计边界 (跟 experiments.yaml 的区别):
      - factory.target:   PyTorch 类引用 (必填)
      - factory.kwargs:   固定结构 kwargs (数据集决定的, 如 in_dim)
      - variants:         便捷预设 (同一模型不同 size 组合, --variant 用)
      - ❌ optimizer:     不在 models — experiments 控制 (或 defaults.DEFAULT_OPTIMIZER)
      - ❌ epochs_max:    不在 models — experiments 控制 (或 defaults.DEFAULT_EPOCHS_MAX)

    --kwargs CLI 透传 (方案 C) 跟 variants 互补: variants 是预设, --kwargs 自由.
    """
    alias: str
    factory_target: str
    factory_kwargs: Dict[str, Any]
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_valid("model_id", self.alias)
        if not isinstance(self.factory_kwargs, dict):
            raise ValueError(
                f"model {self.alias!r}: factory.kwargs must be dict, "
                f"got {type(self.factory_kwargs).__name__}"
            )
        if not isinstance(self.variants, dict):
            raise ValueError(
                f"model {self.alias!r}: variants must be dict[str, dict], "
                f"got {type(self.variants).__name__}"
            )
        for vname, vkwargs in self.variants.items():
            if not isinstance(vkwargs, dict):
                raise ValueError(
                    f"model {self.alias!r}: variants[{vname!r}] must be dict, "
                    f"got {type(vkwargs).__name__}"
                )
        if not self.factory_target or "." not in self.factory_target:
            raise ValueError(
                f"model {self.alias!r}: factory.target must be a dotted path "
                f"'module.attr' (e.g. 'myproj.models.Net'), got {self.factory_target!r}"
            )

    def build(
        self,
        variant: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any], Optional[str]]:
        """反射绑定: import factory.target 并以 kwargs 实例化. 失败即 ValueError.

        kwargs 合并顺序 (后者覆盖前者):
            1. factory_kwargs      (基础, 来自 YAML models.factory.kwargs)
            2. variants[variant]   (如果指定 variant, 来自 YAML models.variants)
            3. extra_kwargs        (CLI --kwargs 透传, optuna sweep 用)

        Returns: (model_instance, merged_kwargs, chosen_variant_name).
        caller 用 merged_kwargs 写 manifest; chosen_variant 用于记录.

        注意: extra_kwargs 里不在 factory.target 签名的 key 会让 cls(**)
        抛 TypeError — 这是用户写错, 不在 registry 层拦截 (宁报错不猜).
        """
        module_path, _, attr = self.factory_target.rpartition(".")
        if not module_path or not attr:
            raise ValueError(
                f"model {self.alias!r}: factory.target must be 'module.attr', "
                f"got {self.factory_target!r}"
            )
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(
                f"model {self.alias!r}: cannot import {module_path!r} "
                f"from factory.target={self.factory_target!r}: {e}. "
                f"Install the package or fix the path."
            ) from e
        cls = getattr(mod, attr, None)
        if cls is None:
            raise ValueError(
                f"model {self.alias!r}: {module_path!r} has no attribute {attr!r}"
            )
        merged_kwargs = dict(self.factory_kwargs)
        chosen_variant: Optional[str] = None
        if variant is not None:
            if variant not in self.variants:
                raise ValueError(
                    f"model {self.alias!r}: variant {variant!r} not declared. "
                    f"Available: {sorted(self.variants)}"
                )
            merged_kwargs.update(self.variants[variant])
            chosen_variant = variant
        if extra_kwargs:
            merged_kwargs.update(extra_kwargs)
        return cls(**merged_kwargs), merged_kwargs, chosen_variant


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}


def load_datasets(configs_dir: Path = Path("configs")) -> Dict[str, DatasetEntry]:
    """加载 configs/datasets.yaml. 返回 alias -> DatasetEntry 字典."""
    path = Path(configs_dir) / "datasets.yaml"
    raw = _load_yaml(path)
    out: Dict[str, DatasetEntry] = {}
    for alias, body in raw.get("datasets", {}).items():
        if not isinstance(body, dict):
            raise ValueError(f"dataset {alias!r}: body must be mapping, got {type(body).__name__}")
        src = body.get("source", {}) or {}
        # splits: optional override; absent -> default 0.7/0.1/0.2
        raw_splits = body.get("splits")
        splits = dict(raw_splits) if raw_splits else dict(_DEFAULT_SPLITS)
        out[alias] = DatasetEntry(
            alias=alias,
            source_kind=str(src.get("kind", "?")),
            source_path=str(src.get("path", "")),
            target_col=body.get("target_col"),
            provider_factory=body.get("provider_factory"),
            splits=splits,
        )
    return out


def load_models(configs_dir: Path = Path("configs")) -> Dict[str, ModelEntry]:
    """加载 configs/models.yaml. 返回 alias -> ModelEntry 字典."""
    path = Path(configs_dir) / "models.yaml"
    raw = _load_yaml(path)
    out: Dict[str, ModelEntry] = {}
    for alias, body in raw.get("models", {}).items():
        if not isinstance(body, dict):
            raise ValueError(f"model {alias!r}: body must be mapping, got {type(body).__name__}")
        fac = body.get("factory", {}) or {}
        if "target" not in fac:
            raise ValueError(f"model {alias!r}: factory.target missing")
        out[alias] = ModelEntry(
            alias=alias,
            factory_target=str(fac["target"]),
            factory_kwargs=fac.get("kwargs", {}) or {},
            variants=body.get("variants", {}) or {},
        )
    return out


def load_experiments(configs_dir: Path = Path("configs")) -> Dict[str, "Experiment"]:
    """加载 configs/experiments.yaml. 返回 name -> Experiment 字典.

    Experiment 在 rundown.experiment 里定义; 延迟导入避免循环.
    """
    from .experiment import Experiment  # local import to avoid circular
    path = Path(configs_dir) / "experiments.yaml"
    raw = _load_yaml(path)
    out: Dict[str, Experiment] = {}
    for name, body in raw.get("experiments", {}).items():
        if not isinstance(body, dict):
            raise ValueError(f"experiment {name!r}: body must be mapping, got {type(body).__name__}")
        # 必填字段
        for required in ("dataset", "model", "epochs_max"):
            if required not in body:
                raise ValueError(f"experiment {name!r}: {required!r} missing")
        out[name] = Experiment(
            name=name,
            dataset=str(body["dataset"]),
            model=str(body["model"]),
            epochs_max=int(body["epochs_max"]),
            variants=list(body.get("variants", []) or []),
            steps_per_epoch=int(body.get("steps_per_epoch", 50)),
            batch_size=(int(body["batch_size"]) if body.get("batch_size") is not None else None),
        )
    return out


__all__ = ["DatasetEntry", "ModelEntry", "load_datasets", "load_models", "load_experiments"]
