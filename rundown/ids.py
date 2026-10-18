"""ID 校验. 一处定义, 处处使用.

正则契约 (与 AGENTS.md §3 同步):
  dataset_alias: ^[a-z][a-z0-9_]{2,40}$
  model_id:      ^[a-z][a-z0-9_]{2,40}$
  trainer_alias: ^[a-z][a-z0-9_]{2,40}$   # C2 新增
  run_id:        ^[a-z][a-z0-9_]+__[a-z][a-z0-9_]+__seed\\d+$

任何写入或解析 ID 的入口 (CLI / 注册表 / artifact) 都应先调 assert_valid.
不合法即 ValueError; 不静默回落.
"""
from __future__ import annotations
import re
from typing import Final

DATASET_ALIAS_RE:  Final[str] = r"^[a-z][a-z0-9_]{2,40}$"
MODEL_ID_RE:       Final[str] = r"^[a-z][a-z0-9_]{2,40}$"
TRAINER_ALIAS_RE:  Final[str] = r"^[a-z][a-z0-9_]{2,40}$"
RUN_ID_RE:         Final[str] = r"^[a-z][a-z0-9_]+__[a-z][a-z0-9_]+__seed\d+$"

_PATTERNS: Final[dict] = {
    "dataset_alias": re.compile(DATASET_ALIAS_RE),
    "model_id":      re.compile(MODEL_ID_RE),
    "trainer_alias": re.compile(TRAINER_ALIAS_RE),
    "run_id":        re.compile(RUN_ID_RE),
}


def assert_valid(kind: str, value: str) -> None:
    """校验 ID 合法性. 不合法抛 ValueError 含原因."""
    if kind not in _PATTERNS:
        raise ValueError(f"unknown id kind: {kind!r}; expected one of {list(_PATTERNS)}")
    if not isinstance(value, str):
        raise ValueError(f"{kind}: expected str, got {type(value).__name__}")
    if not _PATTERNS[kind].match(value):
        raise ValueError(
            f"invalid {kind}: {value!r}; must match {_PATTERNS[kind].pattern}"
        )


def is_valid(kind: str, value: str) -> bool:
    return kind in _PATTERNS and bool(_PATTERNS[kind].match(value))


__all__ = [
    "DATASET_ALIAS_RE", "MODEL_ID_RE", "TRAINER_ALIAS_RE", "RUN_ID_RE",
    "assert_valid", "is_valid",
]
