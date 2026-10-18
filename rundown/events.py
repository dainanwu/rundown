"""训练事件流. 一个 run 一个 events.jsonl.

不是日志库. 不接 stdlib logging. 不开 log_level.
``kind`` 字段即是分级: 训练循环用 epoch_completed / early_stopped;
注册用 dataset_registered / model_registered; 调度用 run_started / run_completed / run_failed.

使用方式:
    set_active_log(run_dir, run_id)   # 在 run 启动时调一次
    emit("epoch_completed", epoch=42, train_loss=0.42)   # 之后业务侧任意位置

不在 run 上下文时, emit() 只写 stderr, 不落文件.

每次 set_active_log() 会 unlink 同一 run_dir 下残留的 events.jsonl, 保证新
run 从空白事件流开始; 但不 eager create 文件 — 第一次 emit() 才会落盘.
IO 错误 (含 unlink 失败) 不会被吞, 直接抛给调用方.

stderr 行格式 (agent 友好 — 统一前缀 + JSON 字段值):
    {t_iso}  {kind}  {json.dumps(fields)}
例:
    2026-10-16T21:23:35  step_completed  {"epoch": 2, "step": 10, "train_loss": 0.42}

agent 解析路径: split 行 by "  " (两个空格), 第 3 段 json.loads() — 复杂 value
(带空格字符串 / list / 嵌套 dict) 安全. 不再裸 key=value 拼字符串.

Thread-safety: NOT thread-safe. _ACTIVE_DIR / _ACTIVE_RUN_ID 是 module global,
emit() 直接读写没有 lock. 多线程并发 emit 会 events.jsonl 行交错 + 丢失 run_id.
要并发 run 走多进程 (一个 run 一个进程, 跟 experiment sweep 现状一致).
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

_ACTIVE_DIR: Optional[Path] = None
_ACTIVE_RUN_ID: Optional[str] = None
_QUIET: bool = False  # True 时 emit() 不写 stderr (events.jsonl 仍写, artifact 完整)


def set_active_log(run_dir: Path, run_id: str) -> Path:
    """在 run 启动时调用.

    1. 创建 run_dir (parents=True, exist_ok=True).
    2. unlink 任何已存在的 events.jsonl — 同 run_dir 被重新激活 (例如
       interrupt + 重跑 / agent 重复触发同名 run) 时, 旧事件流会被静默累积
       进新 run 的文件; 主动清掉, 保证新 run 从干净 stream 开始.
    3. 不 eager create 文件; 第一次 emit() 才会落 events.jsonl (旧 test
       锁住的契约).
    4. IO 错误 (含 unlink 失败 / permission denied) 不吞, 直接抛出 —
       跟「不擅自替你修」一致.

    返回 events.jsonl 路径.
    """
    global _ACTIVE_DIR, _ACTIVE_RUN_ID
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        events_path.unlink()
    _ACTIVE_DIR = run_dir
    _ACTIVE_RUN_ID = run_id
    return events_path


def active_run_id() -> Optional[str]:
    return _ACTIVE_RUN_ID


def set_quiet(quiet: bool) -> None:
    """控制 emit() 是否写 stderr. events.jsonl 永远写 (artifact 完整).

    CLI --quiet 在 main() 开头调一次. 多次调用 OK.
    """
    global _QUIET
    _QUIET = quiet


def is_quiet() -> bool:
    return _QUIET


def emit(kind: str, **fields: Any) -> None:
    """发一条事件. 双写: events.jsonl (永远写) + stderr (受 _QUIET 控制)."""
    # JSON 序列化失败 (含 numpy/torch tensor 等不可序列化对象) 时
    # fallback 到 repr — 不阻塞训练, 跟「不阻塞哲学」一致.
    try:
        fields_json = json.dumps(fields, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        fields_json = json.dumps(
            {k: repr(v) for k, v in fields.items()},
            ensure_ascii=False, sort_keys=True,
        )

    if _ACTIVE_DIR is None:
        # 不在 run 上下文 — 只写 stderr (受 quiet 控制)
        if not _QUIET:
            t_iso = datetime.now().isoformat(timespec="seconds")
            sys.stderr.write(f"{t_iso}  {kind}  {fields_json}\n")
        return

    rec = {
        "t":      time.time(),
        "t_iso":  datetime.now().isoformat(timespec="seconds"),
        "run_id": _ACTIVE_RUN_ID,
        "kind":   kind,
        **fields,
    }
    rec_json = json.dumps(rec, ensure_ascii=False, sort_keys=True)
    path = _ACTIVE_DIR / "events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(rec_json + "\n")
    if not _QUIET:
        sys.stderr.write(f"{rec['t_iso']}  {kind}  {fields_json}\n")


def clear_active_log() -> None:
    """Run 结束 (成功 / 失败) 时调用, 重置全局指针."""
    global _ACTIVE_DIR, _ACTIVE_RUN_ID
    _ACTIVE_DIR = None
    _ACTIVE_RUN_ID = None


__all__ = [
    "set_active_log", "active_run_id", "emit", "clear_active_log",
    "set_quiet", "is_quiet",
]
