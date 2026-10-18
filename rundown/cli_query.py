"""rundown list-runs / compare — artifact 目录查询工具.

设计原则:
  * 不建 DB, glob artifacts/*/manifest.yaml 直接读. 符合 rundown 哲学.
  * 过滤器全是 YAML 字段, 不需要解析 predictions safetensors.
  * 输出 default table (人类读); --format json 给 agent.
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class RunRow:
    run_id: str
    dataset_alias: str
    model_alias: str
    variant: Optional[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    run_dir: Path = Path()


def _load_run_manifests(root: Path) -> List[RunRow]:
    """glob <root>/*/manifest.yaml. 任何 yaml 解析错误跳过 (artifact 损坏不阻塞 list)."""
    rows: List[RunRow] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/manifest.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        run_id = data.get("run_id") or path.parent.name
        dataset_alias = (data.get("dataset") or {}).get("alias", "?")
        model = data.get("model") or {}
        model_alias = model.get("alias", "?")
        variant = model.get("variant")
        rows.append(RunRow(
            run_id=run_id,
            dataset_alias=dataset_alias,
            model_alias=model_alias,
            variant=variant,
            metrics=data.get("metrics") or {},
            config=data.get("config") or {},
            created_at=data.get("created_at"),
            run_dir=path.parent,
        ))
    return rows


def _apply_filters(
    rows: List[RunRow],
    dataset: Optional[str],
    model: Optional[str],
    variant: Optional[str],
    above_metric: Optional[Tuple[str, float]],
    below_metric: Optional[Tuple[str, float]],
) -> List[RunRow]:
    out: List[RunRow] = []
    for r in rows:
        if dataset and r.dataset_alias != dataset:
            continue
        if model and r.model_alias != model:
            continue
        if variant and r.variant != variant:
            continue
        if above_metric:
            name, val = above_metric
            v = r.metrics.get(name)
            if not isinstance(v, (int, float)) or v <= val:
                continue
        if below_metric:
            name, val = below_metric
            v = r.metrics.get(name)
            if not isinstance(v, (int, float)) or v >= val:
                continue
        out.append(r)
    return out


def list_runs(
    *,
    root: Path = Path("artifacts"),
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    variant: Optional[str] = None,
    above_metric: Optional[Tuple[str, float]] = None,
    below_metric: Optional[Tuple[str, float]] = None,
    metric: str = "best_val_mse",
    fmt: str = "table",
) -> int:
    """list-runs: 读 artifacts/, 过滤, 输出.

    C4 — 新增 ``metric: str = "best_val_mse"``: 表格列用哪个 metric 名.
    默认仍是 ``best_val_mse``, 旧 caller 行为不变. JSON 输出字段集合
    不受 ``metric`` 参数影响 (json 永远携带 ``metrics`` 整 dict).

    与 ``above_metric`` / ``below_metric`` 区别: 后两者是 filter (决定 *哪些*
    行出现), ``metric`` 是显示 (决定 *展示的列*).
    """
    rows = _apply_filters(
        _load_run_manifests(root),
        dataset=dataset, model=model, variant=variant,
        above_metric=above_metric, below_metric=below_metric,
    )
    if fmt == "json":
        json.dump(
            [r.__dict__ | {"run_dir": str(r.run_dir)} for r in rows],
            sys.stdout, indent=2, default=str,
        )
        print()
        return 0

    # table format
    if not rows:
        print(f"(no runs under {root} matching filters)")
        return 0
    header = ("run_id", "dataset", "model", "variant", metric, "epochs_to_best", "created_at")
    print("  ".join(f"{h:<22s}" if h != "run_id" else f"{h:<46s}" for h in header))
    for r in rows:
        mv = r.metrics.get(metric)
        mv_str = f"{mv:.4f}" if isinstance(mv, (int, float)) else "-"
        eb = r.metrics.get("epochs_to_best")
        eb_str = str(eb) if eb is not None else "-"
        ca = r.created_at or "-"
        v = r.variant or "-"
        print(f"{r.run_id:<46s}  {r.dataset_alias:<22s}  {r.model_alias:<22s}  "
              f"{v:<22s}  {mv_str:<22s}  {eb_str:<22s}  {ca}")
    return 0


def _flat(d: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
    """把嵌套 dict 展平成 [(path, value), ...] 用于 diff."""
    out: List[Tuple[str, Any]] = []
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flat(v, path))
        else:
            out.append((path, v))
    return out


def compare_runs(a_id: str, b_id: str, *, root: Path = Path("artifacts")) -> int:
    """side-by-side diff 两个 run 的 manifest 关键字段."""
    a_path = root / a_id / "manifest.yaml"
    b_path = root / b_id / "manifest.yaml"
    if not a_path.exists():
        print(f"ERROR: {a_path} not found", file=sys.stderr)
        return 2
    if not b_path.exists():
        print(f"ERROR: {b_path} not found", file=sys.stderr)
        return 2
    a = yaml.safe_load(a_path.read_text(encoding="utf-8")) or {}
    b = yaml.safe_load(b_path.read_text(encoding="utf-8")) or {}

    a_flat = dict(_flat(a))
    b_flat = dict(_flat(b))
    all_keys = sorted(set(a_flat) | set(b_flat))
    print(f"  {'field':<35s}  {a_id:<40s}  {b_id:<40s}  diff")
    print("  " + "-" * 120)
    for k in all_keys:
        av = a_flat.get(k, "<absent>")
        bv = b_flat.get(k, "<absent>")
        if av == bv:
            continue  # 只输出不同的字段 (大量相同噪音)
        # 截断长字符串
        av_str = str(av)[:38]
        bv_str = str(bv)[:38]
        diff_marker = ""
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            diff_marker = f"  Δ={bv - av:+.4f}"
        print(f"  {k:<35s}  {av_str:<40s}  {bv_str:<40s}{diff_marker}")
    return 0


__all__ = ["list_runs", "compare_runs", "RunRow"]