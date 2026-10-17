"""实验抽象: 一组 runs (sweep) + 调度.

MVP 阶段: 只支持 mode=cartesian, 枚举 variants 列表 — 简单、可预测、可复现.
每个 trial 是独立 subprocess, 跑 rundown.cli train, 失败互不影响.

Phase 2 会加: optuna / TPE search (in-process trial + study.tell()).
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .defaults import DEFAULT_STEPS_PER_EPOCH
from ._env import snapshot_code

# engine.py 当前只实现了 adam 调度. 任何其它 kind 都拒绝 (符合"拒绝 3: 不擅自替你修配置").
SUPPORTED_OPTIMIZERS: frozenset = frozenset({"adam"})


@dataclass
class Experiment:
    name: str
    dataset: str
    model: str
    epochs_max: int
    variants: List[str] = field(default_factory=list)
    steps_per_epoch: int = DEFAULT_STEPS_PER_EPOCH
    batch_size: Optional[int] = None
    optimizer: Optional[Dict[str, Any]] = None  # 实验级覆盖; None → defaults

    def trial_variants(self) -> List[Optional[str]]:
        """返回每个 trial 用的 variant 名列表.

        无 variants 时返回 [None] (单 trial, 用 model 默认 kwargs).
        """
        return [v for v in self.variants] if self.variants else [None]


@dataclass
class TrialResult:
    variant: Optional[str]
    seed: int
    best_val_mse: Optional[float]
    run_dir: Path
    run_id: Optional[str]   # CLI 实际生成的 run_id (= out_dir basename); None if FAILED
    duration_s: float
    returncode: int


_RUN_ID_FROM_STDOUT = re.compile(r"^\[run\]\s+(\S+)")


def run_experiment(
    exp: Experiment,
    *,
    out_root: Path = Path("artifacts"),
    seed_base: int = 0,
) -> List[TrialResult]:
    """Run all trials sequentially. Each trial is a subprocess of rundown.cli train.

    Trial run_id 由 CLI 自动组成 (model + variant + dataset + seed); sweep 不预拼
    run_id, 这样复现 trial 时用同样的 CLI args 就能复用同一产物目录 (run_id 一致).
    """
    trial_variants = exp.trial_variants()
    lr_repr = f"lr={exp.optimizer['lr']}" if exp.optimizer and "lr" in exp.optimizer else "lr=defaults"
    print(f"[experiment] {exp.name}  trials={len(trial_variants)}")
    print(f"  dataset={exp.dataset}  model={exp.model}  epochs_max={exp.epochs_max}  {lr_repr}")
    print()

    # ----- Sweep 共享 snapshot 池: 开头建 1 次, 所有 trial 复用 -----
    # sweep 边界是天然 artifact 集合 — 用户拷走整个 out_root 就完整.
    shared_dir = out_root / "_shared"
    snap_info = snapshot_code(shared_dir, cwd=Path.cwd())
    snap_meta_path: Optional[Path] = None
    if snap_info:
        snap_meta_path = shared_dir / f"{snap_info['archive_path']}.meta.json"
        # archive_path 在 shared_dir 内 (e.g. "code_86f357f6.zip"); subprocess 视角
        # 应是相对 out_root 的路径 (out_root/_shared/code_xxx.zip → _shared/code_xxx.zip).
        snap_info_dict = dict(snap_info)
        snap_info_dict["archive_path"] = f"_shared/{snap_info['archive_path']}"
        snap_meta_path.write_text(
            json.dumps(snap_info_dict, indent=2), encoding="utf-8"
        )
        print(f"  snapshot shared: {snap_meta_path}  "
              f"sha8={snap_info['sha256_short']}  size={snap_info['size_bytes']}")
    print()

    results: List[TrialResult] = []
    for i, variant_name in enumerate(trial_variants):
        cmd = [sys.executable, "-m", "rundown.cli", "train",
               "--dataset", exp.dataset,
               "--model", exp.model,
               "--seed", str(seed_base + i),
               "--epochs-max", str(exp.epochs_max),
               "--out-dir", str(out_root)]
        if variant_name is not None:
            cmd += ["--variant", variant_name]
        if exp.batch_size is not None:
            cmd += ["--batch-size", str(exp.batch_size)]
        if exp.steps_per_epoch != DEFAULT_STEPS_PER_EPOCH:
            cmd += ["--steps-per-epoch", str(exp.steps_per_epoch)]
        if exp.optimizer and "lr" in exp.optimizer:
            cmd += ["--lr", str(exp.optimizer["lr"])]
        if snap_meta_path is not None:
            cmd += ["--snapshot-from-pool", str(snap_meta_path)]

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
        dt = time.time() - t0

        cli_run_id = _RUN_ID_FROM_STDOUT.search(proc.stdout)
        run_id = cli_run_id.group(1) if cli_run_id else None
        trial_dir = out_root / run_id if run_id else out_root  # fallback; will be None in result

        if proc.returncode != 0 or run_id is None:
            print(f"  [{i+1:3d}/{len(trial_variants)}] "
                  f"variant={variant_name}  FAILED (rc={proc.returncode})")
            results.append(TrialResult(variant_name, seed_base + i, None,
                                       trial_dir, run_id, dt, proc.returncode or 1))
            continue

        best_val = _read_best_val_from_events(trial_dir)
        results.append(TrialResult(variant_name, seed_base + i, best_val,
                                   trial_dir, run_id, dt, 0))
        bv_str = f"{best_val:.4f}" if best_val is not None else "n/a"
        print(f"  [{i+1:3d}/{len(trial_variants)}] variant={str(variant_name):8s}  "
              f"best_val_mse={bv_str}  duration={dt:.1f}s")

    # ----- summary -----
    ok = [r for r in results if r.returncode == 0 and r.best_val_mse is not None]
    if ok:
        best = min(ok, key=lambda r: r.best_val_mse)  # type: ignore[arg-type,return-value]
        print()
        print(f"[summary] best: variant={best.variant}  seed={best.seed}  "
              f"best_val_mse={best.best_val_mse:.4f}  run_id={best.run_id}")
        print(f"          reproduce: rundown train --dataset {exp.dataset} --model {exp.model}"
              f"{' --variant ' + best.variant if best.variant else ''} "
              f"--seed {best.seed} --epochs-max {exp.epochs_max}"
              f"{' --batch-size ' + str(exp.batch_size) if exp.batch_size else ''}"
              f" --out-dir <same out_root>")
    else:
        print()
        print(f"[summary] no successful trials.")
    return results


def _read_best_val_from_events(run_dir: Path) -> Optional[float]:
    """从 events.jsonl 读 run_completed 事件的 best_val 字段."""
    p = run_dir / "events.jsonl"
    if not p.exists():
        return None
    best_val = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "run_completed":
            v = rec.get("best_val")
            if isinstance(v, (int, float)):
                best_val = float(v)
    return best_val


__all__ = ["Experiment", "TrialResult", "run_experiment"]