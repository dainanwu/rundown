"""实验抽象: 一组 runs (sweep) + 调度.

MVP 阶段: 只支持 mode=cartesian, 枚举 variants 列表 — 简单、可预测、可复现.
每个 trial 是独立 subprocess, 跑 rundown.cli train, 失败互不影响.

Phase 2 会加: optuna / TPE search (in-process trial + study.tell()).

C3 (trainer-layer-plan) 加:
  - Experiment.trainer / trainer_config / metric (YAML 可选字段, sweep 透传)
  - TrialResult 泛化: best_metric + metric_name (兼容 best_val_mse 属性)
  - run_experiment 拼 --trainer / --trainer-config <json> 给 subprocess
  - _read_metric: 优先 manifest.metrics[name], best_val_mse 仍兼容 events.jsonl
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .defaults import DEFAULT_STEPS_PER_EPOCH
from ._env import snapshot_code

# engine.py 当前只实现了 adam 调度. 任何其它 kind 都拒绝 (符合"拒绝 3: 不擅自替你修配置").
SUPPORTED_OPTIMIZERS: frozenset = frozenset({"adam"})

# L6 (trainer-layer-plan) 锁的 metric 解析优先级:
#   exp.metric > manifest.config.primary_metric > "best_val_mse"
_DEFAULT_METRIC: str = "best_val_mse"


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
    # C3 — 可选字段. 不写时全 None / {}: 旧 experiments.yaml 行为不变.
    #   trainer         alias (e.g. "joint") 或 dotted path
    #                   (e.g. "myproj.trainers.JointTrainer"); 非空时给 subprocess
    #                   加 ``--trainer <value>``.
    #   trainer_config  dict; 序列化后给 subprocess 加 ``--trainer-config <json>``.
    #                   启动前用 ``json.dumps`` 校验 JSON 可序列化.
    #   metric          sweep 汇总指标名; None → 读 manifest.config.primary_metric
    #                   → "best_val_mse" (按优先级).
    trainer: Optional[str] = None
    trainer_config: Optional[Dict[str, Any]] = None
    metric: Optional[str] = None

    def trial_variants(self) -> List[Optional[str]]:
        """返回每个 trial 用的 variant 名列表.

        无 variants 时返回 [None] (单 trial, 用 model 默认 kwargs).
        """
        return [v for v in self.variants] if self.variants else [None]


@dataclass
class TrialResult:
    """一次 trial 的产物摘要. C3 泛化自原 ``best_val_mse`` 字段.

    字段 (向后兼容):
        variant       variant 名 (or None)
        seed          trial 用的 seed (= seed_base + trial_index)
        best_metric   sweep 选的汇总指标的 numeric 值 (e.g. best_val_mse);
                      None 时 trial 被认为是不可汇总的 (失败 / metric 缺失).
        run_dir       trial 落地的 run_dir
        run_id        CLI 实际生成的 run_id (= out_dir basename); None if FAILED
        duration_s    trial 总耗时
        returncode    subprocess returncode (0 = 成功)
        metric_name   实际用的汇总指标名 (e.g. "best_val_mse", "custom_metric").
                      ``best_val_mse`` 兼容属性读 ``best_metric`` 时要求
                      metric_name in {None, "best_val_mse"}.

    兼容层 (C3): ``best_val_mse`` 属性在
    ``metric_name in {None, "best_val_mse"}`` 时返回 ``best_metric``, 否则
    None. 旧 caller (test_smoke / examples) 仍按 ``r.best_val_mse`` 读
    best_val_mse, 不需要改.
    """
    variant: Optional[str]
    seed: int
    best_metric: Optional[float]
    run_dir: Path
    run_id: Optional[str]   # CLI 实际生成的 run_id (= out_dir basename); None if FAILED
    duration_s: float
    returncode: int
    metric_name: Optional[str] = None

    @property
    def best_val_mse(self) -> Optional[float]:
        """C2 兼容层: 旧 caller 仍按 ``r.best_val_mse`` 读 best_val_mse.

        只在 metric_name 是 None (legacy 调用方构造) 或 "best_val_mse" 时
        返回 best_metric 数值; 其它 metric (e.g. "custom_metric") → None,
        防止误把 custom metric 当成 best_val_mse.
        """
        if self.metric_name is None or self.metric_name == "best_val_mse":
            return self.best_metric
        return None


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

    C3:
      * ``exp.trainer``  非空 → subprocess 加 ``--trainer <value>``.
      * ``exp.trainer_config`` 非空 → 序列化 (json.dumps, ensure_ascii=True,
        sort_keys=True) 后给 subprocess 加 ``--trainer-config <json>``;
        启动前 JSON 校验, 不通过 → ValueError (CLI 在 _cmd_experiment_run
        接住, return 2).
      * ``exp.metric``    可选; 不传 → 读 trial manifest.config.primary_metric,
        再兜底 "best_val_mse". 汇总用这个 metric 选 best trial.
    """
    trial_variants = exp.trial_variants()
    lr_repr = f"lr={exp.optimizer['lr']}" if exp.optimizer and "lr" in exp.optimizer else "lr=defaults"
    print(f"[experiment] {exp.name}  trials={len(trial_variants)}")
    print(f"  dataset={exp.dataset}  model={exp.model}  epochs_max={exp.epochs_max}  {lr_repr}")
    if exp.trainer:
        print(f"  trainer={exp.trainer}")
    if exp.trainer_config:
        # 已经在下面做了完整校验; 这里只显示简短信息, 调试用
        print(f"  trainer_config={exp.trainer_config}")
    print()

    # ----- trainer_config JSON 校验 (L15). 启动前报错, 不让 trial 跑到一半才炸 -----
    trainer_config_json: Optional[str] = None
    if exp.trainer_config:
        try:
            trainer_config_json = json.dumps(
                exp.trainer_config, ensure_ascii=True, sort_keys=True,
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"experiment {exp.name!r}: trainer_config is not JSON serializable "
                f"({type(exp.trainer_config).__name__}: {e}). "
                f"subprocess 透传需要 ASCII-safe JSON; "
                f"非 JSON 类型请转 str / 删 nested object."
            ) from e

    # ----- Sweep 共享 snapshot 池: 开头建 1 次, 所有 trial 复用 -----
    # sweep 边界是天然 artifact 集合 — 用户拷走整个 out_root 就完整.
    shared_dir = out_root / "_shared"
    snap_info = snapshot_code(shared_dir, cwd=Path.cwd())
    snap_meta_path: Optional[Path] = None
    if snap_info:
        # meta 文件必须落在 _shared/ 级别 (snapshot 归档的 sibling), 不能进
        # tar 解压出来的 code_<sha8>/ 内部 — 那会被 snapshot 复制覆盖, 而且
        # inspect_run._resolve_code_snapshot() 只在 _shared/ 顶层找 meta.
        snap_meta_path = _snapshot_meta_path(shared_dir, snap_info)
        # archive_path 在 shared_dir 内 (e.g. "code_86f357f6.zip"); subprocess 视角
        # 应是相对 out_root 的路径 (out_root/_shared/code_xxx.zip → _shared/code_xxx.zip).
        # 写进 JSON 的 archive_path 字段本身保持原值不变.
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
        # C3: trainer / trainer_config 透传给 subprocess.
        if exp.trainer:
            cmd += ["--trainer", exp.trainer]
        if trainer_config_json is not None:
            cmd += ["--trainer-config", trainer_config_json]
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
            results.append(TrialResult(
                variant=variant_name,
                seed=seed_base + i,
                best_metric=None,
                run_dir=trial_dir,
                run_id=run_id,
                duration_s=dt,
                returncode=proc.returncode or 1,
                metric_name=None,
            ))
            continue

        # C3: 读 metric 走 _read_metric, 优先 manifest, 仅 best_val_mse 兜底 events.
        metric_value, metric_name = _read_metric(trial_dir, exp.metric)
        results.append(TrialResult(
            variant=variant_name,
            seed=seed_base + i,
            best_metric=metric_value,
            run_dir=trial_dir,
            run_id=run_id,
            duration_s=dt,
            returncode=0,
            metric_name=metric_name,
        ))
        # 显示用: metric_name 解析后一定非空 (读 manifest / 兜底 "best_val_mse");
        # 如果 exp.metric 是 None 但 manifest 没 primary_metric, 仍兜底 best_val_mse.
        display_name = metric_name or _DEFAULT_METRIC
        bv_str = f"{metric_value:.4f}" if metric_value is not None else "n/a"
        print(f"  [{i+1:3d}/{len(trial_variants)}] variant={str(variant_name):8s}  "
              f"{display_name}={bv_str}  duration={dt:.1f}s")

    # ----- summary -----
    # C3: 用 best_metric (无论叫 best_val_mse 还是 custom) 选最佳 trial.
    ok = [r for r in results if r.returncode == 0 and r.best_metric is not None]
    if ok:
        best = min(ok, key=lambda r: r.best_metric)  # type: ignore[arg-type,return-value]
        metric_display = best.metric_name or _DEFAULT_METRIC
        print()
        print(f"[summary] best: variant={best.variant}  seed={best.seed}  "
              f"{metric_display}={best.best_metric:.4f}  run_id={best.run_id}")
        # 复现命令: 包含 --trainer (exp.trainer), 不 inline trainer_config JSON
        # (引号 / 编码跨平台差异大 — 精确复现走 `rundown experiment run <name>`,
        # 见 L18 / README)。
        trainer_flag = f" --trainer {exp.trainer}" if exp.trainer else ""
        print(f"          reproduce: rundown train --dataset {exp.dataset} --model {exp.model}"
              f"{' --variant ' + best.variant if best.variant else ''} "
              f"--seed {best.seed} --epochs-max {exp.epochs_max}"
              f"{' --batch-size ' + str(exp.batch_size) if exp.batch_size else ''}"
              f"{trainer_flag}"
              f" --out-dir <same out_root>")
        if exp.trainer or exp.trainer_config:
            print(f"          exact reproduce: rundown experiment run {exp.name}"
                  f"{' --trainer ' + exp.trainer if exp.trainer else ''}"
                  f" --out-root <same out_root>")
    else:
        print()
        print(f"[summary] no successful trials.")
    return results


def _snapshot_meta_path(shared_dir: Path, snap_info: Dict[str, Any]) -> Path:
    """sweep 入口写 snapshot meta json 的 sibling 路径.

    旧实现 ``shared_dir / f"{archive_path}.meta.json"`` 在 tar 模式下会拼
    ``_shared/code_<sha8>/.meta.json`` — 那是 *tar 解压目录内部* 的路径, meta
    会被 snapshot 复制进来, 而且 inspect_run._resolve_code_snapshot() 只在
    _shared/ 顶层找 ``code_<sha>.meta.json`` / ``code_<sha>.zip.meta.json``,
    找不到会回退到自建.

    规则: meta 文件是 archive 的 sibling, 始终在 _shared/ 顶层:
      tar: archive_path="code_<sha8>/"       → ``code_<sha8>.meta.json``
      zip: archive_path="code_<sha8>.zip"    → ``code_<sha8>.zip.meta.json``

    写进 meta JSON 里的 ``archive_path`` 字段保持原值不变 (调用方决定).
    """
    archive = snap_info["archive_path"]
    if archive.endswith("/"):
        # tar: 把尾部 / 去掉 — "code_<sha8>" 即 stem
        stem = archive.rstrip("/")
    else:
        # zip: 完整文件名作 stem
        stem = archive
    return shared_dir / f"{stem}.meta.json"


def _read_metric(
    run_dir: Path,
    metric_name: Optional[str],
) -> Tuple[Optional[float], Optional[str]]:
    """读 run 的 metric 值. 返回 ``(value, resolved_name)``.

    解析优先级 (L6):
        1. ``metric_name`` 入参 (None 时跳过)
        2. ``manifest.config.primary_metric`` (C2 trainer 声明)
        3. ``"best_val_mse"`` (兜底, 旧 run / 未声明 trainer 的 run)

    数据源优先级 (跟 L6 兼容):
        1. ``manifest.yaml.metrics[<resolved>]`` (numeric 才返回)
        2. 仅 ``<resolved> == "best_val_mse"`` 时, fall back 到
           ``events.jsonl`` 的 ``run_completed.best_val`` (兼容旧 run).
           非 best_val_mse 的 metric 不 fallback, 直接 ``(None, name)`` 返回
           (custom metric 没有旧契约可走).

    manifest 缺失 / 不可解析 / 字段不是 numeric → 走 fallback (best_val_mse
    专属) 或返回 ``(None, name)`` (其它 metric).

    返回 ``name`` 给 caller 用 (打印 / 复现命令), 即使 value 是 None 也返回
    非 None 的 name (让 caller 能稳定显示).
    """
    manifest_path = run_dir / "manifest.yaml"
    manifest_cfg: Optional[Dict[str, Any]] = None
    manifest_metrics: Optional[Dict[str, Any]] = None
    if manifest_path.exists():
        try:
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg = data.get("config")
                if isinstance(cfg, dict):
                    manifest_cfg = cfg
                metrics = data.get("metrics")
                if isinstance(metrics, dict):
                    manifest_metrics = metrics
        except (yaml.YAMLError, OSError):
            pass

    # 1) 解析 metric_name (入参 > primary_metric > 默认).
    resolved: str
    if metric_name is not None:
        resolved = metric_name
    elif manifest_cfg is not None:
        pm = manifest_cfg.get("primary_metric")
        resolved = pm if isinstance(pm, str) and pm else _DEFAULT_METRIC
    else:
        resolved = _DEFAULT_METRIC

    # 2) 优先 manifest.metrics[resolved].
    if manifest_metrics is not None:
        v = manifest_metrics.get(resolved)
        if isinstance(v, (int, float)):
            return float(v), resolved

    # 3) 仅 best_val_mse 走 events.jsonl fallback.
    if resolved == _DEFAULT_METRIC:
        v = _read_best_val_from_events(run_dir)
        if v is not None:
            return v, resolved

    return None, resolved


def _read_best_val_from_events(run_dir: Path) -> Optional[float]:
    """Scan ``events.jsonl`` 的 ``run_completed.best_val`` 字段.

    抽出单独函数, 让 ``_read_best_val`` 和 ``_read_metric`` 都复用同一段
    fallback 逻辑. 旧 run 的 events.jsonl 没有 numeric best_val_mse 时
    返回 None (跟 Fix D 行为一致).
    """
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


def _read_best_val(run_dir: Path) -> Optional[float]:
    """读 run 的 best_val_mse (兼容 Fix D 测试).

    优先 ``manifest.yaml.metrics.best_val_mse`` (项目唯一真值来源);
    manifest 缺失 / 不可解析 / 字段不是 numeric 时, fall back 到
    ``events.jsonl`` 的 ``run_completed.best_val`` 字段.

    C3: 改成 ``_read_metric(run_dir, "best_val_mse")`` 的薄包装, 保持
    TestFixD_ReadBestValPrefersManifest 等旧测试仍然可用.
    返回 ``Optional[float]`` (旧契约).
    """
    return _read_metric(run_dir, "best_val_mse")[0]


__all__ = ["Experiment", "TrialResult", "run_experiment"]
