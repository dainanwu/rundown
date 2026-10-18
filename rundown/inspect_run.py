"""rundown inspect — 读 rundown run artifact, 输出 README-style markdown.

哲学:
  * Read-only, 不动 artifacts/.
  * 缺 pyyaml / safetensors 时降级, 至少能读 manifest.yaml.
  * stdout = markdown (agent 可 `rundown inspect ... > run.md`).
  * 一次跑得清楚: purpose / setup / results / evaluate 四个段落.

CLI:
  rundown inspect <run_id>            # 单 run → markdown 到 stdout
  rundown inspect --all               # 当前 out_root 下所有 run
  rundown inspect --since-days 7      # 近 7 天
  rundown inspect --root artifacts <run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- optional deps ----
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

try:
    from safetensors import safe_open  # type: ignore
except ImportError:  # pragma: no cover
    safe_open = None  # type: ignore


SHA8 = 8  # code snapshot short hash length


# ============================================================================
# RunIntel — 一个 run 的只读视图
# ============================================================================

@dataclass
class EpochPoint:
    epoch: int
    train_loss: Optional[float]
    val_mse: Optional[float]
    is_best: bool


@dataclass
class SamplePrediction:
    y_hat_first5: List[float]
    y_true_first5: List[float]
    shape: Tuple[int, int]


@dataclass
class RunIntel:
    """Read-only view of one rundown run dir.  Never writes back."""
    run_id: str
    run_dir: Path
    manifest: Dict[str, Any]
    curves_epochs: List[EpochPoint] = field(default_factory=list)
    predictions_sample: Optional[SamplePrediction] = None
    event_kinds: Dict[str, int] = field(default_factory=dict)
    code_snapshot: Optional[Dict[str, Any]] = None  # resolved from _shared
    artifacts_on_disk: List[Tuple[str, int]] = field(default_factory=list)  # (relpath, bytes)
    # C4 — generic curve series (train_loss / val_loss / val_mse / stage1_… etc).
    # 放在 dataclass 末尾以保持旧 positional 构造兼容; populated by ``load_run``.
    curves: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def created_at(self) -> Optional[str]:
        v = self.manifest.get("created_at")
        return v if isinstance(v, str) else None

    @property
    def git_commit(self) -> Optional[str]:
        env = self.manifest.get("env") or {}
        v = env.get("git_commit")
        return v[:SHA8] if isinstance(v, str) else None

    @property
    def trainer_name(self) -> Optional[str]:
        """C4: 读 manifest.config.trainer (CLI/trainer 注册表写入).

        不存在 → None; 渲染时 Setup 段才决定是否展示 ``- **Trainer**: …`` 一行.
        """
        cfg = self.manifest.get("config") or {}
        v = cfg.get("trainer")
        return v if isinstance(v, str) and v else None


# ============================================================================
# Loaders
# ============================================================================

def _safe_yaml_load(path: Path) -> Optional[Dict[str, Any]]:
    if yaml is None or not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_curves(run_dir: Path) -> List[EpochPoint]:
    path = run_dir / "curves.safetensors"
    if safe_open is None or not path.exists():
        return []
    try:
        with safe_open(str(path), framework="numpy") as f:
            keys = set(f.keys())
            if "train_loss" not in keys or "val_mse" not in keys:
                return []
            train = list(f.get_tensor("train_loss"))
            val = list(f.get_tensor("val_mse"))
    except (OSError, ValueError):
        return []
    if len(train) != len(val):
        return []
    # "best" = argmin(val)
    best_idx = min(range(len(val)), key=lambda i: val[i])
    return [
        EpochPoint(
            epoch=i + 1,
            train_loss=float(train[i]),
            val_mse=float(val[i]),
            is_best=(i == best_idx),
        )
        for i in range(len(train))
    ]


def _load_curve_series(run_dir: Path) -> Dict[str, List[float]]:
    """C4: 读 ``curves.safetensors`` 里所有 1-D numeric series.

    与 ``_load_curves`` 互补: 后者只看 ``train_loss`` / ``val_mse`` 两个键,
    返回 ``List[EpochPoint]``; 这一个把全部键都以 ``{name: [float, ...]}``
    形式返回, 给通用表格 (custom trainers 写 ``stage1_train_loss`` /
    ``stage2_train_loss`` / ``val_loss`` / ``main_train_loss`` 等任意名都直接渲染).

    读取策略:
      * ``safe_open`` 不可用 → ``{}``.
      * 文件不存在 / 读失败 → ``{}``.
      * 每个键:
          - 1-D ``tensor`` (n,)         → 取 ``.tolist()`` 转 ``List[float]``.
          - 0-D ``tensor`` (标量)        → 包成 ``[float]`` (1 元素 list).
          - 其它 shape / dtype           → 跳过 (不污染通用表).
    """
    path = run_dir / "curves.safetensors"
    if safe_open is None or not path.exists():
        return {}
    try:
        with safe_open(str(path), framework="numpy") as f:
            keys = list(f.keys())
            series: Dict[str, List[float]] = {}
            for k in keys:
                try:
                    arr = f.get_tensor(k)
                except (OSError, ValueError):
                    continue
                # Skip non-numeric dtypes (rare; safetensors numpy backend
                # 默认就是 float32, 这里守底).
                if arr.dtype.kind not in ("f", "i", "u"):
                    continue
                if arr.ndim == 0:
                    series[k] = [float(arr.reshape(()))]
                elif arr.ndim == 1:
                    series[k] = [float(v) for v in arr.tolist()]
                # else: skip — 不强行重塑, 让 manifest 字段自负
    except (OSError, ValueError):
        return {}
    return series


def _load_predictions_sample(run_dir: Path) -> Optional[SamplePrediction]:
    """Read predictions/test_best.safetensors (preferred), else predictions/test.safetensors."""
    if safe_open is None:
        return None
    for name in ("test_best.safetensors", "test.safetensors"):
        path = run_dir / "predictions" / name
        if not path.exists():
            continue
        try:
            with safe_open(str(path), framework="numpy") as f:
                keys = set(f.keys())
                if "y_hat" not in keys or "y_true" not in keys:
                    continue
                import numpy as np  # only here, only when path exists
                y_hat = np.asarray(f.get_tensor("y_hat"))
                y_true = np.asarray(f.get_tensor("y_true"))
        except (OSError, ValueError, ImportError):
            continue
        if y_hat.size == 0 or y_true.size == 0:
            continue
        return SamplePrediction(
            y_hat_first5=[round(float(v), 4) for v in y_hat[0, :5].tolist()],
            y_true_first5=[round(float(v), 4) for v in y_true[0, :5].tolist()],
            shape=(int(y_hat.shape[0]), int(y_hat.shape[1])),
        )
    return None


def _event_kinds(run_dir: Path) -> Dict[str, int]:
    """Lightweight scan of events.jsonl — kind count only, no full parse."""
    path = run_dir / "events.jsonl"
    if not path.exists():
        return {}
    counts: Dict[str, int] = {}
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = obj.get("kind")
                if isinstance(k, str):
                    counts[k] = counts.get(k, 0) + 1
    except OSError:
        return {}
    return counts


def _resolve_code_snapshot(manifest: Dict[str, Any], artifacts_root: Path) -> Optional[Dict[str, Any]]:
    """Read the actual code snapshot meta from _shared/code_<sha>.<ext>.

    manifest.env.code_snapshot is the *intent*; we look up the on-disk file to confirm.
    """
    env = manifest.get("env") or {}
    snap = env.get("code_snapshot") or {}
    sha = snap.get("sha256_short")
    if not isinstance(sha, str):
        return None
    # Try tar dir first, then zip
    for ext in ("", ".zip"):
        meta_path = artifacts_root / "_shared" / f"code_{sha}{ext}.meta.json"
        if not meta_path.exists():
            continue
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return snap if isinstance(snap, dict) else None


def _list_artifacts_on_disk(run_dir: Path) -> List[Tuple[str, int]]:
    """(relpath, size_bytes) of top-level files.  Subdirs recursed one level."""
    if not run_dir.exists():
        return []
    rows: List[Tuple[str, int]] = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir).as_posix()
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        rows.append((rel, sz))
    return rows


def load_run(run_id_or_dir: Any, *, root: Path = Path("artifacts")) -> RunIntel:
    """Load one run by id ('lstm__varsmall__etth1__seed0') OR by direct Path.

    Raises FileNotFoundError if not found.  Other decode errors degrade silently.
    """
    if isinstance(run_id_or_dir, Path):
        run_dir = run_id_or_dir
    else:
        # treat as run_id, fall back to direct path
        as_path = Path(run_id_or_dir)
        if as_path.exists() and as_path.is_dir():
            run_dir = as_path
        else:
            run_dir = root / str(run_id_or_dir)

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run not found: {run_dir}")
    manifest = _safe_yaml_load(run_dir / "manifest.yaml") or {}
    run_id = manifest.get("run_id") or run_dir.name
    return RunIntel(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        curves_epochs=_load_curves(run_dir),
        curves=_load_curve_series(run_dir),
        predictions_sample=_load_predictions_sample(run_dir),
        event_kinds=_event_kinds(run_dir),
        code_snapshot=_resolve_code_snapshot(manifest, root),
        artifacts_on_disk=_list_artifacts_on_disk(run_dir),
    )


def _run_dirs(root: Path) -> List[Path]:
    """Glob artifacts/<run_id>/ skipping anything starting with '_' (e.g. _shared)."""
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")])


def _parse_created_at(value: Any) -> Optional[datetime]:
    """Parse manifest.created_at, returning an aware datetime or None.

    - non-str / empty / unparseable → None (caller skips recency filter, no crash)
    - naive iso string → attach LOCAL timezone (legacy manifests were produced
      by `datetime.now().isoformat(...)` which is local naive)
    - aware iso string → leave as-is (already self-describing)
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        # `datetime.astimezone()` on a naive datetime treats it as local time
        # and attaches the local tz. Matches the historical `datetime.now()` producer.
        ts = ts.astimezone()
    return ts


def iter_recent_runs(
    root: Path = Path("artifacts"),
    *,
    since_days: Optional[int] = None,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    variant: Optional[str] = None,
) -> List[RunIntel]:
    """Yield RunIntel for matching runs.  Optional filter by created_at recency.

    Filtering by dataset/model/variant is by manifest.yaml — costs an extra yaml read per run,
    but list-runs/compare already do this pattern.
    """
    cutoff: Optional[datetime] = None
    if since_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    out: List[RunIntel] = []
    for d in _run_dirs(root):
        # Cheap pre-check by directory name pattern (when present)
        # Otherwise read manifest to filter
        try:
            run = load_run(d, root=root)
        except FileNotFoundError:
            continue
        if cutoff is not None and run.created_at:
            ts = _parse_created_at(run.created_at)
            if ts is not None and ts < cutoff:
                continue
        if dataset:
            ds = run.manifest.get("dataset") or {}
            if ds.get("alias") != dataset:
                continue
        m = run.manifest.get("model") or {}
        if model and m.get("alias") != model:
            continue
        if variant and m.get("variant") != variant:
            continue
        out.append(run)
    return out


# ============================================================================
# Markdown renderer
# ============================================================================

def _f(v: Any, prec: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}" if isinstance(v, float) else str(v)
    return str(v)


# Flags that are toggle-only (store_true / store_false) — no value follows.
# argv layout can interleave these; we must NOT prepend "=<path>" to them.
_TOGGLE_FLAGS = {"--quiet", "-q", "--strict", "--verbose", "-v"}


def _format_argv(argv: Optional[List[str]]) -> List[str]:
    """Pick the subcommand + main --flag pairs.  Drop script paths and long values.

    argv[0] is usually the script path (e.g. E:\\rundown\\rundown\\cli.py) — skip.
    argv[1] is usually the subcommand (train / experiment / inspect / ...) — keep.
    argv[2:] is usually --flag VALUE pairs — keep, truncate values that look like paths.

    Boolean `store_true` flags (--quiet, --verbose, ...) accept no value, so don't
    get confused with flag-value pairs even if the next token happens to be path-like.
    """
    if not argv or not isinstance(argv, list):
        return ["(unknown)"]
    toks = [t for t in argv if isinstance(t, str)]
    out: List[str] = []
    # Heuristic: first non-flag token after a script-looking argv[0] is the subcommand.
    start = 1 if toks and (toks[0].endswith(".py") or "\\" in toks[0] or "/" in toks[0]) else 0
    if start < len(toks) and not toks[start].startswith("--"):
        out.append(toks[start])
        start += 1
    for tok in toks[start:]:
        if len(out) > 12:
            break
        if tok.startswith("--") or tok in _TOGGLE_FLAGS:
            out.append(tok)
        elif out and out[-1] not in _TOGGLE_FLAGS and out[-1].startswith("--"):
            # value for the previous flag — shorten if it looks like a real path
            if "\\" in tok or "/" in tok or tok.endswith(".py"):
                out[-1] = out[-1] + "=<path>"
            else:
                out.append(tok)
    return out or ["(empty)"]


def _format_factory_kw(kw: Dict[str, Any]) -> str:
    if not kw:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in kw.items())


def _artifact_summary(rows: List[Tuple[str, int]]) -> str:
    if not rows:
        return "- (none)"
    lines = []
    for rel, sz in rows:
        size_kb = sz / 1024.0
        if size_kb < 1:
            size = f"{sz}B"
        elif size_kb < 1024:
            size = f"{size_kb:.1f}KB"
        else:
            size = f"{size_kb / 1024:.2f}MB"
        lines.append(f"- `{rel}` ({size})")
    return "\n".join(lines)


def render_run_markdown(run: RunIntel) -> str:
    """Render one RunIntel as a README-style markdown section."""
    m = run.manifest or {}
    ds = m.get("dataset") or {}
    mdl = m.get("model") or {}
    cfg = m.get("config") or {}
    env = m.get("env") or {}
    metrics = m.get("metrics") or {}
    factory = mdl.get("factory") or {}

    # ---- Purpose (heuristic from facts present in manifest) ----
    target_col = ds.get("target_col", "?")
    in_dim = (factory.get("kwargs") or {}).get("in_dim", "?")
    horizon = run.predictions_sample.shape[1] if run.predictions_sample else "?"
    purpose = (
        f"Predict `{target_col}` {horizon} steps ahead on `{ds.get('alias', '?')}` "
        f"using `{mdl.get('alias', '?')}"
        + (f"/{mdl.get('variant')}" if mdl.get("variant") else "")
        + "`. Captured for reproducibility — `manifest.yaml` is the source of truth."
    )

    # ---- Setup ----
    splits = ds.get("splits") or {}
    splits_str = (
        f"train={splits.get('train', '?')} / val={splits.get('val', '?')} / "
        f"test={splits.get('test', '?')}"
    ) if splits else "(unsplit)"
    opt = cfg.get("optimizer") or {}
    opt_str = f"{opt.get('kind', 'adam')}(lr={opt.get('lr', '?')})" if opt else "(none)"

    setup_lines = [
        f"- **Dataset**: `{ds.get('alias', '?')}` "
        f"({(ds.get('source') or {}).get('kind', '?')}, target `{target_col}`)",
        f"  - splits: {splits_str}",
        f"  - source: `{(ds.get('source') or {}).get('path', '?')}`",
        f"- **Model**: `{mdl.get('alias', '?')}`"
        + (f" / variant=`{mdl.get('variant')}`" if mdl.get("variant") else ""),
        f"  - factory: `{factory.get('target', '?')}`",
        f"  - kwargs: {_format_factory_kw(factory.get('kwargs') or {})}",
        f"  - in_dim={in_dim}",
        f"- **Config**: epochs_max={cfg.get('epochs_max', '?')}, "
        f"steps_per_epoch={cfg.get('steps_per_epoch', '?')}, "
        f"optimizer={opt_str}",
        f"- **Seed**: {m.get('seed', '?')}",
    ]
    # C4: trainer 字段 (CLI / experiment YAML / model.default_trainer 任意一处).
    # 仅当 manifest.config.trainer 非空时显示, 避免 C3 前的旧 artifact 加空行.
    trainer_name = run.trainer_name
    if trainer_name:
        setup_lines.append(f"- **Trainer**: `{trainer_name}`")

    # ---- Environment ----
    git_short = run.git_commit or "?"
    argv_str = " ".join(_format_argv(env.get("argv")))
    snap = run.code_snapshot or {}
    env_lines = [
        f"- rundown {env.get('rundown_version', '?')} / "
        f"python {env.get('python', '?')} / "
        f"torch {env.get('torch', '?')} / "
        f"numpy {env.get('numpy', '?')}",
        f"- platform: `{env.get('platform', '?')}`",
        f"- hostname: `{env.get('hostname', '?')}`",
        f"- git: `{git_short}`"
        + (" (dirty working tree)" if env.get("git_dirty") else ""),
        f"- argv: `rundown {argv_str}`",
        f"- cwd: `{env.get('cwd', '?')}`",
    ]
    if snap:
        sha = snap.get("sha256_short", "?")
        env_lines.append(
            f"- code snapshot: `{snap.get('format', '?')}` "
            f"{snap.get('num_files', '?')} files, sha256 `{sha[:SHA8]}`, "
            f"{snap.get('size_bytes', 0) / 1024:.1f}KB "
            f"→ `_shared/code_{sha}/`"
            if snap.get("format") == "tar"
            else f"- code snapshot: `{snap.get('format', '?')}` "
            f"{snap.get('num_files', '?')} files, sha256 `{sha[:SHA8]}`"
        )

    # ---- Results ----
    # C4: 通用 metrics 表 — 所有 manifest.metrics 键都列出来.
    # 已知键 (best_val_mse / best_val_loss / epochs_to_best / epochs_actual /
    # val_mse_final) 按固定顺序在最前, 其余按字母升序. 旧 artifact 仍是这套
    # 顺序, 视觉上对齐 README 样例的表格.
    results_lines = [
        "| metric | value |",
        "| --- | --- |",
    ]
    if metrics:
        known_priority = ("best_val_mse", "best_val_loss", "epochs_to_best",
                          "epochs_actual", "val_mse_final")
        seen = []
        for k in known_priority:
            if k in metrics:
                results_lines.append(f"| {k} | {_f(metrics[k])} |")
                seen.append(k)
        for k in sorted(metrics.keys()):
            if k in seen:
                continue
            results_lines.append(f"| {k} | {_f(metrics[k])} |")
    else:
        results_lines.append("| (no metrics) | - |")

    curves_section = ""
    if run.curves_epochs:
        rows = [
            "| epoch | train_loss | val_mse | best? |",
            "| --- | --- | --- | --- |",
        ]
        for p in run.curves_epochs:
            star = "BEST" if p.is_best else ""
            rows.append(
                f"| {p.epoch} | {_f(p.train_loss)} | {_f(p.val_mse)} | {star} |"
            )
        curves_section = "### Loss curves\n\n" + "\n".join(rows)
    elif run.curves:
        # C4: 通用曲线表. 旧 ``train_loss`` / ``val_mse`` 仍能用 — 已映射成
        # ``curves_epochs`` 走 legacy 表, 这里只为 custom trainer (e.g. joint /
        # staged / custom_loss) 的任意 ``val_loss`` / ``stage1_train_loss`` 等
        # 走通用路径.
        #
        # 渲染规则:
        #   * 1 列: ``epoch`` (索引, 1-based, 兼容旧格式).
        #   * N 列: series 名 (按字母排序, deterministic).
        #   * 列数 cap 到 8 + 1 ellipsis note (防止几十列 metrics 撑爆表格).
        #   * 缺失位置填 ``-``.
        series_names = sorted(run.curves.keys())
        cap = 8
        truncated = len(series_names) > cap
        shown = series_names[:cap]
        max_len = max((len(v) for v in run.curves.values()), default=0)
        header_cells = ["epoch"] + shown
        rows = ["| " + " | ".join(header_cells) + " |",
                "| " + " | ".join("---" for _ in header_cells) + " |"]
        for i in range(max_len):
            cells = [str(i + 1)]
            for name in shown:
                series = run.curves.get(name) or []
                cells.append(_f(series[i]) if i < len(series) else "-")
            rows.append("| " + " | ".join(cells) + " |")
        curves_section = "### Curves\n\n" + "\n".join(rows)
        if truncated:
            curves_section += (
                f"\n\n_… {len(series_names) - cap} more series not shown "
                f"(cap={cap}; see `curves.safetensors`)._"
            )

    sample_section = ""
    if run.predictions_sample:
        s = run.predictions_sample
        sample_section = (
            f"### Test predictions sample (shape={s.shape})\n\n"
            f"- y_hat[0, :5] = `{s.y_hat_first5}`\n"
            f"- y_true[0, :5] = `{s.y_true_first5}`\n\n"
            f"_values are z-normalised; to denormalise, apply the dataset's inverse-std_"
        )

    events_section = ""
    if run.event_kinds:
        ev_rows = ["| kind | count |", "| --- | --- |"]
        for k, n in sorted(run.event_kinds.items()):
            ev_rows.append(f"| {k} | {n} |")
        events_section = "### Events\n\n" + "\n".join(ev_rows)

    # ---- Evaluate ----
    eval_lines: List[str] = []
    if run.git_commit and env.get("git_dirty"):
        eval_lines.append(
            "> _Note: git was dirty at run time — `git checkout` may need `git stash` first._"
        )
        eval_lines.append("")  # blank line between blockquote and code fence
    # Use as_posix() so cross-platform copy-paste works (Windows backslashes
    # render fine in py but break bash / POSIX shells).
    run_dir_posix = run.run_dir.as_posix()
    run_parent_posix = run.run_dir.parent.as_posix()
    eval_lines += [
        "```python",
        "from rundown.inspect_run import load_run, render_run_markdown",
        f"run = load_run(r'{run_dir_posix}')",
        "print(render_run_markdown(run))",
        "print(run.manifest['metrics'])",
        "```",
        "",
        "Or re-train with identical code:",
        "",
        "```bash",
        f"git checkout {run.git_commit or '<commit>'}",
        f"uv run rundown train --dataset {ds.get('alias', '?')} "
        f"--model {mdl.get('alias', '?')} "
        + (f"--variant {mdl.get('variant')} " if mdl.get("variant") else "")
        + f"--seed {m.get('seed', '?')} "
        f"--epochs-max {cfg.get('epochs_max', '?')} "
        f"--out-dir {run_parent_posix}",
        "```",
    ]
    eval_section = "## How to evaluate\n\n" + "\n".join(eval_lines)

    # ---- Artifacts on disk ----
    artifacts_section = (
        "## Artifacts on disk\n\n" + _artifact_summary(run.artifacts_on_disk)
    )

    # ---- Compose (each section self-contains its content; join with blank lines) ----
    parts = [
        f"# Run: `{run.run_id}`",
        f"_{run.created_at or 'unknown time'}_",
        "## Purpose\n\n" + purpose,
        "## Setup\n\n" + "\n".join(setup_lines),
        "## Environment\n\n" + "\n".join(env_lines),
        "## Results\n\n" + "\n".join(results_lines),
        curves_section,
        sample_section,
        events_section,
        eval_section,
        artifacts_section,
    ]
    # filter empty parts, then join with a single blank line so markdown renders cleanly
    return "\n\n".join(p.rstrip() for p in parts if p).rstrip() + "\n"


# ============================================================================
# CLI
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    # Windows PowerShell 5.1 default is cp936/gbk — force stdout/stderr to utf-8
    # so markdown renders cleanly.  Safe no-op on POSIX + Windows PS 7+.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(
        prog="rundown inspect",
        description=(
            "Read rundown run artifact(s) and emit README-style markdown. "
            "Read-only. Stream to a file with `rundown inspect ... > run.md`."
        ),
    )
    parser.add_argument("run_id", nargs="?", help="Run id (e.g. lstm__varsmall__etth1__seed0) OR path.")
    parser.add_argument(
        "--root", default="artifacts", type=Path,
        help="Artifacts root (default: ./artifacts)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Iterate all runs under --root.",
    )
    parser.add_argument(
        "--since-days", type=int, default=None,
        help="Filter --all to runs created within last N days.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Filter --all by dataset alias.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Filter --all by model alias.",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Filter --all by variant.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress / warn on stderr (stdout markdown is unchanged).",
    )

    args = parser.parse_args(argv)

    # ---- Mode dispatch ----
    if args.run_id and args.all:
        print("ERROR: pass EITHER <run_id> OR --all, not both", file=sys.stderr)
        return 2

    if not args.run_id and not args.all:
        print("ERROR: pass <run_id> or --all", file=sys.stderr)
        return 2

    if args.all:
        runs = iter_recent_runs(
            args.root,
            since_days=args.since_days,
            dataset=args.dataset,
            model=args.model,
            variant=args.variant,
        )
        if not runs:
            target = f" under {args.root}"
            if args.since_days:
                target += f" within last {args.since_days} days"
            print(f"(no runs match{target})", file=sys.stderr)
            return 1
        if not args.quiet:
            print(
                f"# Found {len(runs)} run(s) under {args.root}"
                + (f" (last {args.since_days} days)" if args.since_days else ""),
                file=sys.stderr,
            )
        for i, run in enumerate(runs):
            if i > 0:
                print("\n---\n")  # separator
            print(render_run_markdown(run))
        return 0

    # single run
    try:
        run = load_run(args.run_id, root=args.root)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"# Inspected {run.run_dir.as_posix()}", file=sys.stderr)
    print(render_run_markdown(run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
