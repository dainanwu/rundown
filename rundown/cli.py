"""CLI: 8 个动词. prepare / train / experiment / read-run / list-runs / compare / inspect / analyze.

MVP 状态:
  - prepare / train: 跑通骨架 (synthetic data, 见 rundown.engine).
  - experiment: sweep — subprocess 调 train, stdout 拿 run_id, 共享 _shared/ pool.
  - read-run: 读 manifest + 从 predictions/*.safetensors 算简单指标.
  - list-runs / compare: glob artifacts/*/manifest.yaml 直接读, 不建 DB.
  - inspect: read-only render run → README-style markdown 到 stdout.
  - analyze: stub, Phase 2.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _quiet_parser() -> argparse.ArgumentParser:
    """共享 quiet flag — 给 sub parser 当 parents 用."""
    q = argparse.ArgumentParser(add_help=False)
    q.add_argument("--quiet", "-q", action="store_true",
                   help="压住 stderr 上的 info 行 (events.jsonl 仍写, artifact 完整). "
                        "适合 CI / agent 自动化调用.")
    return q


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rundown",
        description="声明式运行 · 持久产物. 命令: prepare / train / experiment / read-run / list-runs / compare / inspect / analyze.",
    )
    quiet = _quiet_parser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", parents=[quiet],
                        help="加载数据集, 算 content hash, 注册到 rundown")
    pp.add_argument("--dataset", required=True)

    tp = sub.add_parser("train", parents=[quiet], help="跑一次训练")
    tp.add_argument("--dataset", required=True)
    tp.add_argument("--model", required=True)
    tp.add_argument("--variant", default=None,
                    help="选择 models.yaml 里的 variants 之一 (e.g. small/large/deep)")
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--epochs-max", type=int, default=None,
                    help="训练时长 (epoch 数). 不传 → defaults.DEFAULT_EPOCHS_MAX "
                         "(experiments.yaml 里设会覆盖 defaults)")
    tp.add_argument("--steps-per-epoch", type=int, default=None,
                    help="每个 epoch 跑多少 batch; 默认 50")
    tp.add_argument("--batch-size", type=int, default=None,
                    help="覆盖 provider 默认 batch_size")
    tp.add_argument("--lr", type=float, default=None,
                    help="覆盖 optimizer.lr (experiments.yaml.optimizer.lr 也覆盖这里)")
    tp.add_argument("--out-dir", type=Path, default=None,
                    help="artifacts 根目录; CLI 自动加 run_id 子目录")
    tp.add_argument("--run-dir", type=Path, default=None,
                    help="直接指定 run 目录, 不再嵌套 run_id. 给 experiment 等批量场景用, "
                         "与 --out-dir 互斥")
    tp.add_argument("--no-snapshot", dest="snapshot", action="store_false",
                    help="跳过 code snapshot (默认会打包当前代码进 artifacts/<run_id>/code/)")
    tp.add_argument("--snapshot-from-pool", type=Path, default=None,
                    help="experiment sweep 共享 snapshot meta.json 路径. "
                         "给定时跳过自建, 直接引用. 不传 → 自己建.")
    tp.add_argument("--kwargs", action="append", default=None, metavar="KEY=VALUE",
                    help="透传任意 kwargs 给 model factory (e.g. hidden=128 num_layers=2 lr=0.001). "
                         "可多次. 合并顺序: factory_kwargs < variants < --kwargs (后者覆盖). "
                         "optuna sweep 时绕过 variants 限制, 自由传任意超参. "
                         "key 不在 factory 签名时由 factory 抛 TypeError (不静默).")

    ep = sub.add_parser("experiment", parents=[quiet],
                        help="跑一个注册过的实验 (当前仅支持笛卡尔积 sweep)")
    exp_p = ep.add_subparsers(dest="exp_cmd", required=True)
    erun = exp_p.add_parser("run", parents=[quiet], help="跑实验")
    erun.add_argument("name", help="experiments.yaml 里的实验名")
    erun.add_argument("--out-root", type=Path, default=None)

    rp = sub.add_parser("read-run", parents=[quiet], help="读一个 run 算新指标")
    rp.add_argument("run_id")
    rp.add_argument("--root", type=Path, default=Path("artifacts"),
                    help="artifacts 根目录 (default: ./artifacts); 与 list-runs/compare/inspect 一致")
    rp.add_argument("--metric", default=None,
                    help="要算的指标名; 不传 = 打印 manifest 摘要")
    rp.add_argument("--by", default=None,
                    help="分组维度, 例如 quantile:N (N 桶)")
    rp.add_argument("--write-back", action="store_true", help="把新指标固化进 manifest")

    ap = sub.add_parser("analyze", parents=[quiet], help="Phase 2; 当前 stub")
    ap.add_argument("--analyzer", required=True)
    ap.add_argument("run_id")

    lp = sub.add_parser("list-runs", parents=[quiet],
                        help="列出 artifacts 下的所有 run, 支持按 dataset/model/variant/metric 过滤")
    lp.add_argument("--root", type=Path, default=Path("artifacts"),
                    help="artifacts 根目录 (default: ./artifacts)")
    lp.add_argument("--dataset", default=None, help="过滤 dataset.alias")
    lp.add_argument("--model", default=None, help="过滤 model.alias")
    lp.add_argument("--variant", default=None, help="过滤 model.variant")
    lp.add_argument("--above-metric", nargs=2, default=None, metavar=("NAME", "VALUE"),
                    help="只保留 metrics.NAME > VAL, e.g. --above-metric best_val_mse 2.0")
    lp.add_argument("--below-metric", nargs=2, default=None, metavar=("NAME", "VALUE"),
                    help="只保留 metrics.NAME < VAL, e.g. --below-metric best_val_mse 1.5")
    lp.add_argument("--format", choices=["table", "json"], default="table")

    cp = sub.add_parser("compare", parents=[quiet],
                        help="side-by-side diff 两个 run 的 manifest 字段")
    cp.add_argument("run_id_a")
    cp.add_argument("run_id_b")
    cp.add_argument("--root", type=Path, default=Path("artifacts"))

    ip = sub.add_parser("inspect", parents=[quiet],
                        help="读 run artifact(s), 输出 README-style markdown 到 stdout")
    ip.add_argument("run_id", nargs="?",
                    help="Run id (例如 lstm__varsmall__etth1__seed0) 或直接路径")
    ip.add_argument("--root", default="artifacts", type=Path,
                    help="Artifacts 根目录 (default: ./artifacts)")
    ip.add_argument("--all", action="store_true",
                    help="迭代 --root 下所有 run (可配 --since-days / --dataset / --model / --variant)")
    ip.add_argument("--since-days", type=int, default=None,
                    help="与 --all 同用: 只看最近 N 天的 run")
    ip.add_argument("--dataset", default=None, help="--all: 过滤 dataset.alias")
    ip.add_argument("--model", default=None, help="--all: 过滤 model.alias")
    ip.add_argument("--variant", default=None, help="--all: 过滤 model.variant")

    return p


def _err(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def _utc_now_iso() -> str:
    """UTC-now ISO 8601 with offset, second precision.

    Used for manifest.created_at and read-run --write-back.computed_at so that
    --since-days (which compares against UTC) lines up.
    """
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def _cmd_prepare(args: argparse.Namespace) -> int:
    from .registry import load_datasets
    from .ids import assert_valid
    assert_valid("dataset_alias", args.dataset)
    ds = load_datasets()
    if args.dataset not in ds:
        return _err(f"dataset {args.dataset!r} not in configs/datasets.yaml; "
                    f"have: {list(ds)}")
    entry = ds[args.dataset]
    print(f"[rundown] prepared: alias={entry.alias} "
          f"source_kind={entry.source_kind} path={entry.source_path} "
          f"target_col={entry.target_col!r}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def _load_external_snapshot(meta_path: Optional[Path], run_dir: Path) -> Optional[Dict[str, Any]]:
    """读 experiment sweep 共享 snapshot meta.json. 失败返回 None (走自建 fallback).

    archive_path 在 meta 里是相对 out_root 的; 转成相对 run_dir 后写回,
    让 manifest 读者能直接 cat 看到. 统一用 POSIX 斜杠 (YAML / shell friendly).
    """
    if meta_path is None:
        return None
    from rundown._env import read_snapshot_meta
    info = read_snapshot_meta(meta_path)
    if info is None:
        return None
    arc = Path(info["archive_path"])
    if arc.is_absolute():
        try:
            info["archive_path"] = arc.relative_to(run_dir).as_posix()
        except ValueError:
            # 跨盘符等 — 留绝对, manifest 读者自己处理
            info["archive_path"] = arc.as_posix()
    else:
        # 相对 out_root (out_root 已知: meta_path.parent.parent)
        out_root = meta_path.parent.parent
        try:
            abs_arc = (out_root / arc).resolve()
            info["archive_path"] = abs_arc.relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            info["archive_path"] = f"../{arc.as_posix()}"  # fallback
    return info


def _parse_cli_kwargs(raw: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """解析 `--kwargs KEY=VALUE` token 列表 → dict.

    值类型自动判断: int / float / bool / str. 不在 factory 签名的 key 不拦截
    (让 factory 自己抛 TypeError — 宁报错不猜).

    返回 None 表示「未传 --kwargs」(不是「传了空 dict」) —
    ModelEntry.build 把它跟 None 走同一条分支, 行为一致.
    """
    if not raw:
        return None
    out: Dict[str, Any] = {}
    for token in raw:
        if "=" not in token:
            raise ValueError(
                f"--kwargs token must be 'KEY=VALUE', got {token!r}"
            )
        key, _, val_s = token.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--kwargs token has empty key: {token!r}")
        out[key] = _coerce_scalar(val_s)
    return out


def _coerce_scalar(s: str) -> Any:
    """字符串 → int / float / bool / str (按 Python 字面量)."""
    s = s.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    # int 先, 再 float (避免 '1.0' 被解析成 int=1)
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _cmd_train(args: argparse.Namespace) -> int:
    """End-to-end: registry -> provider -> model -> engine -> artifact.

    通过 configs/{datasets,models}.yaml 的注册项, 反射出 BatchProvider 和
    model. 所有超参走 CLI 覆盖 (--epochs-max / --steps-per-epoch / --batch-size / --lr).
    """
    from .registry import load_datasets, load_models
    from .ids import assert_valid, is_valid
    from .artifact import RunManifest, RunArtifact
    from .engine import EngineConfig, GradientStrategy
    from .events import set_active_log, clear_active_log
    import time as _time

    assert_valid("dataset_alias", args.dataset)
    assert_valid("model_id", args.model)

    datasets = load_datasets()
    models = load_models()
    if args.dataset not in datasets:
        return _err(f"dataset {args.dataset!r} not registered; have: {sorted(datasets)}")
    if args.model not in models:
        return _err(f"model {args.model!r} not registered; have: {sorted(models)}")

    dataset = datasets[args.dataset]
    model_entry = models[args.model]

    # Run id 加上 variant 后缀, 避免同一 alias 多次跑冲突
    variant_suffix = f"__var{args.variant}" if args.variant else ""
    run_id = f"{args.model}{variant_suffix}__{args.dataset}__seed{args.seed}"
    if not is_valid("run_id", run_id):
        return _err(f"composed run_id invalid: {run_id!r}")

    # 拿 providers (registry 反射 provider_factory 拿 3 组)
    provider_overrides: Dict[str, Any] = {}
    if args.batch_size is not None:
        provider_overrides["batch_size"] = args.batch_size
    try:
        providers = dataset.build_providers(**provider_overrides)
    except ValueError as e:
        return _err(f"failed to build providers for {args.dataset!r}: {e}")

    # 实例化 model (variant + --kwargs 透传给 factory).
    # kwargs 合并顺序: factory_kwargs < variants[variant] < CLI --kwargs.
    try:
        model, merged_kwargs, chosen_variant = model_entry.build(
            variant=args.variant,
            extra_kwargs=_parse_cli_kwargs(args.kwargs),
        )
    except ValueError as e:
        return _err(f"failed to instantiate model {args.model!r}: {e}")

    # EngineConfig: CLI 覆盖 > defaults 单点真值
    # (models.yaml 不再含 optimizer/epochs_max — 单点真值在 defaults,
    #  sweep 时由 experiments.yaml 覆盖并通过 subprocess CLI 透传过来.)
    from .defaults import DEFAULT_STEPS_PER_EPOCH, DEFAULT_EPOCHS_MAX, DEFAULT_OPTIMIZER
    epochs_max = args.epochs_max if args.epochs_max is not None else DEFAULT_EPOCHS_MAX
    optimizer = dict(DEFAULT_OPTIMIZER)  # 拷贝, 防止 manifest.config 共享引用
    if args.lr is not None:
        optimizer["lr"] = args.lr
    lr = float(optimizer["lr"])
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else DEFAULT_STEPS_PER_EPOCH

    n_params = sum(p.numel() for p in model.parameters())
    # 三种 run_dir 解析: --run-dir 优先 (experiment 用), 否则 --out-dir 嵌套 run_id.
    if args.run_dir and args.out_dir:
        return _err("--run-dir 与 --out-dir 互斥, 只传一个")
    if args.run_dir:
        out_dir = Path(args.run_dir)
        # --run-dir 模式: 精细控制, snapshot 落 run_dir (self-contained, 不共享 _shared/)
        snapshot_target_dir: Optional[Path] = None
        snapshot_archive_rel: str = ""
    elif args.out_dir:
        out_dir = Path(args.out_dir) / run_id
        # --out-dir 模式: out_root = args.out_dir; snapshot 落 out_root/_shared 让多次
        # train 共享 1 份 (artifact 边界是 out_root, 跟 sweep 一致).
        snapshot_target_dir = Path(args.out_dir) / "_shared"
        snapshot_archive_rel = "../_shared"
    else:
        out_dir = Path("artifacts") / run_id
        snapshot_target_dir = Path("artifacts") / "_shared"
        snapshot_archive_rel = "../_shared"

    # ----- 启动打印 (friendly info block) -----
    print(f"[run] {run_id}")
    print(f"  dataset : {dataset.alias}  "
          f"(source={dataset.source_kind}, path={dataset.source_path})")
    if dataset.target_col:
        print(f"  target  : {dataset.target_col}")
    print(f"  model   : {model_entry.alias}"
          f"{f' (variant={chosen_variant})' if chosen_variant else ''}  "
          f"({n_params:,} params, {merged_kwargs})")
    train_p = providers["train"]; val_p = providers["val"]; test_p = providers["test"]
    print(f"  config  : epochs={epochs_max} steps_per_epoch={steps_per_epoch} "
          f"batch_size={train_p.batch_size} lr={lr}")
    print(f"  run dir : {out_dir}")

    # ----- Run manifest -----
    manifest = RunManifest(
        run_id=run_id,
        created_at=_utc_now_iso(),
        seed=int(args.seed),
        dataset={
            "alias": dataset.alias,
            "source": {"kind": dataset.source_kind, "path": dataset.source_path},
            "target_col": dataset.target_col,
            "splits": dataset.splits,
            "provider_factory": dataset.provider_factory,
        },
        model={
            "alias": model_entry.alias,
            "variant": chosen_variant,
            "factory": {"target": model_entry.factory_target,
                        "kwargs": merged_kwargs},
        },
        config={
            "epochs_max": int(epochs_max),
            "steps_per_epoch": int(steps_per_epoch),
            "optimizer": optimizer,
            "lr": lr,
        },
    )
    artifact = RunArtifact(run_dir=out_dir, manifest=manifest)
    artifact.save_manifest()

    # ----- Train -----
    set_active_log(out_dir, run_id)
    t0 = _time.time()
    try:
        ctx = EngineConfig(
            run_id=run_id, seed=int(args.seed), epochs_max=int(epochs_max),
            batch_size=train_p.batch_size,
            steps_per_epoch=int(steps_per_epoch),
            lr=float(lr),
            snapshot=args.snapshot,
            external_snapshot=_load_external_snapshot(args.snapshot_from_pool, out_dir),
            snapshot_target_dir=snapshot_target_dir,
            snapshot_archive_rel=snapshot_archive_rel,
        )
        result = GradientStrategy().fit(
            model, train_data=train_p, val_data=val_p, test_data=test_p,
            ctx=ctx, artifact=artifact,
        )
        dt = _time.time() - t0

        # ----- 收尾打印 -----
        primary = artifact.manifest.config.get("predictions_primary", "test_best")
        if primary in artifact.predictions:
            pred = artifact.predictions[primary]
            test_mse = float(((pred["y_hat"] - pred["y_true"]) ** 2).mean())
            test_mse_str = f"  test_mse({primary})={test_mse:.4f}"
        else:
            test_mse_str = ""

        print()
        print(f"[done] {run_id}  "
              f"epochs={result['epochs_actual']}  "
              f"best_val_mse={result['best_val']:.4f}"
              f"{test_mse_str}  duration={dt:.1f}s")
        print(f"  artifacts: {out_dir}")
        return 0
    finally:
        clear_active_log()


# ---------------------------------------------------------------------------
# read-run
# ---------------------------------------------------------------------------
def _compute_metric(predictions: Dict, metric: str, by: Optional[str]) -> Dict[str, float]:
    """从 predictions 字典算指标. 默认读 primary (manifest.config.predictions_primary)."""
    if metric == "prediction_error_std":
        y_hat = predictions.get("y_hat")
        y_true = predictions.get("y_true")
        if y_hat is None or y_true is None:
            raise ValueError(
                f"predictions missing y_hat / y_true (keys: {sorted(predictions)})"
            )
        err = y_hat - y_true
        if by is None:
            return {"total": float(err.std().item())}
        # by = "quantile:N" 把 y_hat 沿 horizon 切 N 桶
        if by.startswith("quantile:"):
            try:
                n = int(by.split(":", 1)[1])
            except ValueError as e:
                raise ValueError(f"--by quantile:N requires integer N, got {by!r}") from e
            n = max(1, n)
            n_cols = err.shape[1] if err.dim() > 1 else 1
            # 把 horizon 切 N 桶; 前 rem 桶多 1 列, 让残余均匀分布 (差最大 1).
            base = n_cols // n
            rem = n_cols - base * n  # = n_cols % n
            out = {}
            start = 0
            for i in range(n):
                size = base + (1 if i < rem else 0)
                out[f"q{i+1}/{n} (cols {start}..{start+size})"] = float(
                    err[:, start:start + size].std().item()
                )
                start += size
            out["total"] = float(err.std().item())
            return out
        # by = 其它 key 当成 metric 分组维度; 不支持时报错
        raise ValueError(f"unsupported --by {by!r}; supported: quantile:N")
    if metric in predictions:
        return {metric: float(predictions[metric].mean().item())}
    raise ValueError(f"metric {metric!r} not in predictions (keys: {sorted(predictions)})")


def _cmd_read_run(args: argparse.Namespace) -> int:
    from .artifact import RunArtifact
    from .events import emit

    run_dir = args.root / args.run_id
    if not (run_dir / "manifest.yaml").exists():
        return _err(f"no manifest at {run_dir}; not a rundown run?")

    art = RunArtifact.read(run_dir)
    if not args.metric:
        print(art.manifest.to_yaml())
        return 0

    # 默认读 manifest.config.predictions_primary (test_best); 这样与
    # manifest.metrics.best_val_mse 用同一组权重算出来的预测, 不与
    # 过拟合后的 final weights 错配.
    primary = art.manifest.config.get("predictions_primary", "test_best")
    if primary not in art.predictions:
        return _err(f"no predictions/{primary}.safetensors in {run_dir}; "
                    f"cannot compute {args.metric!r}")

    try:
        vals = _compute_metric(art.predictions[primary], args.metric, args.by)
    except ValueError as e:
        return _err(str(e))

    print(f"run: {args.run_id}  metric: {args.metric}  by: {args.by!r}  "
          f"(source: {primary})")
    print("-" * 60)
    for k, v in vals.items():
        print(f"  {k:30s}  {v:.6f}")
    print("-" * 60)

    if args.write_back:
        art.manifest.derived_metrics[args.metric] = {
            "computed_at": _utc_now_iso(),
            "by": args.by,
            "values": vals,
        }
        art.save_manifest()
        emit("metric_written_back", run_id=args.run_id, metric=args.metric)
        print(f"[rundown] written back to {run_dir / 'manifest.yaml'}")
    return 0


# ---------------------------------------------------------------------------
# experiment (Phase C)
# ---------------------------------------------------------------------------
def _cmd_experiment_run(args: argparse.Namespace) -> int:
    """rundown experiment run <name> — 跑一个注册的 sweep."""
    from .registry import load_experiments
    from .experiment import run_experiment as _run

    experiments = load_experiments()
    if args.name not in experiments:
        return _err(f"experiment {args.name!r} not registered; have: {sorted(experiments)}")
    exp = experiments[args.name]
    out_root = args.out_root or Path("artifacts")
    try:
        _run(exp, out_root=out_root)
    except (ImportError, ValueError) as e:
        return _err(str(e))
    return 0


# ---------------------------------------------------------------------------
# analyze (Phase 2 stub)
# ---------------------------------------------------------------------------
def _cmd_analyze(args: argparse.Namespace) -> int:
    print(f"[rundown] analyze: STUB. analyzer={args.analyzer!r} run_id={args.run_id!r}; "
          f"Phase 2 落地.", file=sys.stderr)
    return 3


# ---------------------------------------------------------------------------
# list-runs / compare
# ---------------------------------------------------------------------------
def _parse_metric_pair(raw: Optional[List[str]]) -> Optional[Tuple[str, float]]:
    """`['best_val_mse', '1.5']` (nargs=2 产物) -> ('best_val_mse', 1.5). None 输入 -> None."""
    if raw is None:
        return None
    if len(raw) != 2:
        raise ValueError(f"--above-metric/--below-metric 需要 'NAME VALUE', got {raw!r}")
    name, val_s = raw
    try:
        return name, float(val_s)
    except ValueError as e:
        raise ValueError(f"--above-metric/--below-metric VALUE must be float, got {val_s!r}") from e


def _cmd_list_runs(args: argparse.Namespace) -> int:
    from .cli_query import list_runs
    try:
        above = _parse_metric_pair(args.above_metric)
        below = _parse_metric_pair(args.below_metric)
    except ValueError as e:
        return _err(str(e))
    return list_runs(
        root=args.root, dataset=args.dataset, model=args.model, variant=args.variant,
        above_metric=above, below_metric=below, fmt=args.format,
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    from .cli_query import compare_runs
    return compare_runs(args.run_id_a, args.run_id_b, root=args.root)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    # 手动从 argv 抓 --quiet / -q, 让 user 放任何位置都生效
    # (argparse 全局 flag 必须出现在子命令前, 实际使用 `train ... --quiet` 更常见).
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    quiet_flag = "--quiet" in argv_list or "-q" in argv_list
    # Windows 中文 stdout/stderr 兜底: 默认 codepage 不是 utf-8,
    # argparse description / emit() 写中文会乱码. 重配成 utf-8 + errors=replace,
    # 跟 inspect_run.py:560 保持一致. 不存在 reconfigure (老 py3) 时跳过.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    # --quiet 影响 emit() stderr 写入; events.jsonl 永远写 (artifact 完整).
    from . import events as _events
    _events.set_quiet(quiet_flag)
    if args.cmd == "prepare":
        return _cmd_prepare(args)
    if args.cmd == "train":
        return _cmd_train(args)
    if args.cmd == "experiment":
        if getattr(args, "exp_cmd", None) == "run":
            return _cmd_experiment_run(args)
        parser.error(f"unknown experiment sub-cmd: {getattr(args, 'exp_cmd', None)}")
        return 2
    if args.cmd == "read-run":
        return _cmd_read_run(args)
    if args.cmd == "analyze":
        return _cmd_analyze(args)
    if args.cmd == "list-runs":
        return _cmd_list_runs(args)
    if args.cmd == "compare":
        return _cmd_compare(args)
    if args.cmd == "inspect":
        from .inspect_run import main as _inspect_main
        # re-build argv from argparse namespace → matches inspect_run's argparse
        argv: List[str] = []
        if args.run_id:
            argv.append(args.run_id)
        argv += ["--root", str(args.root)]
        if getattr(args, "all", False):
            argv.append("--all")
        if getattr(args, "since_days", None) is not None:
            argv += ["--since-days", str(args.since_days)]
        if getattr(args, "dataset", None):
            argv += ["--dataset", args.dataset]
        if getattr(args, "model", None):
            argv += ["--model", args.model]
        if getattr(args, "variant", None):
            argv += ["--variant", args.variant]
        if getattr(args, "quiet", False):
            argv.append("--quiet")
        return _inspect_main(argv)
    parser.error(f"unknown cmd: {args.cmd}")  # parser.error calls sys.exit(2)


if __name__ == "__main__":
    sys.exit(main())
