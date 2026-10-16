"""环境捕获 — 自动写进 manifest.env.

每次 fit() 末尾调用 ``capture_env()``, 记录 git commit / python / torch /
platform / hostname / **当前代码 snapshot** (git archive 或 zip). paper
reviewer / 一年后的自己读到 manifest 就知道:
  - 这数字哪台机器跑的 (hostname + platform)
  - 用哪个 python / torch 版本
  - 哪个 git commit 提交的代码
  - cwd 是不是干净 git tree
  - **代码本身在不在 artifact 里** (code_snapshot.sha256)
"""
from __future__ import annotations
import hashlib
import io
import platform
import socket
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

# 非 git repo fallback 打包时排除的目录 / 后缀 — 防止 artifacts / .venv 误打包
_EXCLUDE_DIRS = frozenset({
    "artifacts", ".venv", "venv", "__pycache__", ".git",
    ".mypy_cache", ".pytest_cache", "node_modules", ".tox", "dist", "build",
})
_EXCLUDE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _safe_git(cwd: Path) -> Dict[str, Any]:
    """Run git rev-parse etc. 在 cwd. 任何 git 失败都返回 {} — 不抛错."""
    out: Dict[str, Any] = {}
    try:
        # git commit hash
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd), capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            out["git_commit"] = r.stdout.strip()
        # dirty / clean
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd), capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            out["git_dirty"] = bool(r.stdout.strip())
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        # git not installed or not a repo — silently omit
        pass
    return out


def snapshot_code(snapshot_dir: Path, *, cwd: Optional[Path] = None) -> Dict[str, Any]:
    """把当前代码打包到 snapshot_dir. 默认开启 (适合 rundown「产物比代码活得久」哲学).

    文件名带 sha256 短前缀 (8 hex) — 同字节代码在同一 run_dir 内天然 dedupe;
    跨 run 各存一份 (artifact self-contained, 不引共享池).

    两种模式:
      1. git repo → ``git archive --format=tar HEAD`` 解 tar 到 ``code_<sha8>/``
         (tracked files only, 干净; 排除 .gitignore 里所有东西)
      2. 非 git → zip 全目录写到 ``code_<sha8>.zip``
         (排除常见噪音: artifacts/.venv/__pycache__/...)

    任何失败 (git timeout / fs readonly) 静默返回空 dict — 跟 _safe_git 同哲学,
    不阻塞训练. caller 通过 ``env.get("code_snapshot")`` 是否为 None 判断是否落盘.

    返回 dict 写进 ``manifest.env.code_snapshot``:
      {
        "format": "tar" | "zip",
        "tracked_only": bool,
        "sha256": str,              # 完整 64-hex hash
        "sha256_short": str,        # 前 8 hex (用于 archive_path 文件名)
        "size_bytes": int,
        "num_files": int,
        "archive_path": str,        # 相对 snapshot_dir; tar 模式为 "code_<sha8>/",
                                    # zip 模式为 "code_<sha8>.zip"
      }
    """
    cwd = cwd or Path.cwd()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 1. git archive path (clean: 只打包 tracked files, 自动应用 .gitignore)
    try:
        r = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"],
            cwd=str(cwd), capture_output=True, timeout=30,
        )
        if r.returncode == 0 and r.stdout:
            tar_bytes = r.stdout
            sha256 = hashlib.sha256(tar_bytes).hexdigest()
            sha8 = sha256[:8]
            extract_dir = snapshot_dir / f"code_{sha8}"
            # 同 hash 已在 → 不重写 (dedupe 天然)
            if not extract_dir.exists():
                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
                    # Python 3.14+ 默认 filter 会拒绝 metadata; 显式 data filter 兼容老版本.
                    tf.extractall(extract_dir, filter="data")
            num_files = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
            return {
                "format": "tar",
                "tracked_only": True,
                "sha256": sha256,
                "sha256_short": sha8,
                "size_bytes": len(tar_bytes),
                "num_files": num_files,
                "archive_path": f"code_{sha8}/",
            }
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, tarfile.TarError):
        pass

    # 2. zip fallback (全目录, 排除常见噪音)
    try:
        buf = io.BytesIO()
        num_files = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(cwd.rglob("*")):
                if not path.is_file():
                    continue
                # 排除
                rel = path.relative_to(cwd)
                if any(part in _EXCLUDE_DIRS for part in rel.parts):
                    continue
                if path.suffix in _EXCLUDE_SUFFIXES:
                    continue
                zf.write(path, rel.as_posix())
                num_files += 1
        zip_bytes = buf.getvalue()
        sha256 = hashlib.sha256(zip_bytes).hexdigest()
        sha8 = sha256[:8]
        zip_path = snapshot_dir / f"code_{sha8}.zip"
        # 同 hash 已在 → 不重写
        if not zip_path.exists():
            zip_path.write_bytes(zip_bytes)
        return {
            "format": "zip",
            "tracked_only": False,
            "sha256": sha256,
            "sha256_short": sha8,
            "size_bytes": len(zip_bytes),
            "num_files": num_files,
            "archive_path": f"code_{sha8}.zip",
        }
    except (OSError, zipfile.BadZipFile):
        return {}


def capture_env(
    cwd: Optional[Path] = None,
    *,
    snapshot_dir: Optional[Path] = None,
    snapshot: bool = True,
    external_snapshot: Optional[Dict[str, Any]] = None,
    archive_rel: str = "",
) -> Dict[str, Any]:
    """Snapshot 当前进程的环境信息. 返回 dict 直接写进 manifest.env.

    Args:
        cwd: 工程根 (默认 ``Path.cwd()``). 影响 git / code snapshot 解析路径.
        snapshot_dir: code snapshot 落盘目录. None → 跳过 snapshot.
        snapshot: 主开关. False → 完全跳过 snapshot.
        external_snapshot: experiment sweep 共享池场景 — caller 已建好 snapshot,
            直接把 dict 写进 env.code_snapshot. 不再调 snapshot_code().
            dict 应含 snapshot_code() 返回的所有字段 + ``archive_path`` (相对
            caller 的 run_dir).
        archive_rel: archive_path 相对 run_dir 的前缀 (e.g. ``"../_shared"``).
            用于 snapshot 落在 out_root/_shared 时, manifest 写相对 run_dir 的路径
            让读者能 ``cd run_dir && cat ../_shared/...`` 直接读. 默认 "" →
            snapshot 在 run_dir 内.
    """
    import rundown  # late import 避免 circular
    env: Dict[str, Any] = {
        "rundown_version": rundown.__version__,
        "python": sys.version.split()[0],          # e.g. "3.10.11"
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "argv": sys.argv[:],
        "cwd": str(cwd or Path.cwd()),
    }
    try:
        import torch
        # torch.__version__ returns TorchVersion (not plain str); YAML can't
        # serialize it. Force plain str.
        env["torch"] = str(torch.__version__)
    except ImportError:
        pass
    try:
        import numpy as np
        env["numpy"] = str(np.__version__)
    except ImportError:
        pass
    env.update(_safe_git(cwd or Path.cwd()))

    # code snapshot — 默认开 (跟 rundown 哲学对齐: 产物比代码活得久)
    # 三种路径:
    #   1. external_snapshot 提供 → experiment sweep 共享池, 直接引用
    #   2. snapshot + snapshot_dir → 单次 train, 自己建 (落 snapshot_dir)
    #   3. snapshot=False → 跳过
    # archive_path: snapshot_dir 是 out_root/_shared 时, 加 archive_rel 前缀
    # 让 manifest 写相对 run_dir 的路径.
    if external_snapshot is not None:
        env["code_snapshot"] = dict(external_snapshot)
    elif snapshot and snapshot_dir is not None:
        snap = snapshot_code(snapshot_dir, cwd=cwd or Path.cwd())
        if snap:
            if archive_rel:
                snap["archive_path"] = f"{archive_rel}/{snap['archive_path']}"
            env["code_snapshot"] = snap

    return env


def read_snapshot_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    """读 sweep 入口写的 snapshot meta json. 失败返回 None (subprocess 降级到自建).

    sweep 入口把 snapshot_code() 的返回值写到 ``_shared/code_<sha8>.zip.meta.json``,
    subprocess 通过 ``--snapshot-from-pool <meta.json>`` 拿到 path, 这里读它.
    """
    import json
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "archive_path" not in data:
        return None
    return data


__all__ = ["capture_env", "snapshot_code", "read_snapshot_meta"]