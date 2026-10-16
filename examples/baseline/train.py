"""CLI 一键启动示例 (Phase D).

Phase D 起, examples/baseline 不再演示 library API — 它演示「标准用户
用法」: ``python -m rundown.cli train --dataset ... --model ...``. 这个
脚本只是一个 thin wrapper: 接收 RUNDOWN_OUT_DIR 环境变量 (test_baseline.py
用它隔离 tmp dir), 然后 subprocess 调 rundown.cli train.

Test artifacts 仍落在 ``RUNDOWN_OUT_DIR`` 下, run_id 由 CLI 根据
``model__dataset__seed<N>`` 自动生成.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 项目根 (rundown 包所在目录)
ROOT = Path(__file__).resolve().parents[2]

# 默认 CLI 参数. 真正的项目用法是直接:
#   python -m rundown.cli train --dataset etth1 --model lstm [--variant small] ...
EP_BASE = [
    sys.executable, "-m", "rundown.cli", "train",
    "--dataset", "etth1",
    "--model",   "lstm",
]


def main() -> int:
    env_out = os.environ.get("RUNDOWN_OUT_DIR")
    cmd = list(EP_BASE)
    if env_out:
        # 让 run 落到测试指定目录 (test_baseline.py 通过这个 env var 隔离 tmp dir)
        cmd += ["--out-dir", env_out]
    # cwd 必须是 rundown 根 (configs/datasets.yaml 等在那里)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())