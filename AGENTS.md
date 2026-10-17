# rundown — Agent Instructions

这一份是给 AI Agent 看的项目约束, **不**是给最终用户的产品 README.
最终用户看 [README.md](README.md).

## 1. Module Boundaries

| 目录 / 文件 | 允许 import | 禁止 import | 谁可以写 | 谁可以读 |
|---|---|---|---|---|
| `rundown/*.py` 内部 | stdlib, numpy, pandas, torch, pyyaml, safetensors | 用户项目代码; 避免互相循环 | rundown 自身 | 全部 |
| `rundown/cli.py` | rundown 其他子模块 | 用户项目代码 | rundown 自身 | 全部 |
| `rundown/agent/` (Phase 2) | rundown.artifact, rundown.registry | `matplotlib`, `seaborn`, `plotly` (lint 禁) | 用户 | 全部 |
| `artifacts/` | 无 | 无 | 仅 rundown.artifact 写入 | 全部 |
| `configs/` | 无 | 无 | 用户 | 全部 |
| `data/cas/` (Phase 2) | 无 | 无 | 仅 rundown.datasets.cas | 全部 |

## 2. Anti-patterns

- ❌ 把训练逻辑写进 `rundown/cli.py`. CLI 只做参数解析 + 调度, 训练在 `rundown/engine.py`.
- ❌ 让用户模型 `import rundown`. rundown 调用用户的模型, 不反过来.
- ❌ 用 `pickle` / `torch.save` 保存任何东西. 用 `safetensors`.
- ❌ 把 dataset 内容写进 git. CAS 路径在 `data/cas/`, 必须 gitignore.
- ❌ hard-code 模型 / 数据集 / device 任何之一. 一律从配置 + 环境变量读.
- ❌ 手编 `manifest.yaml`. 改完会与 `events.jsonl` 漂移; 用 `rundown read-run --write-back` 走工具.
- ❌ 在 `rundown/agent/` (Phase 2) import 任何绘图库. 文本是 ground truth.
- ❌ 用 try/except 吞掉 ID 校验错误. 让 ValueError 传上去.

## 3. Naming Conventions

| 类型 | 正则 |
|---|---|
| `dataset_alias` | `^[a-z][a-z0-9_]{2,40}$` |
| `model_id`      | `^[a-z][a-z0-9_]{2,40}$` |
| `run_id`        | `^[a-z][a-z0-9_]+__[a-z][a-z0-9_]+__seed\d+$` |

任何 ID 写入前必须 `rundown.ids.assert_valid` 校验. 不合法抛 ValueError, 不静默回落.

`factory.target` 必须是 `module.attr` 形式 (例如 `myproj.models.Net`), 不接受 `:` 或裸字符串.

## 4. Test Requirements

- 每加一个 public 函数, 配一个 test.
- `tests/test_smoke.py` 必须能 `pytest` 跑通, 不依赖网络 / 大文件 / GPU.
- 用户写的 `myproj/models/*.py` 不进 rundown 测试, 那是用户的责任.
- 跑通门槛: `pytest` 全绿 + `rundown --help` 不报错 + `rundown train --dataset my_data --model my_net` 跑通.

## 5. Forbidden Imports (lint rule, Phase 2 加 ruff config)

- `pickle` / `cPickle`
- `torch.save` (在 artifact 写路径里)
- `matplotlib.pyplot` / `seaborn` / `plotly` (在 rundown/agent/ 与 rundown/cli.py)
