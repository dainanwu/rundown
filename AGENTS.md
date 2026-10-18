# rundown — Agent Instructions

这一份是给 AI Agent 看的项目约束与上手指引. **不是**给最终用户的产品 README.
最终用户看 [README.md](README.md).

> **TL;DR (agent 视角):**
> 1. `uv sync --extra dev` 一行初始化 (建 `.venv` + 装 5 个运行时依赖 + pytest).
> 2. 注册新数据集 / 模型 / trainer → 改 `configs/{datasets,models,trainers,experiments}.yaml`, 不动训练脚本.
> 3. 跑训练 → `uv run rundown train --dataset <alias> --model <id>`. 产物自动落到 `artifacts/<run_id>/`.
> 4. 跨运行抓指标 → `uv run rundown list-runs --format json`. 不要 try/except 吞 ID 校验错误.
> 5. 任何改动前必读 §1-5 (硬约束) 与 §6 (契约/约定).

## 0. Quick Start

首次拿到仓库, 三步走:

```bash
cd F:\rundown                                # 仓库根, configs/* 在这里, 路径相对此假设
uv sync --extra dev                          # 建 .venv + 装 5 个运行时依赖 + pytest
uv run rundown train --dataset etth1 --model lstm --epochs-max 5
uv run pytest                                # 全套测试, 跑通门槛之一
```

optuna extra 保留给 Phase 2 (智能超参搜索). 当前 `experiment run` 只支持 cartesian 笛卡尔积扫描, 不需要 optuna — 见 §6.8.

不用 uv:

```bash
pip install -e ".[dev,optuna]"               # dev = pytest; optuna 占位, 当前未启用
python -m rundown.cli train --dataset etth1 --model lstm
```

### 跑通门槛 (CI / 本地都按这个验)

- `uv run pytest` 全绿 (不依赖网络 / 大文件 / GPU).
- `uv run rundown --help` 不报错, 8 个子命令都列出来 (prepare / train / experiment / read-run / list-runs / compare / inspect / analyze).
- `uv run rundown train --dataset my_data --model my_net` 跑通 (用自己的 YAML 时也保证这一步工作).

### Agent 会话期临时目录约定

`.scratch/<session_id>/` 留给本会话的探针 / 草稿代码 —— rundown 不读不写, 也不进 git. 与其把 probe 脚本扔 `rundown/` 或 `tests/`, 不如放这里.

---

## 1. Module Boundaries

依赖方向 (C1–C4 trainer layer): `engine` import `session`; `session` import `artifact`;
`trainers` import `engine`; `experiment` 编排; `cli` 适配; 用户拥有自己的 models / trainers.

| 目录 / 文件 | 允许 import | 禁止 import | 谁可以写 | 谁可以读 |
|---|---|---|---|---|
| `rundown/*.py` 内部 | stdlib, numpy, pandas, torch, pyyaml, safetensors | 用户项目代码; 避免互相循环 | rundown 自身 | 全部 |
| `rundown/cli.py` | rundown 其他子模块 | 用户项目代码 | rundown 自身 | 全部 |
| `rundown/session.py` | rundown.artifact, rundown._env, rundown.events | engine / trainers / registry / cli (单向边界) | rundown 自身 | 全部 |
| `rundown/trainers.py` | rundown.engine (GradientStrategy 适配), rundown.ids | 用户代码 | rundown 自身 | 全部 |
| `rundown/agent/` (Phase 2) | rundown.artifact, rundown.registry | `matplotlib`, `seaborn`, `plotly` (lint 禁) | 用户 | 全部 |
| `examples/custom_trainers/` | `rundown.trainers.TrainResult` (单点例外) | 其它 rundown 子模块; **模型层 import rundown 仍禁止** | 用户 | 全部 |
| `configs/trainers.yaml` (可选) | 无 | 无 | 用户 | 全部 |
| `artifacts/` | 无 | 无 | 仅 rundown.artifact 写入 | 全部 |
| `configs/` | 无 | 无 | 用户 | 全部 |
| `data/cas/` (Phase 2) | 无 | 无 | 仅 rundown.datasets.cas | 全部 |

CLI 入口位于 `rundown/cli.py:main`, 由 `pyproject.toml` `[project.scripts]` 注册为 `rundown` 命令.
所有命令**同时**支持 CLI / `from rundown.cli import main(argv)` / Python 子模块调用.

## 2. Anti-patterns

- ❌ 把训练逻辑写进 `rundown/cli.py`. CLI 只做参数解析 + 调度, 训练在 `rundown/engine.py`（内置）或 `rundown/trainers.py`（自定义）.
- ❌ 让用户模型 `import rundown`. rundown 调用用户的模型, 不反过来.
- ❌ 用 `pickle` / `torch.save` 保存任何东西. 用 `safetensors` (见 `rundown/artifact.py`).
- ❌ 把 dataset 内容写进 git. CAS 路径在 `data/cas/`, 必须 gitignore (`.gitignore` 已含).
- ❌ hard-code 模型 / 数据集 / device 任何之一. 一律从配置 + 环境变量读.
- ❌ 手编 `manifest.yaml`. 改完会与 `events.jsonl` 漂移; 用 `rundown read-run --write-back` 走工具.
- ❌ 在 `rundown/agent/` (Phase 2) import 任何绘图库. 文本是 ground truth.
- ❌ 用 try/except 吞掉 ID 校验错误. 让 `ValueError` 传上去 (来自 `rundown.ids.assert_valid`).
- ❌ 合并 `experiment.py` 与 `engine.py` (职责不同). 编排 vs 训练语义, 各自独立.
- ❌ 在 `GradientStrategy` 里加 loss / stage / joint / 多模型参数. 这些走 custom trainer (`rundown/trainers.py` 协议).
- ❌ 让 trainer 绕过 `RunSession` 手写 manifest / env / events. 训练产物一律走 `session.write_*` / `session.finalize` / `session.fail`.
- ❌ 把 `--kwargs` 当 trainer 配置通道. trainer 配置走 `--trainer-config` / `experiments.yaml.trainer_config`; `--kwargs` 只触达模型工厂.
- ❌ 让用户 **模型** import rundown; 只有 trainer 集成代码可 import `rundown.trainers.TrainResult` (单点例外).

## 3. Naming Conventions

| 类型 | 正则 |
|---|---|
| `dataset_alias` | `^[a-z][a-z0-9_]{2,40}$` |
| `model_id`      | `^[a-z][a-z0-9_]{2,40}$` |
| `trainer_alias` | `^[a-z][a-z0-9_]{2,40}$` |
| `run_id`        | `^[a-z][a-z0-9_]+__[a-z][a-z0-9_]+__seed\d+$` |

任何 ID 写入前必须 `rundown.ids.assert_valid` 校验. 不合法抛 `ValueError`, 不静默回落.
定义/校验集中在一处 (`rundown/ids.py`), 不要在文件里手写正则.

`factory.target` 必须是 `module.attr` 形式 (例如 `myproj.models.Net`), 不接受 `:` 或裸字符串.
注册时立即 `importlib.import_module` + `getattr` 校验.

`run_id` 由 CLI 按 `{model}[__var{variant}]__{dataset}__seed{N}` 自动生成. **不要**自己解析或硬编码格式
——从 `manifest.run_id` 字段读.

## 4. Test Requirements

- 每加一个 public 函数, 配一个 test.
- `tests/test_smoke.py` 必须能 `pytest` 跑通, 不依赖网络 / 大文件 / GPU.
- 用户写的 `myproj/models/*.py` 不进 rundown 测试, 那是用户的责任.
- 测试夹具: `tests/fixtures.py` 提供 `ToyNet` (最小模型) + `RandomProvider` / `build_random_providers` (最小 BatchProvider) + `DummyTrainer` / `IncompleteTrainer` (trainer 协议反例/正例) —— 写 smoke 测试时复用, 不要重新造.
- 测试基线: `pytest` 共 213 个 (210 passed + 3 skipped) — `tests/` 内 209 (`test_smoke.py` 77 / `test_facts.py` 15 / `test_inspect_run.py` 18 / `test_session.py` 10 / `test_time_and_metrics.py` 12 / `test_trainers.py` 31 / `test_experiment_trainer.py` 31 / `test_custom_trainers.py` 15) + `examples/baseline/test_baseline.py` 4.
- 跑通门槛 (CI 把关): `pytest` 全绿 + `rundown --help` 不报错 + 单次 `rundown train` 端到端跑通.

## 5. Forbidden Imports (lint rule, Phase 2 加 ruff config)

- `pickle` / `cPickle`
- `torch.save` (在 artifact 写路径里)
- `matplotlib.pyplot` / `seaborn` / `plotly` (在 `rundown/agent/` 与 `rundown/cli.py`)

理由对应 README §「设计哲学」: 5 个运行时依赖 (torch / numpy / pandas / pyyaml / safetensors) 到此为止.
任何新依赖都得先解释为什么不能不开新包.

---

## 6. Contracts & Conventions (an agent 容易漏的)

### 6.1 BatchProvider 契约 (用户数据加载器要满足)

```python
provider(split: str) -> {
    "x":          torch.Tensor[B, T, F],
    "y":          torch.Tensor[B, T],   # engine 会切成 [:, warm:warm+yhat.shape[1]]
    "warm_start": int,
}
```

注册入口 (`configs/datasets.yaml`):

```yaml
datasets:
  my_data:
    provider_factory: myproj.data.build_providers   # module.attr, 不是裸字符串
```

builder 函数签名 (集中约定在 `rundown/registry.py`):

```python
def build_providers(source_path, splits: dict, target_col: str | None, **overrides) -> dict:
    return {"train": ..., "val": ..., "test": ...}   # 必须三键, 缺一即 ValueError
```

splits 比例求和 = 1.0, 各比例 ∈ [0, 1]; registry 在 `__post_init__` 强校验.

参考实现 `examples/baseline/data.py:SlidingWindowProvider`; smoke 测试用 `tests/fixtures.py:RandomProvider`.

### 6.2 模型 forward 契约

```python
def forward(self, x: torch.Tensor[B, T, F], warm_start: int = 0) -> torch.Tensor[B, T']:
    # T' = T - warm_start; engine 把 y 切到同样长度与你的输出对齐
```

详见 `examples/baseline/model.py:SequenceLSTM`. 不满足这个签名 → engine 反射调用就崩.

### 6.3 字段归属与覆盖优先级

- `models.{alias}.factory.kwargs` < `models.{alias}.variants[name]` < CLI `--kwargs KEY=VAL` (后者覆盖前者).
- `models.{alias}.default_trainer` < `experiments.{name}.trainer` < CLI `--trainer` (后者覆盖前者).
  默认 trainer 是内置 `gradient`; `trainer_config` 与 `--kwargs` 走不同通道, **不互通**.
- `rundown/defaults.py` 是单次 `train` 的兜底 (LR / batch_size / epochs_max / DEFAULT_OPTIMIZER).
  sweep 时由 `configs/experiments.yaml` 覆盖 defaults, 再由 `--kwargs` / `--trainer-config` 覆盖 experiments.
- ❌ `configs/models.yaml` 里写 `optimizer` / `epochs_max` / `dataset` / `trainer_config` —— 这些会**静默忽略**, 是历史字段.
  `trainer_config` 不属于 models (那是 trainer 训练配置, 不是模型结构).
  改边界就改 yaml, 不用改模型.

### 6.4 命令行 flags 容易踩的

- `--out-dir` 与 `--run-dir` 互斥 (CLI argparse 已校验). `--run-dir` 跳过 `_shared/code_*/` 共享池.
- `--no-snapshot` 关代码快照 —— 默认开, 不推荐关 (一年后找不到当时代码).
- `--quiet` / `-q` 在任何子命令位置都生效 (main() 从 sys.argv 手动抓, 绕 argparse 限制).
- `--kwargs KEY=VALUE` 类型自动判别 `int / float / bool / str`. **只触达模型工厂**; optimizer LR 走 `--lr` 而不是 `--kwargs lr=...`.
  不存在的 key → factory 自己抛 `TypeError` (不静默吞, 符合 README §「拒绝 3: 不擅自替你修」).
- `--trainer NAME` 选 trainer; NAME 是 alias (走 `configs/trainers.yaml` 或内置 `gradient`) 或 dotted path (`myproj.trainers.X`).
  优先级: CLI `--trainer` > `experiments.yaml.trainer` > `models.yaml.default_trainer` > 内置 `gradient`.
- `--trainer-config <json-or-yaml-file>` 透传 trainer 配置. 接受 JSON 字符串 (`'{"huber_delta":1.0}'`)
  或 YAML/JSON 文件路径 (`examples/custom_trainers/custom_loss.yaml`). 内部 JSON 校验, 失败 `ValueError`.
  trainer 训练配置一律走这里, 不走 `--kwargs`.
- `experiment run <name> --trainer NAME` 覆盖 experiment YAML 里的 `trainer` 字段 (sweep 时逐 trial 透传).
- `list-runs --metric NAME` 指定表格里要展示的指标列 (默认 `best_val_mse`).
- `--format json` 仅 `list-runs` 当前保证可用; 其它 verb 在 README §「Agent 调用契约」有说, 别假设.

### 6.5 stderr events 解析

```
2026-10-16T21:23:35  step_completed  {"epoch": 2, "step": 10, "train_loss": 0.42, "grad_norm": 1.5}
```

字段分隔是**双空格**, 复杂值 (含空格的字符串 / list / 嵌套 dict) 走 JSON 不会破拆分:

```python
t_iso, kind, fields_json = line.rstrip().split("  ", 2)
fields = json.loads(fields_json)
```

不要从 `events.jsonl` 推断运行状态 —— 它是事件流, 不是结构定义. 真实状态读 `manifest.yaml`.
`--quiet` 压 stderr 但 events.jsonl 仍写、产物完整 —— CI / Agent 用 `2>/dev/null` 不是 `--quiet`.

`RunSession` 拥有事件生命周期: `start()` 时 `set_active_log` 并 emit `run_started`;
训练中允许 `emit(kind, **fields)` 写 `epoch_completed` / `step_completed` / `stage_started` 等
(trainer 自定义事件名也行, 文本流不限 schema); `finalize()` emit `run_completed`;
`fail(exc)` emit `run_failed`. trainer 不直接调 `set_active_log` / `clear_active_log`.
events 是模块全局状态, **不支持同进程嵌套/并发 session** (多 session 会串, 需要并发 → 多进程).

### 6.6 代码快照布局

```
artifacts/
├── _shared/
│   ├── code_<sha8>/          # git 仓库: git archive 仅跟踪文件
│   ├── code_<sha8>.meta.json # tar 模式元数据; sweep 写, 单次 train 不写
│   ├── code_<sha8>.zip       # 非 git 仓库: zip 全目录 (排除 artifacts/.venv/__pycache__)
│   └── code_<sha8>.zip.meta.json   # zip 模式元数据; sweep 写, 单次 train 不写
└── <run_id>/
    ├── manifest.yaml         # 唯一真值来源
    ├── weights.safetensors
    ├── weights_best.safetensors
    ├── curves.safetensors
    ├── events.jsonl
    └── predictions/{train, val, test, test_best}.safetensors
```

`<sha8>.meta.json` 是 sweep (`rundown experiment run`) 写的, 单次 `rundown train` 不会产生.
tar 模式的 `code_<sha8>/` 是裸目录, 元数据用同级 `code_<sha8>.meta.json` 承载;
zip 模式同理, `.meta.json` 与 `.zip` 平级. 总之: meta 在 sibling, 不在 `code_<sha8>/` 内部.

读 artifacts 用 `rundown inspect <run_id>` / `rundown list-runs` / `rundown compare` —— 都是**只读**,
不写日志, 不创建文件 (除非你 `>` 重定向).
Python 入口: `from rundown.inspect_run import load_run, render_run_markdown`.

### 6.7 可复现性边界

`rundown/engine.py:_set_seed` 在 `_fit_with_session` 内被调一次, 设了 `torch.manual_seed` +
`cudnn.deterministic` + `set_num_threads(1)` + `use_deterministic_algorithms`. 内置 `GradientStrategy`
靠它保证**同 seed 下 manifest 字段 / events 序列 / weights 形状完全一致**. **自定义 trainer
不会自动调用 `_set_seed`** —— trainer 自己负责 seed (见
`examples/custom_trainers/common.py:seed_everything`); 不调的话同 seed 不能保证 manifest / events /
weights 一致.
PyTorch CPU 上的 LSTM 仍有 ±5% val_mse 漂移 (CPU 无 cuDNN, 单线程只能减小不能消除漂移). 这是已知 trade-off,
不在 agent 修复范围.

### 6.8 Phase 2 占位 (不要写实现)

- `rundown/agent/` — 当前不存在目录, 文本是 ground truth 的接口约束. 看到 `from rundown.agent import ...`
  的需求时, 先确认是否真的在 Phase 2.
- `data/cas/` — `.gitignore` 已含, 没有 onboarding 数据.
- `rundown analyze --analyzer X <run_id>` — CLI stub, `analyze` 子命令会接受但不执行.
- `rundown.ids`/`rundown.registry` 等内部的 Phase 2 字段 (例如 `extends:` 隐式继承) 还没启用.
- optuna 智能超参搜索 — 当前 `experiment run` 只支持 cartesian; optuna extra 是占位, 未启用.

### 6.9 Trainer 契约与 RunSession (C2–C4)

**`Trainer` 协议** (`rundown/trainers.py`, 结构子类型, 不强制继承):

```python
class MyTrainer:
    name: str                            # e.g. "custom_loss"
    required_provider_keys: tuple        # e.g. ("x", "y", "warm_start")
    primary_metric: str                  # e.g. "best_val_loss"

    def validate(self, providers: dict) -> None: ...   # 训练前契约校验; 失败 ValueError

    def fit(self, *, model, providers, ctx, session) -> TrainResult:
        # 训练循环: 走 session.emit / session.checkpoint_best /
        # session.write_curves / session.write_predictions / session.finalize.
        ...
        session.finalize(metrics={...}, predictions_primary="test_best", ...)
        return TrainResult(metrics=..., epochs_actual=..., ...)
```

**硬契约**:

- no-arg 可构造 (`resolve_trainer` 反射 `cls()`).
- 配置走 `ctx.trainer_config` (CLI `--trainer-config` 注入); trainer 不知道有 `--trainer-config` 这个 flag.
- `validate(providers)`: 内置 `GradientTrainer` 是 no-op; 自定义 trainer 必须实现 (可 no-op, 不消耗 batch / 不污染 RNG).
- `session.finalize(...)` 必须调**恰好一次**; 返回 `rundown.trainers.TrainResult`.
- 训练产物一律走 `RunSession`; trainer 不直接调 `save_manifest` / `set_active_log` / `clear_active_log`.

**选择优先级** (CLI / 编排 / 默认 三方碰面):

1. CLI `--trainer NAME` (最高)
2. `experiments.yaml.trainer` (sweep 透传到每个 trial)
3. `models.yaml.default_trainer`
4. 内置 `gradient` (兜底, 不可被覆盖)

NAME 可以是 alias (走 `configs/trainers.yaml` 注册) 或 dotted path (`myproj.trainers.X`).
含 `.` → 视为 dotted path, 直接 import; 不含 → 视为 alias, 找不到抛 `ValueError`.

**Manifest 新字段** (`config` 段, C2+):

- `config.trainer` — alias 或完整 dotted path (写到 `manifest.config.trainer`).
- `config.trainer_config` — trainer 训练配置 dict (启动时 `json.dumps` 校验; 子进程透传时序列化).
- `config.trainer_contract` — `required_provider_keys` 列表 (实际校验用的 provider 键快照).
- `config.primary_metric` — sweep 汇总 / inspect 渲染 / `list-runs --metric` 用的指标名.

**多模型权重前缀** (joint / multi-model trainer 专属):
权重 key 加前缀防冲突 — `main.<param>` 是主模型 (manifest 评估走它),
`aux.<param>` 是辅助模型. 读回时按前缀拆. 详见 `examples/custom_trainers/joint.py` 和
`examples/custom_trainers/README.md`.

**Caveats — 上手前必读**:

1. **import / sys.path** — `python -m rundown.cli` 把 cwd 加到 `sys.path`,
   `examples.custom_trainers.X` 直接 import 即可. 走全局装的 `rundown` console script 时,
   `examples.custom_trainers` 不在搜索路径里 — 需要 `pip install -e .` 或 `PYTHONPATH=.`.
2. **代码快照只含 tracked files** — `trainers.py` 必须提交进 git, 否则 snapshot 抓不到,
   1 年后用 `_shared/code_xxx` 还原不出来.
3. **trainer.py 不许 import rundown** (除 `rundown.trainers.TrainResult` 这一个例外);
   模型也不许 import rundown. CLI / registry 反射 trainer, 不反过来.

---

## 7. 改动 / 加新东西的检查清单

加新模型 / 数据集 / 实验 / verb / trainer 之前, 自检:

1. 新增 ID (alias / trainer_alias / run_id) → 走 `rundown.ids.assert_valid`, 不手写正则.
2. `factory.target` → `module.attr` 双段点分, 单段不允许.
3. 新数据集 builder → 返回 `{train, val, test}` 三键 dict, 缺一 ValueError.
4. 新模型 → `forward(x, warm_start=0) -> [B, T']`, `T' = T - warm_start`.
5. 新 trainer → 满足 `Trainer` 协议 (name / required_provider_keys / primary_metric /
   validate / fit); `validate()` 必须实现 (可 no-op; 内置 gradient 是 no-op); 产物一律走 `RunSession`;
   `session.finalize()` 恰好一次.
6. 新 trainer 注册到 `configs/trainers.yaml` 时, alias 走 `rundown.ids.assert_valid("trainer_alias", ...)`.
7. 新 public 函数 → 配 test, 不依赖 GPU / 网络 / 大文件.
8. 新依赖 → 先回答"为什么不开新包", README 5-deps 政策是硬约束.
9. 改动完后跑: `uv run pytest` → `uv run rundown --help` → `uv run rundown train --dataset etth1 --model lstm --epochs-max 1`.

需要新增 lint / CI / pre-commit 时再打开 `Phase 2 ruff config` (README 没写现成的
`ruff.toml` —— 5 个禁 import 在这里是口头契约, 未配置钩子).
