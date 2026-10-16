# rundown

**声明式运行 · 持久产物 · 为 Agent 而生**

rundown 把"一个数据集 + 一个模型 + 一个实验"的 YAML 描述变成一次训练运行,
以及一个明年还能完整读懂的目录. 一次 `rundown train` 命令搞定 dataset / model / 超参 / checkpoint 全套.

> "为 Agent 而生"是长期承诺. 本仓库实现的是前两条.
> 完整 Agent 工具集在 Phase 2.

---

## 设计哲学

rundown 不是 MLflow 替代品, 不是 PyTorch 包装, 不是实验追踪器.
它是一个 **artifact 自描述框架**——一次跑完留下的目录本身, 一年后还能让 reviewer
完整复算.

下面是 3 个承诺 + 3 个拒绝 + 1 个物量约束. 后面所有章节都在印证这些原则.

### 3 个承诺

**1. 一年后还能复算.**

跑完的瞬间, 复算这件事的所有信息都在 run 目录里.
不依赖 git log, 不依赖外部 DB, 不依赖人记.
"表 4 第二行那个数字是哪个 split / 哪个 seed / 哪份代码 / 哪台机器"——
拷走目录就能答完.

**2. manifest 是事实本身.**

`manifest.yaml` 写盘即合同. `list-runs` / `compare` / `read-run` 全走文件系统,
没有 server, 没有 schema migration, 没有「回查」步骤.
manifest 既是数据, 也是索引.

**3. 让 Agent 用起来舒服, 人类也不难受.**

两者不冲突时两者都好; 冲突时选 Agent 友好的方案.
`--format json` 永远在, 所有动词支持 Python import,
artifact schema 干净到 Agent 几十行代码能消费完整状态.

### 3 个拒绝

**1. 不做 UI dashboard.**

UI 是好看, 不是「复算」. 看板需要 server / DB / auth,
会破坏"拷走就能验证"的承诺.
要看板, 把 MLflow 嫁接过来 (5 行 adapter).

**2. 不在省空间上动歪脑筋.**

硬链接 / 共享池 / DVC 那种省空间方案, 看起来省了几 MB,
但你把产物拷走 / 打包 / 跨机器复制时, 它们会偷偷失败,
artifact 在边界处就撒了谎.

我们宁可多存几 MB code snapshot, 也不让这种事发生.

**3. 不擅自替你修配置.**

配置错了就硬报错, 列出哪里错 + 已经注册的合法值,
不悄悄给你「智能修复」.

错在 `configs/*.yaml` 注册时, 比错在训练 4 小时后才崩强 10 倍.

### 1 个物量约束

**5 个运行时依赖, 到此为止.**

```
torch / numpy / pandas / pyyaml / safetensors
```

optuna 是 `[extras]` 可选, 不污染核心.
不引 hydra, 不引 pydantic, 不引 click, 不引 tensorboard, 不引 wandb.

---

## rundown 不做什么

(具体边界——「rundown 是什么」的反面, 不是哲学. 用来区分
它和 MLflow / Optuna / PyTorch Lightning 等同类工具的差异.)

- **不是特征工程框架.** 窗口化、列名、掩码、时间特征——你声明, rundown 不知道.
- **不是实验追踪器.** 一次 run 一个目录. 要看板, 把 MLflow 指过去 (5 行 adapter, Phase 2).
- **不是调参工具.** Optuna 是 [extras], 不是核心依赖.
- **不是部署平台.** rundown 的产物是 `.safetensors`, 不是 Docker 镜像.
- **不是深度学习框架.** 没有自动微分魔法, 没有 DSL, 没有构建系统.
  它是三样东西: **注册表协议、训练引擎、产物格式**.

---

## Fact 系统: 保存结论与判断

跑 30 次 sweep 后, 你真正想问的不是「哪个 run 的 best_val_mse 最小」——
而是「**我证过什么 / 推翻了什么 / 哪些结论还有效**」.

`manifest.yaml` 记「这次 run 跑了什么」(客观事实).
Fact 系统记「这次 run **意义**是什么」(主观判断).

### 两层结构

```
artifacts/_facts/
├── main_facts.yaml              # 索引: 所有 fact 的标题 + 状态
├── _main_facts_template.yaml    # 主 fact YAML 模板
├── _sub_fact_template.md        # 子 fact Markdown 模板
├── fact_001.md                  # 一条结论: "method_v2 比 baseline 下降 12% MSE"
├── fact_002.md                  # 另一条结论: "ablation no_attention 失败"
└── ...
```

### Fact 状态字段 (生命周期)

| 状态 | 含义 |
|------|------|
| `pending` | 待验证 (单次实验, 还没复现) |
| `confirmed` | 证实 (多次复现 / 多证据支持) |
| `refuted` | 证伪 (新 run 推翻了先前结论) |
| `superseded` | 被超越 (结论没错, 被更好方法盖过) |

### 使用流程

```bash
# 1. 跑完 sweep, agent/人读 main_facts.yaml 看现状
$ cat artifacts/_facts/main_facts.yaml

# 2. 决定写一条新 fact (或更新老的)
$ cp artifacts/_facts/_sub_fact_template.md artifacts/_facts/fact_xxx.md
$ $EDITOR artifacts/_facts/fact_xxx.md
  # 填: 实验时间 / 设置 / 相关 run 链接 / 结论 / 证据 artifacts

# 3. 把 entry 加进 main_facts.yaml (id / title / status / path)
```

### 哲学边界

- **手动保存** — framework 不自动捕获「这次实验意义」, 那是 user/agent 自己决定的
- **无 CLI** — fact 是文本文件, agent/人用 `read_file` / `write_file` 直接读写
- **无 schema 校验** — 内容由 user/agent 自我负责 (跟 manifest 冻结契约对比: manifest 是事实, fact 是判断, 判断需要弹性)

### 为什么跟 rundown 哲学一致

| 哲学 | fact 系统 |
|------|----------|
| 承诺 1: 一年后还能复算 | fact 是「结论」的产物, 跟 predictions 同级 |
| 承诺 2: manifest 是事实本身 | fact 文件写在 out_root/_facts/, 跟 manifest 同地位 |
| 承诺 3: 让 Agent 用起来舒服 | Agent 能直接 read_file / write_file, 无 CLI 中介 |
| 拒绝 3: 不擅自替你修 | 不自动捕获「这次实验意义」——那是 user/agent 决策 |

---

## 安装 (uv)

```bash
uv sync                                          # 建 .venv + 装依赖 (生成 uv.lock)
uv run rundown train --dataset etth1 --model lstm  # 用 .venv 跑命令
uv run pytest                                    # 跑测试
```

可选: `uv sync --extra optuna` 装 optuna 启用 experiment 的 optuna 模式.

不用 uv 也行:

```bash
pip install -e .[dev,optuna]
python -m rundown.cli train --dataset etth1 --model lstm
```

---

## 第一个 run (3 行命令)

rundown 已经 bundled 一份完整 ETTh1 (17k 行小时级 benchmark) + 2 层 LSTM 模型,
直接跑就有结果:

```bash
uv run rundown train --dataset etth1 --model lstm          # base LSTM
uv run rundown train --dataset etth1 --model lstm --variant large  # hidden=128
uv run rundown train --dataset etth1 --model lstm --epochs-max 5    # 覆盖默认 15
```

输出:

```
[run] lstm__etth1__seed0
  dataset : etth1  (source=csv, path=examples/baseline/data/ETTh1.csv)
  target  : OT
  model   : lstm  (51,777 params, {'in_dim': 6, 'hidden': 64, 'num_layers': 2})
  config  : epochs=15 steps_per_epoch=50 batch_size=32 lr=0.001
  run dir : artifacts\lstm__etth1__seed0
[done] lstm__etth1__seed0  epochs=15  best_val_mse=2.27  test_mse(test_best)=1.20  duration=4.0s
```

跑完去 `artifacts/lstm__etth1__seed0/` 看产物——7 个文件 + 1 个 code 快照,
manifest 一眼看穿所有配置.

---

## 三件套: dataset / model / experiment

rundown 把这三件事做成「在 YAML 里声明 → CLI 一键跑」:

### 1. Dataset (configs/datasets.yaml)

```yaml
datasets:
  etth1:
    source: {kind: csv, path: examples/baseline/data/ETTh1.csv}
    target_col: OT
    feature_cols: [HUFL, HULL, MUFL, MULL, LUFL, LULL]
    provider_factory: examples.baseline.data.build_etth1_providers
    splits: {train: 0.7, val: 0.2, test: 0.1}    # 时间顺序切分
```

- `source` 数据来源 (`csv` / `parquet` / 用户自定义)
- `target_col` 模型预测的目标列
- `provider_factory` 反射调用的 builder 函数 (签名约定: `builder(source_path, splits, target_col, **overrides) -> {train, val, test}`)
- `splits` 时间切分比例 (sum=1)

### 2. Model (configs/models.yaml)

```yaml
models:
  lstm:
    factory:
      target: examples.baseline.model.SequenceLSTM
      kwargs: {in_dim: 6, hidden: 64, num_layers: 2}
    optimizer: {kind: adam, lr: 0.001}
    epochs_max: 15
    variants:                    # 同一 alias 不同超参组合
      small: {hidden: 32, num_layers: 1}
      large: {hidden: 128, num_layers: 2}
      deep:  {hidden: 64, num_layers: 3}
```

- `factory.target` 反射绑定, 注册时立即校验
- `variants` 列模型变体, CLI `--variant large` 切换

### 3. Experiment (configs/experiments.yaml) — sweep

```yaml
experiments:
  # 笛卡尔积: 枚举 variants (顺序跑)
  etth1_lstm_size_sweep:
    dataset: etth1
    model: lstm
    epochs_max: 5
    variants: [small, large, deep]
```

跑 sweep:

```bash
uv run rundown experiment run etth1_lstm_size_sweep
```

输出:

```
[experiment] etth1_lstm_size_sweep  trials=3
  dataset=etth1  model=lstm  epochs_max=5

  snapshot shared: artifacts\_shared\code_08070185.zip  sha8=08070185  size=597168

  [  1/3] variant=small     best_val_mse=2.79  duration=3.3s
  [  2/3] variant=large     best_val_mse=2.08  duration=4.3s
  [  3/3] variant=deep      best_val_mse=2.47  duration=4.2s

[summary] best: variant=large  seed=1  best_val_mse=2.08  run_id=lstm__varlarge__etth1__seed1
          reproduce: rundown train --dataset etth1 --model lstm --variant large --seed 1 --epochs-max 5 --out-dir <same out_root>
```

### 复现 best trial

`[summary]` 输出包含**可直接复制的 reproduce 命令**——同 dataset / model / variant / seed / epochs-max / out-dir 跑会落到**同一个 run_id 目录**（CLI 默认 run_id 规则：`{model}[__var{variant}]__{dataset}__seed{N}`）。

Sweep 自己不再预拼 run_id prefix；trial run_id 与单次 `rundown train` 完全一致，方便复现和 read-run。

### Reproducibility 注意

`engine._set_seed(seed)` 已设 `torch.manual_seed` + `np.random.seed` + `set_num_threads(1)` + `cudnn.deterministic=True` + `use_deterministic_algorithms`。

但 **PyTorch CPU LSTM 仍可能有 ±5% val_mse 漂移**——CPU 上 cuDNN 不工作，LSTM 内部浮点累加顺序受线程数影响，单线程只能减小漂移不能消除。要严格 reproducibility 需 GPU + cuDNN deterministic mode（用户自己开） + commit 代码 + 锁依赖版本（`uv.lock` 已锁）。

框架能保证的是「**同 seed 下 manifest 字段、events.jsonl 序列、weights shape 完全一致**——预测 / 指标可重算、可比较」——具体数值精度依赖 PyTorch 行为。

每个 trial 是一个独立 run, 落自己的 artifacts 目录, **互不影响**. sweep 失败/重启/换 seed 都能局部重跑.

---

## 七个动词

```bash
rundown prepare --dataset <alias>              # 加载数据集, 算 content hash, 注册
rundown train --dataset <ds> --model <m> [...] # 跑一次训练
rundown experiment run <name>                  # 跑注册过的 sweep (笛卡尔积)
rundown read-run <run_id> [--metric X]         # 读已有 run, 算新指标
rundown list-runs [filters]                    # 列出 artifacts/*/manifest.yaml
rundown compare <run_a> <run_b>                # 两个 run 的 manifest diff
rundown analyze --analyzer X <run_id>          # Phase 2 stub
```

每个动词同时支持 CLI / Python (`from rundown.cli import main`) / Agent 调用.

### 全局 `--quiet` / `-q`

压住 stderr 上的 info 行 (events.jsonl 仍写, 产物完整). 适合 CI / agent 自动化调用.

```bash
# 任何位置都生效 (argparse 限制绕过, main() 手动从 argv 抓)
rundown train ... --quiet
rundown --quiet experiment run exp1
rundown -q list-runs
```

stderr 行格式 (新, agent 友好):

```
2026-10-16T21:23:35  step_completed  {"epoch": 2, "step": 10, "train_loss": 0.42, "grad_norm": 1.5}
```

- `{t_iso}` ISO 时间戳
- `{kind}` event 名 (便于 grep)
- JSON 字段值 (复杂 value 安全: 带空格字符串 / list / 嵌套 dict 不会破坏 split)

Agent 解析路径: `line.split("  ", 2)` → `(t_iso, kind, fields_json)` → `json.loads(fields_json)`.

### 查询 run: list-runs / compare

sweep 跑完后, 你想在 `artifacts/` 里找 best run 或者对比两次跑的差异, 不需要写脚本:

```bash
# 列出所有 run (table 格式, 字段: run_id / dataset / model / variant / best_val_mse)
rundown list-runs

# 过滤: dataset=etth1 + model=lstm + best_val_mse < 2.0
rundown list-runs --dataset etth1 --model lstm --below-metric best_val_mse 2.0

# 给 agent 用的 JSON
rundown list-runs --format json
```

```bash
# 两个 run 的 manifest diff — 只输出有差异的字段, 数值显示 Δ
rundown compare lstm__etth1__seed0 cnn__etth1__seed0
```

list-runs 走文件系统 (`glob artifacts/*/manifest.yaml`), 不建 DB——manifest 自己就是索引.

---

## run 跑完后的目录

```
artifacts/
├── _shared/                              # sweep / 多次 train 同代码共享的 snapshot 池
│   ├── code_<sha8>.zip                   # 物理 1 份, 597KB 量级
│   └── code_<sha8>.zip.meta.json         # sweep 模式 subprocess 读这个拿完整 info
└── <run_id>/                             # 每次 run 一个目录
    ├── manifest.yaml
    ├── weights.safetensors
    ├── weights_best.safetensors
    ├── curves.safetensors
    ├── events.jsonl
    ├── predictions/
    │   ├── test.safetensors
    │   └── test_best.safetensors
    └── (无 code_*.zip — 引用 ../_shared/code_<sha8>.zip)
```

**默认行为**: `rundown train --out-dir artifacts` 跑完一次后, snapshot 落在 `artifacts/_shared/code_xxx.zip`, run_dir 内**没有**自己的 code_*.zip. 多次 train 同代码 → 自动跳过写, 物理 1 份.

**`--run-dir` 模式**: snapshot 直接落 run_dir 内 (单 run 自给自足, 不参与 `_shared/` 池).

**git repo 项目**: snapshot 走 `git archive` (tracked files only, 自动 `.gitignore`), 落 `code_<sha8>/` 目录. **非 git repo**: zip 全目录 (排除 `artifacts/` `.venv/` `__pycache__/`), 落 `code_<sha8>.zip`.

**`--no-snapshot`**: 关闭 code snapshot. 不推荐——一年后找不到当时的代码.

跑完一次训练, **整个 out_root 自描述**: 读 `manifest.yaml` 就知道当时跑了什么; 读 `predictions/*.safetensors` 就能算出当时没问过的指标, **不需要重训**; 读 `_shared/code_*.zip` 就能拿到当时的源码.

---

## manifest.yaml 长这样

```yaml
run_id: lstm__etth1__seed0
created_at: 2026-10-16T20:18:00
seed: 0

dataset:
  alias: etth1
  source: {kind: csv, path: examples/baseline/data/ETTh1.csv}
  target_col: OT
  splits: {train: 0.7, val: 0.2, test: 0.1}
  provider_factory: examples.baseline.data.build_etth1_providers

model:
  alias: lstm
  variant: null
  factory: {target: examples.baseline.model.SequenceLSTM, kwargs: {...}}

config:
  epochs_max: 15
  steps_per_epoch: 50
  optimizer: {kind: adam, lr: 0.001}
  predictions_primary: test_best     # reader 读哪个 predictions
  step_event_every_n: 10
  grad_norm_threshold: 100.0

metrics: {best_val_mse: 2.27, epochs_to_best: 11, epochs_actual: 15}

# 自动记录: 跑这次时 host 环境是什么
env:
  rundown_version: 0.1.0
  python: 3.10.21
  platform: Windows-10-10.0.26300-SP0
  torch: 2.14.1+cpu
  numpy: 1.26.4
  argv: [rundown, train, --dataset, etth1, --model, lstm]
  cwd: E:\rundown
  git_commit: a1b2c3d     # 缺省时省略 (not a git repo / git 未装)
  # code snapshot — 默认开; git repo 走 git archive (tracked only), 否则 zip 全目录
  code_snapshot:
    format: tar            # 或 "zip" (非 git repo)
    tracked_only: true     # git archive 路径为 true, zip 路径为 false
    sha256: 421778...      # 包内容 hash, 跟 git_commit 互为校验
    sha256_short: 421778bc # sha256[:8], 用作 archive_path 文件名
    size_bytes: 593071
    num_files: 36
    archive_path: ../_shared/code_421778bc.zip  # 相对 run_dir, 读者直接 cat 即可
```

读者路径: `cd artifacts/<run_id> && cat ../_shared/code_xxx.zip | less` —— manifest 直接告诉代码在哪.

---

## 目录结构

```
rundown/
├── __init__.py
├── cli.py              # 7 个动词: prepare / train / experiment / read-run / list-runs / compare / analyze
├── registry.py         # YAML 加载 + 反射绑定 + alias 校验 (datasets / models / experiments)
├── engine.py           # GradientStrategy.fit() + per-step events + 自动记录 env
├── artifact.py         # 写/读 manifest + weights + curves + predictions
├── _env.py             # capture_env() + snapshot_code() (自动记录 host env + code)
├── events.py           # emit() 全局事件发射器
├── ids.py              # alias / run_id 正则校验
├── cli_query.py        # list_runs / compare_runs (glob manifest.yaml 直接读)
├── experiment.py       # sweep: 笛卡尔积 (subprocess 调 train)
├── defaults.py         # DEFAULT_BATCH_SIZE / DEFAULT_LR 等单点真值
├── facts/              # fact 系统: main_facts.yaml 索引 + 子 fact .md
└── (11 个 package 文件 / 子包, ~1900 行)

configs/
├── datasets.yaml
├── models.yaml
└── experiments.yaml

artifacts/              # run 目录, 一个 run 一个文件夹
├── _shared/            # 多次 train / sweep 共享的 code snapshot 池
└── <run_id>/
examples/
└── baseline/           # ETTh1 bundled + LSTM + SlidingWindowProvider
    ├── data/ETTh1.csv
    ├── data.py
    ├── model.py
    ├── train.py        # thin wrapper: subprocess 调 rundown.cli train
    └── test_baseline.py
tests/                  # pytest, 65 个测试
├── fixtures.py         # ToyNet + RandomProvider
├── test_smoke.py       # 47 测试: 核心流程 / cli / registry / artifact / snapshot
└── test_facts.py       # 15 测试: facts 系统路径 / 模板 / 哲学边界
                       # (+ 3 测试: events stderr JSON 格式 + --quiet)
AGENTS.md               # 给 AI Agent 看的约束文档
pyproject.toml          # PEP 621 metadata + [optuna] extras
uv.lock                 # 锁定的依赖版本 (commit 进 git)
```

11 个 package 文件 / 子包, ~1900 行. 完整跑通 ETTh1 (17k 行真实数据) + 2 层 LSTM,
test MSE ≈ 1.10 (标准化空间), **对齐 ETT 论文 baseline**.

---

## 什么时候用 rundown

- 你跑了 ≥ 3 次实验, 开始记不住哪次用了什么配置 → 用 rundown
- 你的论文要发出去, 你担心 18 个月后复现不出来 → 用 rundown
- 你的实验代码会被同事 / 审稿人 / Agent 接手 → 用 rundown
- 你想批量 sweep (笛卡尔积或 optuna) 而不想手写循环 → 用 rundown
- 你只是想跑 1 次 demo, 跑完就丢 → **别用, 直接用 PyTorch**

## 什么时候**不要**用 rundown

- 你是 PyTorch 初学者, 在做"如何写第一个训练循环"的练习 → 直接写
- 你的项目有强 MLOps 需求 (A/B 部署、流量切分、监控告警) → 用 MLflow / Ray
- 你的数据是图片/文本, 没有时序结构 → rundown 的窗口化对你没有意义
- 你的项目已经定型, 跑 5 年不会改 → 别折腾, 现状能跑就行

---

## License

MIT.
