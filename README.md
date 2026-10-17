# rundown

**面向个人算法开发者 (无论是否使用 AI 辅助) · 你管 run, rundown 管 down**

rundown 把写在 YAML 里的「数据集 + 模型 + 实验」变成一次训练 run. 跑完时 manifest / weights /
curves / predictions 自动落到磁盘, 用 manifest 当索引 — 6 个月后直接 `cat manifest.yaml`
找回, 不依赖 git log, 不依赖未来自己的 Onenote.

你跟 rundown 是 **1:1**: 不需要 server, 不需要 team, 不需要共享 dashboard. 框架作为一
个人都用的 lightweight 工具, 给个人算法开发者用. Agent 集成是顺手兼容, 不是主干.

---

## 为什么

这 5 个痛点每个都对应一个 rundown 子系统, 是每个子系统要解决的反向定义:

1. **Pain A — 想换 dataset/model 版本时, 训练脚本里写满 raw import, grep 全项目改**
   → 回应: **Registry** 用 alias 把 yaml 与 import 解耦, 改 yaml 即可换 dataset/model.

2. **Pain B — 跑完忘了这次 lr / hidden / num_layers 是多少、对应哪个 git commit**
   → 回应: **Engine** 每次 run 强制 capture `config` + `env.git_commit`, 跑完即落盘.

3. **Pain C — 跑 50 次 sweep, 哪个 run 对应 best_val_mse, 不靠 Onenote 找不到**
   → 回应: **Artifact** 用 manifest.yaml 做索引 (run 同时是数据 + 索引), `list-runs` 一行 SQL 都不要.

4. **Pain D — 6 个月后 cat 自己的 run 目录也读不懂 — manifest 是机器格式, 人需要 projection 才能 grasp**
   → 回应: **Inspect** 渲出 README-style markdown (purpose / setup / environment / results / how-to-evaluate),
   你 / reviewer / Agent 拿到 consolidated view 不再拼.

5. **Pain E — 跑完 30 次, 「method_v2 比 baseline 强 12%」这种主观结论自己记得, reviewer 没法 trace**
   → 回应: **Fact** 把 manifest 自动写事实, 把主观判断写到 `artifacts/_facts/*.md` 标状态, 走 lifecycle.

每个子系统的具体机制看下面「rundown 的子系统」段 — 五对五, 一一对应.

下面告诉你 rundown 的设计哲学 (做什么 / 不做什么) + 系统组成 + 怎么上手.

---

## 设计哲学

`rundown` 的命名已经暗示 framework 的核心 split: **run** 是你做的唯一动作, **down**
是 framework 自动做的——把这次 run 涉及的产物 (算法 / 实验 / 结果 / 结论) 自动落盘、
自动建索引、自动跨时间查询. 这一节是这 split 落到 manifesto 上的两层表达: 我们做
什么, 与我们不做什么.

### 我们做什么

**1 个物量约束 — rundown 是一个轻量化框架.**

```
torch / numpy / pandas / pyyaml / safetensors    # 5 个运行时依赖, 到此为止
```

optuna 是 `[extras]` 可选, 不污染核心. **不引** hydra, 不引 pydantic, 不引 click,
不引 tensorboard, 不引 wandb. 选择理由: 每个被拒的包都尝试替 user 做「更聪明的事」,
而 smart-default 的代价是把你的工作流吞进去. 拒绝这个诱惑, framework 才能保持 *5 个
依赖* 级别的轻量.

**3 个承诺 (跟 ## 为什么 段的 5 个 Pain 一一对应)。**

1. **一年后还能复算** (Pain A, B 的工程保障)
   跑完的瞬间, 复算这件事的所有信息都在 run 目录里. 不依赖 git log, 不依赖外部 DB,
   不依赖人记. 「表 4 第二行那个数字是哪个 split / 哪个 seed / 哪份代码 / 哪台机器」——
   拷走目录就能答完.

2. **manifest 是事实本身** (Pain C 的索引机制)
   `manifest.yaml` 写盘即合同. 跨 run 查询靠 glob manifest 自己, 不建 DB. manifest
   既是数据, 也是索引. 「拷走即 verify」——这是承诺的物理形态, 不是营销词.

3. **Agent 友好, 人类也好用**
   两者不冲突时两者都好; 冲突时选 Agent 友好的方案. `--format json` 永远在, 所有动词
   支持 Python import (`from rundown.cli import main` / `from rundown.inspect_run
   import load_run`), artifact schema 干净到 Agent 几十行代码能消费完整状态.
   Agent 集成是兼容层, **不是** 主干 — 主干仍是「个人开发者 + 轻量化框架」.

### 我们不做什么

**2 个拒绝 (framework 的设计选择, 可证).**

1. **不做 UI dashboard**
   UI 是好看, 不是「复算」. 看板要 server / DB / auth, 必然引入运维依赖, 会破坏
   "拷走即 verify" 的承诺. 要看板, 把 MLflow 嫁接过来 (5 行 adapter, 在 Phase 2).

2. **不在省空间上动歪脑筋**
   硬链接 / 共享池 / DVC 那种省空间方案, 看起来省几 MB, 但你拷走产物 / 跨机器复制时
   它们会偷偷失败. artifact 在边界处就撒了谎. rundown 宁可多存几 MB code snapshot,
   让边界处永远诚实地自描述.

**anti-scope list (rundown 不是这些)**

> 这个清单跟「2 个拒绝」区别: 拒绝是 *设计选择* (原则上可改), anti-scope 是 *领域边界*
> (原则上不该做). 你想加这些功能, 请用 MLflow / Optuna / Hydra 等专门工具.

- **不是特征工程框架.** 窗口化、列名、掩码、时间特征——你声明, rundown 不知道.
- **不是实验追踪器.** 一次 run 一个目录. 要看板, 用 MLflow adapter.
- **不是调参工具.** Optuna 是 [extras], sweep 模式下走 rundownexperiment run.
- **不是部署平台.** rundown 产物是 `.safetensors`, 不是 Docker 镜像.
- **不是深度学习框架.** 没有自动微分魔法, 没有 DSL, 没有构建系统.

它是三样东西: **注册表协议 (Registry) / 训练引擎 (Engine) / 产物格式 (Artifact)**.
下面那节展开.

---

## rundown 的子系统

5 个 subsystem, 跟 ## 为什么 段 5 个 Pain 1:1 对应. 每个子系统给具体的入口、机制、一句话
特征.

### Registry — 回应 Pain A (训练脚本 raw import 难解耦)

**痛**: 想换 dataset / model / loss / optimizer 版本时, 训练脚本里写满 raw import
(`from mymodels.transformer import MyTransformer` 等), grep 全项目改, 容易漏.

**机制**: `configs/{datasets,models,experiments}.yaml` 用 alias 命名 (`dataset: etth1`,
`model: lstm`), rundown 的 `Registry.load_*()` 用 dotted-path 反射 (`factory.target:
examples.baseline.model.SequenceLSTM`) 把 alias → 实际类. 改 yaml 就换 dataset/model,
不动训练脚本.

**入口**: 写一个 `configs/models.yaml` 里加 entry, 改 `factory.target` 指向你的 PyTorch 类.

### Engine — 回应 Pain B (跑完忘了啥超参/哪个 commit)

**痛**: 跑完忘了这次 lr / hidden / num_layers 是多少, 对应哪个 git commit. git log
不直观 (commit message ≠ run config).

**机制**: `Engine.fit(provider, model, run_dir)` 入口, 训练循环里强制 capture
`config` (你传的 optimizer / epochs) + `env.git_commit` + `env.git_dirty`, 落盘到
`run_dir/manifest.yaml`. 配 BPTT / early stop / grad clipping 等训练策略.

**入口**: 调 `rundown.train` 这条 CLI 即可. 训练逻辑封装在 `rundown/engine.py`,
不替 user 写模型.

### Artifact — 回应 Pain C (sweep 完 best 找不到)

**痛**: 跑 50 次 sweep, 想问「这次 best_val_mse 来自哪个 run」要 `grep` manifest
+ 屏幕截图 + Onenote.

**机制**: 每次 run 落 `manifest.yaml` + `weights.safetensors` + `curves.safetensors`
+ `predictions/test_best.safetensors` + `events.jsonl`. `manifest` 既是数据也是索引,
跨 run 查询靠 `rundown list-runs --above-metric best_val_mse 2.0` 走文件系统 glob,
不建 DB.

**入口**: 跑完一次训练产物自动落到 `artifacts/<run_id>/`, `rundown list-runs` / `compare`
/ `inspect` 都是 read-only.

### Inspect — 回应 Pain D (run 目录人读不懂)

**痛**: 自己 6 个月后 cat 自己的 run 目录 (manifest + safetensors) 也读不懂 —
manifest 是机器格式, 人脑需要 *projection* 才能 grasp.

**机制**: `rundown inspect <run_id>` 渲出 README-style markdown, 把 manifest +
curves + predictions + code snapshot 元信息 投影成人脑可读的 7 段: purpose / setup
/ environment / results / how-to-evaluate / artifacts on disk. 这是 artifact ↔
人类阅读的 canonical translator, 也是 Agent 第一次接陌生 run 的 entry point.

**入口**: `rundown inspect <run_id>` 输出到 stdout (可 pipe `.md`); 缺 pyyaml /
safetensors 时降级.

### Fact — 回应 Pain E (主观结论难 trace)

**痛**: 跑完 30 次, 「method_v2 比 baseline 强 12%」这种主观结论自己记得, 写论文 /
给 reviewer / 半年后的自己 trace 时没法验证.

**机制**: 跟前面 4 个 subsystem 不同——这是 **手动 escape hatch** (framework 不写
事实). `artifacts/_facts/main_facts.yaml` 索引每条结论, `fact_<id>.md` 一条结论.
manifest 自动记事实, fact 文件写判断, 4 状态 (`pending / confirmed / refuted /
superseded`) lifecycle.

**入口**: 直接 `cat artifacts/_facts/main_facts.yaml` 读, `write_file` 写事实卡.
有 CLI 的化反而违反原则——事实/判断的边界要保持人手覆盖.

---

## 快速入手

不管你是 bash / Python / Agent 用户, 都是这 3 步.

**bash** — 跑一次训练并拿到产物:

```bash
uv sync                                                       # 建 venv + 装 5 个运行时依赖
uv run rundown train --dataset etth1 --model lstm --variant small --epochs-max 5
ls artifacts/lstm__varsmall__etth1__seed0/                    # manifest + weights + curves + predictions
```

**Python** — 读已存 run, 渲成 README-style markdown:

```python
from rundown.inspect_run import load_run, render_run_markdown
print(render_run_markdown(load_run("artifacts/lstm__varsmall__etth1__seed0")))
```

**Agent** — 跨 run 抓指标, 不要写 Python:

```bash
uv run rundown list-runs --format json | jq '[.[] | {id: .run_id, mse: .metrics.best_val_mse}]'
```

下面章节是 modules 具体怎么用 (三件套配置 + 8 个动词 + 目录布局 + artifact 例子).

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
[run] lstm__varsmall__etth1__seed0
  dataset : etth1  (source=csv, path=examples/baseline/data/ETTh1.csv)
  target  : OT
  model   : lstm  variant=small  (5,153 params, {'in_dim': 6, 'hidden': 32, 'num_layers': 1})
  config  : epochs=5 steps_per_epoch=50 batch_size=32 lr=0.001
  run dir : artifacts\lstm__varsmall__etth1__seed0
[done] lstm__varsmall__etth1__seed0  epochs=5  best_val_mse=2.7334  test_mse(test_best)=1.16  duration=2.6s
```

跑完去 `artifacts/lstm__varsmall__etth1__seed0/` 看产物——7 个文件 + 共享的 `_shared/` code 快照,
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
    variants:                    # 便捷预设 — 同一模型不同 size 组合
      small: {hidden: 32, num_layers: 1}
      large: {hidden: 128, num_layers: 2}
      deep:  {hidden: 64, num_layers: 3}
```

- `factory.target` 反射绑定, 注册时立即校验
- `factory.kwargs` 固定结构 (数据集决定的, 如 `in_dim`)
- `variants` 便捷预设 — CLI `--variant large` 切换
- **不在 models.yaml**: `optimizer` / `epochs_max` / `dataset` / 训练超参 — 见下方边界表

### 3. Experiment (configs/experiments.yaml) — sweep

```yaml
experiments:
  # 笛卡尔积: 枚举 variants (顺序跑)
  etth1_lstm_size_sweep:
    dataset: etth1
    model: lstm
    epochs_max: 5
    optimizer: {kind: adam, lr: 0.001}    # 可选, 不写走 defaults.DEFAULT_OPTIMIZER
    variants: [small, large, deep]        # 引用 models.lstm.variants 的 key
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

### models vs experiments 边界

| 字段 | 归属 | 谁消费 | 例 |
|------|------|--------|-----|
| `factory.target` | **models** | registry 反射 | `examples.baseline.model.SequenceLSTM` |
| `factory.kwargs` | **models** | `cls(**kwargs)` | `{in_dim: 6}` (固定结构) |
| `variants` | **models** | `--variant large` | `{small: {hidden: 32}, large: {hidden: 128}}` (便捷预设) |
| `optimizer` (默认) | defaults.py | 单次 train fallback | `defaults.DEFAULT_OPTIMIZER = {kind: adam, lr: 1e-3}` |
| `optimizer` (覆盖) | **experiments** | sweep 子进程 | `optimizer: {kind: adam, lr: 0.0005}` |
| `dataset` | **experiments** | provider 反射 | `etth1` |
| `model` | **experiments** | registry 查 | `lstm` |
| `epochs_max` | **experiments** | engine | `5` (sweep 跑短点) |
| `batch_size` / `steps_per_epoch` | **experiments** | engine / provider | 覆盖默认 |
| `variants` (列表) | **experiments** | sweep 笛卡尔积 | `[small, large, deep]` (引用 models 的 key) |
| `--kwargs key=value` | CLI 透传 | engine.fit | optuna sweep 自由传 |

**核心原则**:

- **models** = 「一个 PyTorch 类 + 默认 kwargs + 便捷预设」。它不知道训练时长、不知道用哪个 optimizer、不知道跑什么数据集。
- **experiments** = 「一次 sweep 的设置」。它决定用哪个 model/dataset/optimizer，训练多久，sweep 哪些 variant。
- **defaults.py** = 「单次 `rundown train`（无 experiment）的单点真值」。sweep 时由 experiment 覆盖。
- **`--kwargs`** = 「runtime 透传，跟 YAML 正交」。optuna / agent 用，绕过 variants 限制。

**老 config 向后兼容**: models.yaml 里的 `optimizer:` / `epochs_max:` 字段会被静默忽略（不再读）。新 config 应该删掉这些字段——见 `test_load_models_ignores_optimizer_field`。

### 复现 best trial

`[summary]` 输出包含**可直接复制的 reproduce 命令**——同 dataset / model / variant / seed / epochs-max / out-dir 跑会落到**同一个 run_id 目录**（CLI 默认 run_id 规则：`{model}[__var{variant}]__{dataset}__seed{N}`）。

Sweep 自己不再预拼 run_id prefix；trial run_id 与单次 `rundown train` 完全一致，方便复现和 read-run。

### Reproducibility 注意

`engine._set_seed(seed)` 已设 `torch.manual_seed` + `np.random.seed` + `set_num_threads(1)` + `cudnn.deterministic=True` + `use_deterministic_algorithms`。

但 **PyTorch CPU LSTM 仍可能有 ±5% val_mse 漂移**——CPU 上 cuDNN 不工作，LSTM 内部浮点累加顺序受线程数影响，单线程只能减小漂移不能消除。要严格 reproducibility 需 GPU + cuDNN deterministic mode（用户自己开） + commit 代码 + 锁依赖版本（`uv.lock` 已锁）。

框架能保证的是「**同 seed 下 manifest 字段、events.jsonl 序列、weights shape 完全一致**——预测 / 指标可重算、可比较」——具体数值精度依赖 PyTorch 行为。

每个 trial 是一个独立 run, 落自己的 artifacts 目录, **互不影响**. sweep 失败/重启/换 seed 都能局部重跑.

---

## 八个动词

```bash
rundown prepare --dataset <alias>              # 加载数据集, 算 content hash, 注册
rundown train --dataset <ds> --model <m> [...] # 跑一次训练
rundown experiment run <name>                  # 跑注册过的 sweep (笛卡尔积)
rundown read-run <run_id> [--metric X]         # 读已有 run, 算新指标
rundown list-runs [filters]                    # 列出 artifacts/*/manifest.yaml
rundown compare <run_a> <run_b>                # 两个 run 的 manifest diff
rundown inspect <run_id|--all [--since-days N]] # 读 run, 输出 README-style markdown
rundown analyze --analyzer X <run_id>          # Phase 2 stub
```

每个动词同时支持 CLI / Python (`from rundown.cli import main`) / Agent 调用.

### 给 Agent 消费者 (consume contract)

**Do**:

- `--format json` 永远在 (list-runs 是最先实现的, 后续 verb 会陆续加); 不解析 stderr 进度行.
- `artifacts/<run_id>/manifest.yaml` 是单点真值 — 不要尝试从 events.jsonl 推断 run 状态 (events 是事件流, 不是 schema).
- `rundown inspect <run_id> -q` 拉一份 README-style markdown — 含 purpose/setup/environment/results/curves/predictions/events/how-to-evaluate/artifacts 8 段, 可喂给 LLM.
- Python 入口: `from rundown.cli import main(argv)` / `from rundown.inspect_run import load_run, render_run_markdown`. standalone, 不需要 train / engine.
- Run 排序用 manifest.created_at, 用 `--since-days N` 切时间窗 (UTC).

**Don't**:

- 不要 hardcode `run_id` 格式去解析 (例如 `lstm__varsmall__etth1__seed0`); 从 manifest.run_id 字段读.
- 不要假定产出物路径不变; 唯一可靠入口是 `manifest.config` 里 `out_dir` / `_shared/code_<sha>/` 引用.
- 不要解析 stderr 上的进度行 — 那些 stderr 是给人看的, 会被 `--quiet` 压掉.
- 不要 split `events.jsonl` 行用空格 — 用 `line.split("  ", 2)` (agent 友好格式里间是双空格).

**When CLI is unavailable** (例如 submodule 调用, 无 system python): Python 入口仍然工作; `from rundown.inspect_run import load_run` 不依赖 cli 解析.

### 全局 `--quiet` / `-q`

压住 stderr 上的 info 行 (events.jsonl 仍写, 产物完整). 适合 CI / agent 自动化调用.

### `--kwargs KEY=VALUE` — 透传任意超参给 model factory

`models.variants` 是**便捷预设**（`small` / `large` / `deep` 是常见 size 组合）。
**optuna sweep 或临时超参**用 `--kwargs` 自由传, 绕过 variants 限制:

```bash
# 透传多个 — 合并顺序: factory_kwargs < variants < --kwargs (后者覆盖)
uv run rundown train --model lstm --variant small \
  --kwargs hidden=128 --kwargs num_layers=2

# optuna 调 rundown 时 (任意超参, 不用预注册 variant)
uv run rundown train --model lstm \
  --kwargs hidden=$HIDDEN --kwargs lr=$LR
```

值类型自动判断: `int` / `float` / `bool` / `str`.

不存在的 kwargs key → factory 自己抛 `TypeError` (不静默吃掉 — 跟「拒绝 3: 不擅自替你修」一致).

`--kwargs` 不影响 code snapshot hash — 同代码不同 kwargs → 同 sha → 自动 dedupe.

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

### 读一个 run: `inspect` — README-style markdown

`list-runs` 给一行摘要; `inspect` 给一份完整 README: 跑这次的目的 / 设置 / 环境 / 结果 / 怎么复算 / 产物清单. stdout 是 markdown, 直接重定向到 `.md` 文件.

```bash
# 单个 run
rundown inspect lstm__varsmall__etth1__seed0 > lstm__varsmall__etth1__seed0.md

# 批量 (按创建时间过滤最近 7 天)
rundown inspect --all --since-days 7 > weekly_runs.md

# 批量 + 按 dataset / model / variant 过滤
rundown inspect --all --dataset etth1 --model lstm --variant small
```

输出例子 (节选):

````markdown
# Run: `lstm__varsmall__etth1__seed0`
_2026-10-17T00:38:50_

## Purpose
Predict `OT` 25 steps ahead on `etth1` using `lstm/small`.

## Setup
- **Dataset**: `etth1` (csv, target `OT`)
  - splits: train=0.7 / val=0.2 / test=0.1
- **Model**: `lstm` / variant=`small`
  - factory: `examples.baseline.model.SequenceLSTM`
  - kwargs: in_dim=6, hidden=32, num_layers=1
- **Config**: epochs_max=5, steps_per_epoch=50, optimizer=adam(lr=0.001)
- **Seed**: 0

## Results
| metric | value |
| --- | --- |
| best_val_mse | 2.7334 |
| epochs_to_best | 4 |
| epochs_actual | 5 |

### Loss curves
| epoch | train_loss | val_mse | best? |
| --- | --- | --- | --- |
| 1 | 0.7528 | 3.2733 |  |
| 2 | 0.5969 | 4.1101 |  |
| 4 | 0.5421 | 2.7334 | BEST |

## How to evaluate
```python
from rundown.inspect_run import load_run, render_run_markdown
run = load_run(r'artifacts\lstm__varsmall__etth1__seed0')
print(render_run_markdown(run))
```

Or re-train with identical code:
```bash
git checkout 1b5e06c
uv run rundown train --dataset etth1 --model lstm --variant small --seed 0 \
    --epochs-max 5 --out-dir artifacts
```

## Artifacts on disk
- `curves.safetensors` (176B)
- `manifest.yaml` (1.5KB)
- `predictions/test_best.safetensors` (6.4KB)
- `weights.safetensors` (20.6KB)
- `weights_best.safetensors` (20.6KB)
````

设计哲学:
- **read-only** — 不动 `artifacts/`, 不写日志, 不创建文件 (除非你 `>` redirect).
- **降级** — 缺 `pyyaml` / `safetensors` 时只读 `manifest.yaml`, 仍然能渲染 Setup / Environment / Results 段.
- **agent 友好** — stdout 是干净 markdown, 可以 `cat` / `grep` / 喂给 LLM.
- **Python 入口** — `from rundown.inspect_run import load_run, render_run_markdown` 直接当 lib 用, 不一定要 CLI.

---

## 目录布局约定

跑 ML 实验的项目根目录有 4 类互不重叠的子目录. 区分维度是 **生命周期** (durable vs ephemeral), 不是 **作者** (agent vs human):

| 区域 | 生命周期 | git | 角色 |
|------|----------|-----|------|
| `src/` `tests/` `configs/` | durable | **tracked** | 项目代码 — 进 commit / 论文 / release |
| `examples/<name>/`      | durable | **tracked** | 参考 showcase — 1 套最小可跑的 model + data + train (`examples/baseline/` 默认存在). 你写的真实代码可不放这里 |
| `artifacts/` | durable (量大) | **ignored** | rundown 跑出来的产物 — manifest + weights + curves + predictions |
| `.scratch/` | ephemeral | **ignored** | 过程工具代码 — agent session 临时产物 / human debug 脚本 / 实验探针 |

> Agent 与 human **都可以** 写 `src/`, 也都可以写 `.scratch/`. 是否进产品是 lifecycle 决定, 不是作者决定.

### `examples/<name>/` 怎么用

它是 **rundown 自带的参考 showcase** — 1 套最小可跑的 model + data + train.
跑 `python -m examples.baseline.train` 就能源码级看完 rundown 的 end-to-end 流程.

你**真实项目**的模型 / 数据 loader 不必放这里. `configs/models.yaml` 里写的是
dotted path (`examples.baseline.model.SequenceLSTM`), 你写 `mymodels.transformer.MyTformer`
也 work. `examples/` 是 **showcase 锚点, 不是 framework 强制的家**.

adapt guide 在 `examples/baseline/README.md` 里:「把 `data.py` / `model.py` 换成你的,
`train.py` 改 RUN_ID / WARM_START 等常量」。



### `.scratch/` 怎么用

子目录名 = agent session id 一对一, 避免多 session 互踩:

```
.scratch/
├── agent_cursor_abc123/    # Cursor session abc123 的临时脚本
├── agent_cline_xyz789/     # Cline session xyz789 的探索代码
├── debug_2024_10_17/       # 人类 debug 一日的临时 queries
└── probe_lr_sweep/         # 实验性 lr sweep 探针, 留不留看心情
```

**enforcement 谁负责**: agent framework 自己. Cursor / Cline / Devin 都有 session work-dir 配置, 把 session 临时文件指向 `.scratch/<session_id>/`. rundown **不读不写** 这个目录, manifest 也不捕获 — 这是 integration contract, 不是 framework invariant.

### `.gitignore` 推荐

```gitignore
# rundown 产物
artifacts/

# 过程代码 (agent 临时 / human debug)
.scratch/

# 通用
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
```

拷走项目代码 (`git clone`) → 默认不会拷 `.scratch/`, 也不会拷 `artifacts/`. 这是「拷走即 verify」的物理保障: durable artifacts 在 commit 里, ephemeral 留本地.

### 跟 rundown 哲学的关系

- **承诺 2 (拷走即 verify)** — `.scratch/` 默认不拷走, durable/ephemeral 自动分离, 不用 framework 强制.
- **拒绝 3 (不擅自替你修)** — agent session id 是 agent framework 自己声明的, 不由 rundown 强加. rundown 不假设 "session id 在 manifest 里", 因为 session boundary 是 agent framework 的语义, rundown 不抢.
- **1 个物量约束** — directory convention 是 zero-engineering 解. 不引新机制, 不加 manifest 字段. 复用现有 `artifacts/` 已有的 gitignore pattern, 自然推广.

---

## run 跑完后的目录

```
artifacts/
├── _shared/                              # sweep / 多次 train 同代码共享的 snapshot 池
│   ├── code_<sha8>/                      # git repo 项目: `git archive` 输出 tar 目录 (preferred)
│   │   └── .meta.json                    # sha256 / size_bytes / num_files
│   ├── code_<sha8>.zip                   # 非 git repo 项目: zip 全目录 (excludes artifacts/.venv/__pycache__)
│   └── code_<sha8>.zip.meta.json         # .zip 模式的 metadata sidecar
└── <run_id>/                             # 每次 run 一个目录
    ├── manifest.yaml
    ├── weights.safetensors
    ├── weights_best.safetensors
    ├── curves.safetensors
    ├── events.jsonl
    ├── predictions/
    │   ├── test.safetensors
    │   └── test_best.safetensors
    └── (run_dir 内无 code_* — 引用 ../_shared/code_<sha8>)
```

**默认行为**: `rundown train --out-dir artifacts` 跑完一次后, snapshot 落在 `artifacts/_shared/code_xxx/` 或 `code_xxx.zip` (取决于是否 git repo), run_dir 内**没有**自己的 code 副本. 多次 train 同代码 → 自动跳过写, 物理 1 份.

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
├── cli.py              # 8 个动词: prepare / train / experiment / read-run / list-runs / compare / inspect / analyze
├── registry.py         # YAML 加载 + 反射绑定 + alias 校验 (datasets / models / experiments)
├── engine.py           # GradientStrategy.fit() + per-step events + 自动记录 env
├── artifact.py         # 写/读 manifest + weights + curves + predictions
├── _env.py             # capture_env() + snapshot_code() (自动记录 host env + code)
├── events.py           # emit() 全局事件发射器
├── ids.py              # alias / run_id 正则校验
├── cli_query.py        # list_runs / compare_runs (glob manifest.yaml 直接读)
├── experiment.py       # sweep: 笛卡尔积 (subprocess 调 train)
├── defaults.py         # DEFAULT_BATCH_SIZE / DEFAULT_LR / DEFAULT_OPTIMIZER 等单点真值
├── inspect_run.py      # rundown inspect: read-only run → README-style markdown
├── facts/              # fact 系统: main_facts.yaml 索引 + 子 fact .md
└── (12 个 package 文件 / 子包, ~2600 行)

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
tests/                  # pytest, 90 个测试
├── fixtures.py         # ToyNet + RandomProvider
├── test_smoke.py       # 60 测试: 核心流程 / cli / registry / artifact / snapshot / events
├── test_facts.py       # 15 测试: facts 系统路径 / 模板 / 哲学边界
└── test_inspect_run.py # 15 测试: load_run / render / since_days / CLI
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

## 附录

这一段是 reference 性质的内容. **约束、规则、模板、命名规范、forbidden imports 都
不在这里重复** — 它们在 `AGENTS.md` (项目给 AI Agent 看的约束文档). 本附录只
放 README 涉及的指向与概览.

### Pointer 列表

| 内容 | 位置 |
|------|------|
| Module 边界 / 谁能 import 什么 | `AGENTS.md` §1 |
| Anti-patterns (不要做的 8 件事) | `AGENTS.md` §2 |
| 命名规范 (dataset_alias / model_id / run_id 正则) | `AGENTS.md` §3 |
| 测试要求 / 跑通门槛 | `AGENTS.md` §4 |
| Forbidden imports (lint rule, pickle / torch.save / matplotlib / seaborn / plotly) | `AGENTS.md` §5 |
| `rundown.ids.assert_valid` (写入前校验) | `rundown/ids.py` |
| Adapter 模式 (MLflow / WandB 等怎么接 rundown) | `examples/baseline/README.md` (「Adapting to your project」段) |
| inspect 输出 schema | `rundown/inspect_run.py` (RunIntel dataclass) |
| Manifest schema | `rundown/artifact.py` + `examples/` 的实际 manifest 输出 |

### 一些「不用进 README」的隐性规则

- **run_id 格式**: `^[a-z][a-z0-9_]+__[a-z][a-z0-9_]+__seed\d+$` — 详见 `AGENTS.md` §3
  及 `rundown.ids.assert_valid`.
- **directory convention** (lifecycle 维度): 详见前面「目录布局约定」段, 本附录不重复.
- **.gitignore 推荐块**: 详见「目录布局约定」段内嵌块.

### 给读者的 meta 建议

README + AGENTS.md 应该一起读, 但顺序很重要:

1. **本文档** — 先 grasp 设计哲学 + 5 个 subsystem + 怎么上手. 给「做什么」/「不做什么」/「机制对应痛点」的 narrative 答案.
2. **`AGENTS.md`** — 当你要 *写* 代码进来 (新 model / 新 dataset / PR), 读这个. 给你 lint/rule/anti-pattern 的硬约束.
3. **`examples/baseline/`** — 当你 *adapt* rundown 到自己项目时, 这是 source-of-truth: 改 `data.py` + `model.py` + `train.py` 三处.

---

## License

MIT License. Copyright © 2026 Dynan Wu & rundown contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
