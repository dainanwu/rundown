# rundown

**面向个人算法开发者（无论是否使用 AI 辅助）· 你负责 run，rundown 负责 down**

rundown 把写在 YAML 里的「数据集 + 模型 + 实验」变成一次训练运行。跑完时，manifest / weights /
curves / predictions 自动落到磁盘，用 manifest 当索引 —— 6 个月后直接 `cat manifest.yaml`
就能找回，不依赖 git log，也不依赖未来自己的笔记。

你和 rundown 是**一对一**的关系：不需要服务器、不需要团队、不需要共享看板。框架定位为一个
人就能用的轻量工具，服务个人算法开发者。Agent 集成只是顺手兼容，不是主干。

---

## 为什么

这 5 个痛点各自对应一个 rundown 子系统，是每个子系统要解决的反向定义：

1. **痛点 A —— 想换数据集 / 模型版本时，训练脚本里写满了硬编码 import，只能全项目搜改**
   → 回应：**Registry（注册表）**用别名（alias）把 YAML 与 import 解耦，改 YAML 即可换数据集 / 模型。

2. **痛点 B —— 跑完忘了这次学习率 / 隐藏层 / 层数是多少、对应哪个 git 提交**
   → 回应：**Engine（引擎）**每次运行强制记录 `config` + `env.git_commit`，跑完即落盘。

3. **痛点 C —— 跑 50 次扫描，哪个运行对应 best_val_mse，不靠笔记找不到**
   → 回应：**Artifact（产物）**用 manifest.yaml 做索引（运行既是数据也是索引），`list-runs` 一行 SQL 都不用写。

4. **痛点 D —— 6 个月后翻自己的运行目录也读不懂 —— manifest 是机器格式，人需要一番转换才能理解**
   → 回应：**Inspect（检视）**渲染出 README 风格的 Markdown（目的 / 设置 / 环境 / 结果 / 如何评估），
   你 / 审稿人 / Agent 拿到汇总视图后不用再拼。

5. **痛点 E —— 跑完 30 次，「method_v2 比 baseline 强 12%」这类主观结论自己记得，审稿人却没法追溯**
   → 回应：**Fact（事实）**是一组**手动写的事实卡**，把主观判断写到 `artifacts/_facts/*.md`
   并标记状态，走完整的生命周期。`manifest.yaml` 仍是自动写的事实记录（机器可读），
   fact 文件写的是人脑下的判断，两层互不替代。

每个子系统的具体机制见下面「rundown 的子系统」一节 —— 五对五，一一对应。

下面介绍 rundown 的设计哲学（做什么 / 不做什么）、系统组成，以及怎么上手。

---

## 设计哲学

`rundown` 这个名字已经暗示了框架的核心分工：**run** 是你做的唯一动作，**down**
是框架自动完成的部分——把这次运行涉及的产物（算法 / 实验 / 结果 / 结论）自动落盘、
自动建索引、自动跨时间查询。这一节是这层分工体现在设计纲领上的两个层面：我们做
什么，以及我们不做什么。

### 我们做什么

**1 个硬性约束 —— rundown 是一个轻量化框架。**

```
torch / numpy / pandas / pyyaml / safetensors    # 5 个运行时依赖, 到此为止
```

optuna 是 `[extras]` 可选, 不污染核心. **不引入** Hydra、pydantic、click、
tensorboard、wandb。选择理由：每个被拒绝的包都在试图替使用者做「更聪明的事」，
而智能默认值的代价是把你的工作流整个吞进去。拒绝这个诱惑，框架才能保持 *5 个
依赖* 级别的轻量。

**3 个承诺（与「为什么」一节的 5 个痛点一一对应）。**

1. **一年后还能复算**（痛点 A、B 的工程保障）
   跑完的瞬间，复算所需的所有信息都在运行目录里。不依赖 git log、不依赖外部数据库、
   不依赖人记。「表 4 第二行那个数字来自哪个数据划分 / 哪个随机种子 / 哪份代码 / 哪台机器」——
   拷走目录就能全部回答。

2. **manifest 是事实本身**（痛点 C 的索引机制）
   `manifest.yaml` 写盘即合同。跨运行查询靠 glob manifest 自己，不建数据库。manifest
   既是数据，也是索引。「拷走即可验证」——这是承诺的物理形态，不是营销措辞。

3. **Agent 友好，人类也好用**
   两者不冲突时都好；冲突时选 Agent 友好的方案。`--format json` 永远可用，所有动词
   都支持 Python 调用（`from rundown.cli import main` / `from rundown.inspect_run
   import load_run`），产物结构干净到 Agent 几十行代码就能消费完整状态。
   Agent 集成是兼容层，**不是**主干 —— 主干仍是「个人开发者 + 轻量化框架」。

### 我们不做什么

**2 个拒绝（框架的设计选择，可验证）。**

1. **不做 UI 看板**
   看板好看，但不是「复算」。看板需要服务器 / 数据库 / 鉴权，必然引入运维依赖，会破坏
   「拷走即可验证」的承诺。需要看板时，把 MLflow 嫁接过来（5 行适配器，在 Phase 2）。

2. **不在省空间上动歪脑筋**
   硬链接 / 共享池 / DVC 那类省空间方案，看起来省几 MB，但你拷走产物或跨机器复制时
   它们会悄悄失败。产物在边界处就撒了谎。rundown 宁可多存几 MB 代码快照，
   让边界处永远诚实地自描述。

**边界清单（rundown 不是这些）**

> 这份清单与「2 个拒绝」的区别：拒绝是 *设计选择*（原则上可改），边界清单是 *领域边界*
> （原则上不该做）。想加这些功能，请用 MLflow / Optuna / Hydra 等专门工具。

- **不是特征工程框架。** 窗口化、列名、掩码、时间特征——这些由你声明，rundown 不感知。
- **不是实验追踪器。** 一次运行一个目录。要看板，用 MLflow 适配器。
- **不是调参工具。** Optuna 是 [extras]（当前未接线，预留给 Phase 2）; `rundown experiment run` 目前只跑 cartesian 笛卡尔积扫描。
- **不是部署平台。** rundown 的产物是 `.safetensors`，不是 Docker 镜像。
- **不是深度学习框架。** 没有自动微分的魔法，没有 DSL，没有构建系统。

它是三样东西：**注册表协议（Registry）/ 训练引擎（Engine）/ 产物格式（Artifact）**。
下面那节展开。

---

## rundown 的子系统

5 个子系统, 与「为什么」一节 5 个痛点一一对应. 每个子系统给出具体入口、机制和一句话
特征.

### Registry —— 回应痛点 A (训练脚本硬编码 import 难解耦)

**痛**: 想换数据集 / 模型 / 损失函数 / 优化器版本时, 训练脚本里写满了硬编码 import
(`from mymodels.transformer import MyTransformer` 等), 只能全项目搜改, 容易漏.

**机制**: `configs/{datasets,models,experiments}.yaml` 用别名命名 (`dataset: etth1`,
`model: lstm`), rundown 的 `Registry.load_*()` 用点分路径反射 (`factory.target:
examples.baseline.model.SequenceLSTM`) 把别名映射到实际类. 改 YAML 就能换数据集 / 模型,
不动训练脚本.

**入口**: 在 `configs/models.yaml` 里加一条记录, 把 `factory.target` 指向你的 PyTorch 类.

### Engine —— 回应痛点 B (跑完忘了超参数 / 哪个提交)

**痛**: 跑完忘了这次的学习率 / 隐藏层 / 层数是多少, 对应哪个 git 提交. git log
不直观 (提交信息 ≠ 运行配置).

**机制**: `Engine.fit(provider, model, run_dir)` 入口, 训练循环里强制记录
`config` (你传入的 optimizer / epochs) + `env.git_commit` + `env.git_dirty`, 落盘到
`run_dir/manifest.yaml`. 配 BPTT / early stop / grad clipping 等训练策略.

**入口**: 调 `rundown.train` 这条 CLI 即可. 训练逻辑封装在 `rundown/engine.py`,
不替使用者写模型.

### Artifact —— 回应痛点 C (扫描完找不到最佳)

**痛**: 跑 50 次扫描, 想问「这次 best_val_mse 来自哪个运行」要 `grep` manifest
+ 屏幕截图 + 笔记.

**机制**: 每次运行落 `manifest.yaml` + `weights.safetensors` + `curves.safetensors`
+ `predictions/test_best.safetensors` + `events.jsonl`. `manifest` 既是数据也是索引,
跨运行查询靠 `rundown list-runs --above-metric best_val_mse 2.0` 走文件系统 glob,
不建数据库.

**入口**: 跑完一次训练, 产物自动落到 `artifacts/<run_id>/`, `rundown list-runs` / `compare`
/ `inspect` 都是只读.

### Inspect —— 回应痛点 D (运行目录人读不懂)

**痛**: 自己 6 个月后翻自己的运行目录 (manifest + safetensors) 也读不懂 ——
manifest 是机器格式, 人脑需要一番转换才能理解.

**机制**: `rundown inspect <run_id>` 渲染出 README 风格的 Markdown, 把 manifest +
curves + predictions + 代码快照元信息投影成人脑可读的 7 段: 目的 / 设置
/ 环境 / 结果 / 如何评估 / 磁盘产物. 这是产物与人类阅读之间的标准转换器,
也是 Agent 第一次接手陌生运行的入口.

**入口**: `rundown inspect <run_id>` 输出到 stdout (可重定向到 `.md` 文件); 缺 pyyaml /
safetensors 时降级.

### Fact —— 回应痛点 E (主观结论难追溯)

**痛**: 跑完 30 次, 「method_v2 比 baseline 强 12%」这类主观结论自己记得, 写论文 /
给审稿人 / 给半年后的自己追溯时却没法验证.

**机制**: 与前面 4 个子系统不同——这是一个**手动兜底出口** (框架不写
事实). `artifacts/_facts/main_facts.yaml` 索引每条结论, `fact_<id>.md` 存一条结论.
manifest 自动记事实, fact 文件写判断, 用 4 个状态 (待确认 / 已确认 / 已证伪 /
已取代) 走生命周期.

**入口**: 直接 `cat artifacts/_facts/main_facts.yaml` 读, 用 `write_file` 写事实卡.
若加 CLI 反而违背原则——事实与判断的边界要保持人工覆盖.

---

## 快速入手

不管你是 bash / Python / Agent 用户, 都是这 3 步.

**bash** —— 跑一次训练并拿到产物:

```bash
uv sync                                                       # 建 venv + 装 5 个运行时依赖
uv run rundown train --dataset etth1 --model lstm --variant small --epochs-max 5
ls artifacts/lstm__varsmall__etth1__seed0/                    # manifest + weights + curves + predictions
```

**Python** —— 读已存运行, 渲染成 README 风格的 Markdown:

```python
from rundown.inspect_run import load_run, render_run_markdown
print(render_run_markdown(load_run("artifacts/lstm__varsmall__etth1__seed0")))
```

**Agent** —— 跨运行抓指标, 不用写 Python:

```bash
uv run rundown list-runs --format json | jq '[.[] | {id: .run_id, mse: .metrics.best_val_mse}]'
```

下面章节是各模块具体怎么用 (三件套配置 + 8 个动词 + 目录布局 + 产物例子).

---

## 安装 (uv)

```bash
uv sync                                          # 建 .venv + 装依赖 (生成 uv.lock)
uv run rundown train --dataset etth1 --model lstm  # 用 .venv 跑命令
uv run pytest                                    # 跑测试
```

可选: `uv sync --extra optuna` 仅供未来 optuna 模式占位; 当前 `experiment run` 只支持 cartesian 笛卡尔积扫描, 不需要 optuna.

不用 uv 也行:

```bash
pip install -e .[dev,optuna]
python -m rundown.cli train --dataset etth1 --model lstm
```

---

## 第一次运行 (3 行命令)

rundown 已经内置一份完整 ETTh1 (17k 行小时级基准数据集) + 2 层 LSTM 模型,
直接跑就有结果:

```bash
uv run rundown train --dataset etth1 --model lstm          # 基础 LSTM
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

跑完去 `artifacts/lstm__varsmall__etth1__seed0/` 看产物——7 个文件 + 共享的 `_shared/` 代码快照,
manifest 一眼看穿所有配置.

---

## 三件套: 数据集 / 模型 / 实验

rundown 把这三件事做成「在 YAML 里声明 → CLI 一键跑」:

### 1. 数据集 (configs/datasets.yaml)

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
- `provider_factory` 反射调用的构建函数 (签名约定: `builder(source_path, splits, target_col, **overrides) -> {train, val, test}`)
- `splits` 时间切分比例 (总和 = 1)

### 2. 模型 (configs/models.yaml)

```yaml
models:
  lstm:
    factory:
      target: examples.baseline.model.SequenceLSTM
      kwargs: {in_dim: 6, hidden: 64, num_layers: 2}
    variants:                    # 便捷预设 — 同一模型不同规模的组合
      small: {hidden: 32, num_layers: 1}
      large: {hidden: 128, num_layers: 2}
      deep:  {hidden: 64, num_layers: 3}
```

- `factory.target` 反射绑定, 注册时立即校验
- `factory.kwargs` 固定结构 (由数据集决定, 如 `in_dim`)
- `variants` 便捷预设 — CLI `--variant large` 切换
- **不在 models.yaml 里**: `optimizer` / `epochs_max` / `dataset` / 训练超参 — 见下方边界表

### 3. 实验 (configs/experiments.yaml) — 扫描

```yaml
experiments:
  # 笛卡尔积: 枚举 variants (顺序跑)
  etth1_lstm_size_sweep:
    dataset: etth1
    model: lstm
    epochs_max: 5
    optimizer: {kind: adam, lr: 0.001}    # 可选, 不写走 defaults.DEFAULT_OPTIMIZER
    variants: [small, large, deep]        # 引用 models.lstm.variants 的键
```

跑扫描:

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

### 模型与实验的边界

| 字段 | 归属 | 谁消费 | 例 |
|------|------|--------|-----|
| `factory.target` | **models** | 注册表反射 | `examples.baseline.model.SequenceLSTM` |
| `factory.kwargs` | **models** | `cls(**kwargs)` | `{in_dim: 6}` (固定结构) |
| `variants` | **models** | `--variant large` | `{small: {hidden: 32}, large: {hidden: 128}}` (便捷预设) |
| `optimizer` (默认) | defaults.py | 单次训练的兜底 | `defaults.DEFAULT_OPTIMIZER = {kind: adam, lr: 1e-3}` |
| `optimizer` (覆盖) | **experiments** | 扫描子进程 | `optimizer: {kind: adam, lr: 0.0005}` |
| `dataset` | **experiments** | provider 反射 | `etth1` |
| `model` | **experiments** | 注册表查询 | `lstm` |
| `epochs_max` | **experiments** | 引擎 | `5` (扫描跑短一点) |
| `batch_size` / `steps_per_epoch` | **experiments** | 引擎 / provider | 覆盖默认 |
| `variants` (列表) | **experiments** | 扫描笛卡尔积 | `[small, large, deep]` (引用 models 的键) |
| `--kwargs key=value` | CLI 透传 | 模型工厂 (`ModelEntry.build` → `cls(**kwargs)`) | 临时超参自由传 |

**核心原则**:

- **models** = 「一个 PyTorch 类 + 默认 kwargs + 便捷预设」。它不知道训练时长、不知道用哪个优化器、不知道跑什么数据集。
- **experiments** = 「一次扫描的设置」。它决定用哪个模型 / 数据集 / 优化器，训练多久，扫描哪些变体。
- **defaults.py** = 「单次 `rundown train`（无实验）的唯一真值来源」。扫描时由实验覆盖。
- **`--kwargs`** = 「运行时透传，与 YAML 正交」。Agent / 临时调一组超参用，绕过 variants 限制。

**老配置向后兼容**: models.yaml 里的 `optimizer:` / `epochs_max:` 字段会被静默忽略（不再读取）。新配置应删掉这些字段——见 `test_load_models_ignores_optimizer_field`。

### 复现最佳试验

`[summary]` 输出包含**可直接复制的复现命令**——用同样的数据集 / 模型 / 变体 / 随机种子 / 最大轮数 / 输出目录去跑，会落到**同一个 run_id 目录**（CLI 默认 run_id 规则：`{model}[__var{variant}]__{dataset}__seed{N}`）。

扫描本身不再预拼 run_id 前缀；试验的 run_id 与单次 `rundown train` 完全一致，方便复现和 read-run。

### 可复现性说明

`engine._set_seed(seed)` 已设 `torch.manual_seed` + `np.random.seed` + `set_num_threads(1)` + `cudnn.deterministic=True` + `use_deterministic_algorithms`。

但 **PyTorch CPU 上的 LSTM 仍可能有 ±5% 的 val_mse 漂移**——CPU 上 cuDNN 不工作，LSTM 内部浮点累加顺序受线程数影响，单线程只能减小漂移而不能消除。要严格可复现，需要 GPU + cuDNN 确定性模式（由使用者自行开启）+ 提交代码 + 锁定依赖版本（`uv.lock` 已锁）。

框架能保证的是「**同一种子下 manifest 字段、events.jsonl 序列、weights 形状完全一致**——预测 / 指标可重算、可比较」——具体数值精度依赖 PyTorch 行为。

每个试验是一次独立运行，落自己的 artifacts 目录，**互不影响**。扫描失败 / 重启 / 换种子都能局部重跑。

---

## 八个动词

```bash
rundown prepare --dataset <alias>              # 加载数据集, 算内容哈希, 注册
rundown train --dataset <ds> --model <m> [...] # 跑一次训练
rundown experiment run <name>                  # 跑注册过的扫描 (笛卡尔积)
rundown read-run <run_id> [--metric X]         # 读已有运行, 算新指标
rundown list-runs [filters]                    # 列出 artifacts/*/manifest.yaml
rundown compare <run_a> <run_b>                # 两个运行的 manifest 差异
rundown inspect <run_id|--all> [--since-days N] # 读运行, 输出 README 风格的 Markdown
rundown analyze --analyzer X <run_id>          # Phase 2 占位
```

每个动词同时支持 CLI / Python (`from rundown.cli import main`) / Agent 调用.

### 给 Agent 调用方 (调用契约)

**应该做**:

- `--format json` 永远可用 (list-runs 是最先实现的, 后续动词会陆续加); 不解析 stderr 进度行.
- `artifacts/<run_id>/manifest.yaml` 是唯一真值来源 — 不要尝试从 events.jsonl 推断运行状态 (events 是事件流, 不是结构定义).
- `rundown inspect <run_id> -q` 拉一份 README 风格的 Markdown — 含目的 / 设置 / 环境 / 结果 / 曲线 / 预测 / 事件 / 如何评估 / 产物共 8 段, 可喂给 LLM.
- Python 入口: `from rundown.cli import main(argv)` / `from rundown.inspect_run import load_run, render_run_markdown`. 独立可用, 不需要 train / engine.
- 运行排序用 manifest.created_at, 用 `--since-days N` 切时间窗 (UTC).

**不要做**:

- 不要硬编码 `run_id` 格式去解析 (例如 `lstm__varsmall__etth1__seed0`); 从 manifest.run_id 字段读.
- 不要假定产物路径不变; 唯一可靠入口是 `manifest.config` 里 `out_dir` / `_shared/code_<sha>/` 的引用.
- 不要解析 stderr 上的进度行 — 那些 stderr 是给人看的, 会被 `--quiet` 压掉.
- 不要用空格拆分 `events.jsonl` 的行 — 用 `line.split("  ", 2)` (Agent 友好格式里字段间是双空格).

**当 CLI 不可用时** (例如作为子模块调用, 无系统 Python): Python 入口仍然可用; `from rundown.inspect_run import load_run` 不依赖 CLI 解析.

### 全局 `--quiet` / `-q`

压住 stderr 上的 info 行 (events.jsonl 仍写, 产物完整). 适合 CI / Agent 自动化调用.

### `--kwargs KEY=VALUE` — 透传任意超参给模型工厂

`models.variants` 是**便捷预设**（`small` / `large` / `deep` 是常见规模组合）。
**临时超参**用 `--kwargs` 自由传, 绕过 variants 限制: `--kwargs` 只把键值
合并到模型工厂的 `**kwargs`（`ModelEntry.build` → `cls(**kwargs)`），不会
触达 optimizer / 引擎 / 数据加载器。

```bash
# 透传多个 — 合并顺序: factory_kwargs < variants < --kwargs (后者覆盖)
uv run rundown train --model lstm --variant small \
  --kwargs hidden=128 --kwargs num_layers=2

# 临时调一组超参 (不用预注册 variant; 都进模型工厂)
uv run rundown train --model lstm \
  --kwargs hidden=$HIDDEN --kwargs num_layers=$NLAYERS
```

**注意**: optimizer 学习率走 `--lr`（不是 `--kwargs lr=...`）。`--kwargs`
只到达模型工厂；传 `--kwargs lr=...` 一般会让 `cls(**kwargs)` 抛 `TypeError`。

值类型自动判断: `int` / `float` / `bool` / `str`.

不存在的 kwargs 键 → 工厂自己抛 `TypeError` (不静默吞掉 — 跟「拒绝 3: 不擅自替你修」一致).

`--kwargs` 不影响代码快照哈希 — 同代码不同 kwargs → 同 sha → 自动去重.

```bash
# 任何位置都生效 (绕过 argparse 限制, main() 手动从 argv 抓)
rundown train ... --quiet
rundown --quiet experiment run exp1
rundown -q list-runs
```

stderr 行格式 (新, Agent 友好):

```
2026-10-16T21:23:35  step_completed  {"epoch": 2, "step": 10, "train_loss": 0.42, "grad_norm": 1.5}
```

- `{t_iso}` ISO 时间戳
- `{kind}` 事件名 (便于 grep)
- JSON 字段值 (复杂值安全: 带空格的字符串 / list / 嵌套 dict 不会破坏拆分)

Agent 解析路径: `line.split("  ", 2)` → `(t_iso, kind, fields_json)` → `json.loads(fields_json)`.

### 查询运行: list-runs / compare

扫描跑完后, 你想在 `artifacts/` 里找最佳运行, 或对比两次跑的差异, 不需要写脚本:

```bash
# 列出所有运行 (表格格式, 字段: run_id / dataset / model / variant / best_val_mse)
rundown list-runs

# 过滤: dataset=etth1 + model=lstm + best_val_mse < 2.0
rundown list-runs --dataset etth1 --model lstm --below-metric best_val_mse 2.0

# 给 Agent 用的 JSON
rundown list-runs --format json
```

```bash
# 两个运行的 manifest 差异 — 只输出有差异的字段, 数值显示 Δ
rundown compare lstm__etth1__seed0 cnn__etth1__seed0
```

list-runs 走文件系统 (`glob artifacts/*/manifest.yaml`), 不建数据库——manifest 自己就是索引.

### 读一个运行: `inspect` — README 风格的 Markdown

`list-runs` 给一行摘要; `inspect` 给一份完整 README: 这次运行的目的 / 设置 / 环境 / 结果 / 怎么复算 / 产物清单. stdout 是 Markdown, 直接重定向到 `.md` 文件.

```bash
# 单个运行
rundown inspect lstm__varsmall__etth1__seed0 > lstm__varsmall__etth1__seed0.md

# 批量 (按创建时间过滤最近 7 天)
rundown inspect --all --since-days 7 > weekly_runs.md

# 批量 + 按数据集 / 模型 / 变体过滤
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

- **只读** — 不动 `artifacts/`, 不写日志, 不创建文件 (除非你 `>` 重定向).
- **降级** — 缺 `pyyaml` / `safetensors` 时只读 `manifest.yaml`, 仍能渲染「设置 / 环境 / 结果」段.
- **Agent 友好** — stdout 是干净的 Markdown, 可以 `cat` / `grep` / 喂给 LLM.
- **Python 入口** — `from rundown.inspect_run import load_run, render_run_markdown` 直接当库用, 不一定要走 CLI.

---

## 自定义 trainer (Custom trainers)

rundown 的 4 层架构:

```
artifact.py        产物: manifest / weights / curves / predictions / events
session.py         RunSession — 一次 run 的生命周期与统一收尾
trainers.py        Trainer 协议 + 内置 GradientTrainer 适配器 + registry
engine.py          保留: GradientStrategy (内置梯度训练)
registry.py        YAML: dataset / model / experiment / trainer
cli.py             CLI: train / experiment / list-runs / inspect …
```

第 1 层是「记录 / 不变量」(artifact 边界), 第 2 层是「训练语义」(loss / stage /
多模型). `RunSession` 把第 1 层收口, `Trainer` 协议把第 2 层抽象. 默认 trainer
(`gradient`) 写在 `engine.py`; 自定义 trainer 通过 `--trainer` / `--trainer-config`
接入, **不必** 改 `engine.py`.

### `Trainer` 协议

```python
from rundown.trainers import TrainResult  # 函数内 import

class MyTrainer:
    name = "my_trainer"
    required_provider_keys = ("x", "y", "warm_start")
    primary_metric = "best_val_mse"        # 实验汇总读这个

    def validate(self, providers): ...     # 训练前契约校验; 默认 no-op
    def fit(self, *, model, providers, ctx, session) -> TrainResult:
        # 训练循环: 走 session.emit / session.checkpoint_best /
        # session.write_curves / session.write_predictions / session.finalize.
        # 永远不调 save_manifest / set_active_log / clear_active_log.
        ...
        session.finalize(
            metrics={"best_val_mse": ..., "epochs_actual": ...},
            predictions_primary="test_best",
            epochs_actual=...,
        )
        return TrainResult(metrics=..., epochs_actual=..., ...)
```

### `RunSession` 职责

- `start(run_info=...)` — 落初始 manifest, 激活 events, 捕获 env, emit `run_started`.
- `emit(kind, **fields)` — 训练中事件 (epoch_completed / stage_started 等).
- `checkpoint_best(state)` / `write_weights(state, best=False)` — 写 weights_best / weights.
- `write_curves({"train_loss": tensor, "val_loss": tensor})` — 写 curves.safetensors.
- `write_predictions("test_best", {...})` — 写 predictions/test_best.safetensors.
- `finalize(metrics, ...)` — 写 manifest.metrics / env, emit `run_completed`, 幂等.
- `fail(exc)` — 失败路径: emit `run_failed`, 不写 `run_completed`.

`trainer` 自己**不必** import rundown 任何东西 (除 `from rundown.trainers import
TrainResult` 这一个 import). session 是 trainer 与 artifact 之间的唯一桥梁.

### 跑模板示例

`examples/custom_trainers/` 给三个 runnable 模板 — 自定义 loss / 分阶段训练 /
联合训练 (主模型 + 辅助模型). 任何一个都能直接跑:

```bash
uv run python -m rundown.cli train \
    --dataset etth1 --model lstm \
    --trainer examples.custom_trainers.custom_loss.CustomLossTrainer \
    --trainer-config examples/custom_trainers/custom_loss.yaml \
    --epochs-max 1
```

YAML 配置 (JSON 也行):

```yaml
# examples/custom_trainers/custom_loss.yaml
huber_delta: 1.0        # HuberLoss 的 delta
l1_weight: 0.001        # |yhat|.mean() 的权重, 鼓励预测趋近 0
```

Experiment sweep 透传:

```yaml
# configs/experiments.yaml
experiments:
  custom_loss_sweep:
    dataset: etth1
    model: lstm
    epochs_max: 1
    variants: [small]
    trainer: examples.custom_trainers.custom_loss.CustomLossTrainer
    trainer_config: {huber_delta: 0.5}
    metric: best_val_loss    # 汇总用这个
```

`experiment run` 启动时把 `--trainer` / `--trainer-config` 传给 subprocess,
subprocess 端按 `Trainer` 协议解析 → 跑 → 写产物.

### 多模型权重前缀 (joint trainer 专属)

`JointTrainer` 同时训主 + 辅两个模型, 写进同一个 `weights.safetensors` /
`weights_best.safetensors`. key 加前缀防冲突:

```
main.<param>     主模型 (manifest.config.primary_metric 走它)
aux.<param>      辅助模型
```

读回时按前缀拆:

```python
from safetensors.torch import load_file
state = load_file("artifacts/.../weights_best.safetensors")
main_only = {k[len("main."):]: v
             for k, v in state.items() if k.startswith("main.")}
model.load_state_dict(main_only)
```

约定见 `examples/custom_trainers/joint.py` 和 `examples/custom_trainers/README.md`.

### Caveats — 上手前请读

1. **`python -m rundown.cli` 模式** vs **`rundown` console script**.
   - `python -m rundown.cli` 把 cwd 加到 `sys.path`, `examples.custom_trainers.X`
     直接 import 即可.
   - 全局装的 `rundown` 命令没把项目根加 `sys.path`, 需要 `pip install -e .`
     或 `PYTHONPATH=.`.
2. **代码快照只含 tracked files**. trainer.py 必须提交进 git, 否则 snapshot
   抓不到, 1 年后用 `_shared/code_xxx` 还原不出来.
3. **trainer.py 不许 import rundown** (除 `rundown.trainers.TrainResult`).
   trainer / 模型都不知道 rundown 存在 — rundown 来找它们, 不是反过来.

`inspect_run` 自动读 `manifest.config.trainer` 在 Setup 段显示 `- **Trainer**:
\`custom_loss\`` 一行; 通用 metrics / curves 表让任意 `stage1_train_loss` /
`val_loss` / `aux_train_loss` 都被渲染, 不破坏旧 artifact 的输出. 详见
[`examples/custom_trainers/README.md`](examples/custom_trainers/README.md).

---

## 目录布局约定

跑 ML 实验的项目根目录有 4 类互不重叠的子目录. 区分的维度是 **生命周期** (长期 vs 临时), 不是 **作者** (Agent vs 人):

| 区域 | 生命周期 | git | 角色 |
|------|----------|-----|------|
| `src/` `tests/` `configs/` | 长期 | **纳入版本控制** | 项目代码 — 进提交 / 论文 / 发布 |
| `examples/<name>/`      | 长期 | **纳入版本控制** | 参考示例 — 一套最小可跑的模型 + 数据 + 训练 (`examples/baseline/` 默认存在). 你写的真实代码不必放这里 |
| `artifacts/` | 长期 (量大) | **忽略** | rundown 跑出来的产物 — manifest + weights + curves + predictions |
| `.scratch/` | 临时 | **忽略** | 过程工具代码 — Agent 会话临时产物 / 人工调试脚本 / 实验探针 |

> Agent 与人**都可以**写 `src/`, 也都可以写 `.scratch/`. 是否进入产品由生命周期决定, 不是由作者决定.

### `examples/<name>/` 怎么用

它是 **rundown 自带的参考示例** — 一套最小可跑的模型 + 数据 + 训练.
跑 `python -m examples.baseline.train` 就能在源码层面看完 rundown 的端到端流程.

你**真实项目**的模型 / 数据加载器不必放这里. `configs/models.yaml` 里写的是
点分路径 (`examples.baseline.model.SequenceLSTM`), 你写 `mymodels.transformer.MyTformer`
也照样能用. `examples/` 是 **示例锚点, 不是框架强制的家**.

适配指南在 `examples/baseline/README.md` 里:「把 `data.py` / `model.py` 换成你的,
`train.py` 改 RUN_ID / WARM_START 等常量」.



### `.scratch/` 怎么用

子目录名与 Agent 会话 ID 一一对应, 避免多个会话互相踩踏:

```
.scratch/
├── agent_cursor_abc123/    # Cursor 会话 abc123 的临时脚本
├── agent_cline_xyz789/     # Cline 会话 xyz789 的探索代码
├── debug_2024_10_17/       # 人调试一天的临时查询
└── probe_lr_sweep/         # 实验性 lr 扫描探针, 留不留看心情
```

**谁负责执行**: Agent 框架自己. Cursor / Cline / Devin 都有会话工作目录配置, 把会话临时文件指向 `.scratch/<session_id>/`. rundown **不读不写** 这个目录, manifest 也不捕获 — 这是集成约定, 不是框架的不变量.

### `.gitignore` 推荐

```gitignore
# rundown 产物
artifacts/

# 过程代码 (Agent 临时 / 人工调试)
.scratch/

# 通用
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
```

拷走项目代码 (`git clone`) → 默认不会拷 `.scratch/`, 也不会拷 `artifacts/`. 这是「拷走即可验证」的物理保障: 长期产物在提交里, 临时文件留本地.

### 与 rundown 哲学的关系

- **承诺 2 (拷走即可验证)** — `.scratch/` 默认不拷走, 长期 / 临时自动分离, 不用框架强制.
- **拒绝 3 (不擅自替你修)** — Agent 会话 ID 是 Agent 框架自己声明的, 不由 rundown 强加. rundown 不假设 "会话 ID 在 manifest 里", 因为会话边界是 Agent 框架的语义, rundown 不抢.
- **1 个硬性约束** — 目录约定是零工程成本解. 不引新机制, 不加 manifest 字段. 复用现有 `artifacts/` 已有的 gitignore 规则, 自然推广.

---

## 运行跑完后的目录

```
artifacts/
├── _shared/                              # 扫描 / 多次训练同代码共享的快照池
│   ├── code_<sha8>/                      # git 仓库项目: `git archive` 输出 tar 目录 (首选)
│   ├── code_<sha8>.meta.json             # tar 模式的元数据 (sha256 / size_bytes / num_files), sweep (experiment run) 写
│   ├── code_<sha8>.zip                   # 非 git 仓库项目: zip 全目录 (排除 artifacts/.venv/__pycache__)
│   └── code_<sha8>.zip.meta.json         # .zip 模式的元数据伴生文件, sweep (experiment run) 写
└── <run_id>/                             # 每次运行一个目录
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

`<sha8>.meta.json` 是 sweep (`rundown experiment run`) 写的, 单次 `rundown train` 不会产生.
tar 模式的 `code_<sha8>/` 是裸目录, 元数据用同级 `code_<sha8>.meta.json` 承载;
zip 模式同理, `.meta.json` 与 `.zip` 平级.

**默认行为**: `rundown train --out-dir artifacts` 跑完一次后, 快照落在 `artifacts/_shared/code_xxx/` 或 `code_xxx.zip` (取决于是否 git 仓库), run_dir 内**没有**自己的代码副本. 多次训练同代码 → 自动跳过写入, 物理上只有一份.

**`--run-dir` 模式**: 快照直接落 run_dir 内 (单次运行自给自足, 不参与 `_shared/` 池).

**git 仓库项目**: 快照走 `git archive` (仅纳入版本控制的文件, 自动遵循 `.gitignore`), 落 `code_<sha8>/` 目录. **非 git 仓库**: zip 全目录 (排除 `artifacts/` `.venv/` `__pycache__/`), 落 `code_<sha8>.zip`.

**`--no-snapshot`**: 关闭代码快照. 不推荐——一年后找不到当时的代码.

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
  predictions_primary: test_best     # 读取方读哪个 predictions
  step_event_every_n: 10
  grad_norm_threshold: 100.0

metrics: {best_val_mse: 2.27, epochs_to_best: 11, epochs_actual: 15}

# 自动记录: 跑这次时宿主机环境是什么
env:
  rundown_version: 0.1.0
  python: 3.10.21
  platform: Windows-10-10.0.26300-SP0
  torch: 2.14.1+cpu
  numpy: 1.26.4
  argv: [rundown, train, --dataset, etth1, --model, lstm]
  cwd: E:\rundown
  git_commit: a1b2c3d     # 缺省时省略 (非 git 仓库 / git 未装)
  # 代码快照 — 默认开; git 仓库走 git archive (仅跟踪文件), 否则 zip 全目录
  code_snapshot:
    format: tar            # 或 "zip" (非 git 仓库)
    tracked_only: true     # git archive 路径为 true, zip 路径为 false
    sha256: 421778...      # 包内容哈希, 与 git_commit 互为校验
    sha256_short: 421778bc # sha256[:8], 用作 archive_path 文件名
    size_bytes: 593071
    num_files: 36
    archive_path: ../_shared/code_421778bc.zip  # 相对 run_dir, 读取方直接 cat 即可
```

读者路径: `cd artifacts/<run_id> && cat ../_shared/code_xxx.zip | less` —— manifest 直接告诉代码在哪.

---

## 目录结构

```
rundown/
├── __init__.py
├── cli.py              # 8 个动词: prepare / train / experiment / read-run / list-runs / compare / inspect / analyze
├── registry.py         # YAML 加载 + 反射绑定 + 别名校验 (datasets / models / experiments)
├── engine.py           # GradientStrategy.fit() + per-step events + 自动记录 env
├── artifact.py         # 写/读 manifest + weights + curves + predictions
├── _env.py             # capture_env() + snapshot_code() (自动记录宿主机环境 + 代码)
├── events.py           # emit() 全局事件发射器
├── ids.py              # 别名 / run_id 正则校验
├── cli_query.py        # list_runs / compare_runs (glob manifest.yaml 直接读)
├── experiment.py       # 扫描: 笛卡尔积 (子进程调 train)
├── defaults.py         # DEFAULT_BATCH_SIZE / DEFAULT_LR / DEFAULT_OPTIMIZER 等唯一真值来源
├── inspect_run.py      # rundown inspect: 只读运行 → README 风格 Markdown
├── facts/              # fact 系统: main_facts.yaml 索引 + 子 fact .md
└── (12 个包文件 / 子包, 约 2600 行)

configs/
├── datasets.yaml
├── models.yaml
└── experiments.yaml

artifacts/              # 运行目录, 一个运行一个文件夹
├── _shared/            # 多次训练 / 扫描共享的代码快照池
└── <run_id>/
examples/
└── baseline/           # ETTh1 内置 + LSTM + SlidingWindowProvider
    ├── data/ETTh1.csv
    ├── data.py
    ├── model.py
    ├── train.py        # 薄封装: 子进程调 rundown.cli train
    └── test_baseline.py    # 4 个测试: 全链路 smoke
tests/                  # pytest, 118 个测试 (与 examples/baseline 合计 122)
├── fixtures.py         # ToyNet + RandomProvider
├── test_smoke.py       # 75 个测试: 核心流程 / cli / registry / artifact / snapshot / events
├── test_facts.py       # 15 个测试: facts 系统路径 / 模板 / 哲学边界
├── test_inspect_run.py # 16 个测试: load_run / render / since_days / CLI
└── test_time_and_metrics.py # 12 个测试: 指标异常 / UTC 时间戳 / since-days 解析
AGENTS.md               # 给 AI Agent 看的约束文档
pyproject.toml          # PEP 621 元数据 + [optuna] extras
uv.lock                 # 锁定的依赖版本 (提交进 git)
```

13 个包文件 / 子包, 约 2500 行. 完整跑通 ETTh1 (17k 行真实数据) + 2 层 LSTM,
测试 MSE ≈ 1.10 (标准化空间), **对齐 ETT 论文基准**.

---

## 什么时候用 rundown

- 你跑了 ≥ 3 次实验, 开始记不住哪次用了什么配置 → 用 rundown
- 你的论文要发出去, 你担心 18 个月后复现不出来 → 用 rundown
- 你的实验代码会被同事 / 审稿人 / Agent 接手 → 用 rundown
- 你想批量扫描 (当前支持笛卡尔积) 而不想手写循环 → 用 rundown
- 你只是想跑 1 次演示, 跑完就丢 → **别用, 直接用 PyTorch**

## 什么时候**不要**用 rundown

- 你是 PyTorch 初学者, 在做"如何写第一个训练循环"的练习 → 直接写
- 你的项目有强 MLOps 需求 (A/B 部署、流量切分、监控告警) → 用 MLflow / Ray
- 你的数据是图片 / 文本, 没有时序结构 → rundown 的窗口化对你没有意义
- 你的项目已经定型, 跑 5 年不会改 → 别折腾, 现状能跑就行

---

## 附录

这一段是参考性质的内容. **约束、规则、模板、命名规范、禁止的 import 都不在这里重复** — 它们在 `AGENTS.md` (项目给 AI Agent 看的约束文档). 本附录只放 README 涉及的指引与概览.

### 指引列表

| 内容 | 位置 |
|------|------|
| 模块边界 / 谁能 import 什么 | `AGENTS.md` §1 |
| 反模式 (不要做的 8 件事) | `AGENTS.md` §2 |
| 命名规范 (dataset_alias / model_id / run_id 正则) | `AGENTS.md` §3 |
| 测试要求 / 跑通门槛 | `AGENTS.md` §4 |
| 禁止的 import (lint 规则: pickle / torch.save / matplotlib / seaborn / plotly) | `AGENTS.md` §5 |
| `rundown.ids.assert_valid` (写入前校验) | `rundown/ids.py` |
| 适配器模式 (MLflow / WandB 等怎么接 rundown) | `examples/baseline/README.md` (「Adapting to your project」段) |
| inspect 输出结构 | `rundown/inspect_run.py` (RunIntel dataclass) |
| Manifest 结构 | `rundown/artifact.py` + `examples/` 的实际 manifest 输出 |

### 一些「不用进 README」的隐性规则

- **run_id 格式**: `^[a-z][a-z0-9_]+__[a-z][a-z0-9_]+__seed\d+$` — 详见 `AGENTS.md` §3
  及 `rundown.ids.assert_valid`.
- **目录约定** (生命周期维度): 详见前面「目录布局约定」一节, 本附录不重复.
- **.gitignore 推荐块**: 详见「目录布局约定」一节内嵌块.

### 给读者的阅读建议

README 与 AGENTS.md 应一起读, 但顺序很重要:

1. **本文档** — 先理解设计哲学 + 5 个子系统 + 怎么上手. 给「做什么」/「不做什么」/「机制对应痛点」的叙述性答案.
2. **`AGENTS.md`** — 当你要 *写* 代码进来 (新模型 / 新数据集 / PR), 读这个. 给你 lint / 规则 / 反模式的硬约束.
3. **`examples/baseline/`** — 当你要 *适配* rundown 到自己项目时, 这是唯一事实来源: 改 `data.py` + `model.py` + `train.py` 三处.

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
