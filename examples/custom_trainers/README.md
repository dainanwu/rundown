# examples/custom_trainers — 自定义 trainer 模板 (C4)

三个**能直接跑**的 trainer 模板, 覆盖 rundown 用户最常需要自己写的三种训练范式:

| 模板 | 解决什么需求 | `primary_metric` | 关键约定 |
|---|---|---|---|
| `custom_loss.CustomLossTrainer` | 自定义 loss (Huber + L1 magnitude penalty) | `best_val_loss` | val 用同一 loss, `best_val_mse` 附带 |
| `staged.StagedTrainer` | 分阶段训练 (Stage 1 全训练 → Stage 2 冻结 LSTM 训 head) | `best_val_mse` | `stages` list, 跨 stage 重建 optimizer |
| `joint.JointTrainer` | 主模型 + 辅助模型, 多优化器, 蒸馏 / 多任务 | `best_val_mse` | 权重用 `main.*` / `aux.*` 前缀 |

每个模板都满足 `rundown.trainers.Trainer` 协议, 走 `RunSession` 写标准产物
(manifest / weights / curves / predictions / events). 改一行超参就能跑.

---

## 1. Trainer 协议 (Trainer protocol)

`rundown.trainers.Trainer` 是结构子类型 — 不强制继承, 只要满足 shape 就被
registry 接受:

```python
class Trainer(Protocol):
    name: str                                  # e.g. "custom_loss"
    required_provider_keys: tuple              # e.g. ("x", "y", "warm_start")
    primary_metric: str                        # e.g. "best_val_loss"

    def validate(self, providers: Dict[str, Any]) -> None: ...
    def fit(self, *, model, providers, ctx, session) -> TrainResult: ...
```

**硬契约**:

- `no-arg` 可构造 (resolve_trainer 反射 `cls()`). 配置走 `ctx.trainer_config`
  (`--trainer-config` 注入).
- 训练产物一律走 `RunSession`: 写 weights / curves / predictions /
  finalize; 不直接调 `save_manifest` / `set_active_log` / `clear_active_log`.
- `session.finalize(...)` 必须调**恰好一次**, 返回 `rundown.trainers.TrainResult`.
- `providers["train"]("train")` / `providers["val"]("val")` /
  `providers["test"]("test")` 是 BatchProvider 契约 (callable, 返回
  `{"x": Tensor[B,T,F], "y": Tensor[B,T], "warm_start": int}`).

## 2. `RunSession` — 一次 run 的生命周期 owner

trainer 不写 manifest, 不激活 events. 这些都由 `RunSession` 拥有:

```python
session = RunSession(artifact=artifact, ctx=ctx,
                     trainer_name="custom_loss",
                     primary_metric="best_val_loss")
session.start(run_info={...})                       # manifest + run_started
session.emit("epoch_completed", ...)                # events.jsonl
session.checkpoint_best(state_dict)                 # weights_best.safetensors
session.write_weights(state_dict, best=False)       # weights.safetensors
session.write_curves({"train_loss": tensor})        # curves.safetensors
session.write_predictions("test_best", {...})       # predictions/test_best.safetensors
session.finalize(metrics={...}, epochs_actual=...)  # run_completed + 收尾
```

详见 `rundown/session.py`. trainer 永远不绕过它.

## 3. trainer_config 怎么传

| 形式 | 例子 | 说明 |
|---|---|---|
| CLI JSON 字符串 | `--trainer-config '{"huber_delta": 1.0}'` | 适合临时调一组超参 |
| CLI YAML 文件 | `--trainer-config examples/custom_trainers/custom_loss.yaml` | 推荐 — 跟其它 yaml 配置风格一致 |
| Experiment YAML | `experiments.yaml: trainer_config: {huber_delta: 1.0}` | sweep 时由 experiment 透传给 subprocess |

`trainer_config` 必须是 **JSON 可序列化** (dict / list / str / int / float / bool).
启动前 `json.dumps` 校验, 失败 `ValueError` (L15).

## 4. 多模型权重前缀 (joint trainer 专属)

`JointTrainer` 同时训主 + 辅两个模型, 写进同一个 `weights.safetensors` /
`weights_best.safetensors`. key 加前缀防冲突:

```python
state = {
    "main.lstm.weight_ih_l0": tensor,   # 主模型
    "main.head.weight": tensor,
    "aux.lstm.weight_ih_l0": tensor,    # 辅助模型
    "aux.head.weight": tensor,
}
```

读取方按前缀拆回:

```python
from safetensors.torch import load_file
state = load_file("artifacts/.../weights_best.safetensors")
main_only = {k[len("main."):]: v
             for k, v in state.items() if k.startswith("main.")}
model.load_state_dict(main_only)
```

约定来自 trainer-layer-plan §4.7 (D8). 不是 schema 校验 — 文档为主, 你的
trainer 用其它前缀也行, 只要读方知道.

## 5. 跑三个模板

### 5.1 custom_loss

```bash
uv run python -m rundown.cli train \
    --dataset etth1 --model lstm \
    --trainer examples.custom_trainers.custom_loss.CustomLossTrainer \
    --trainer-config examples/custom_trainers/custom_loss.yaml \
    --epochs-max 1
```

### 5.2 staged

```bash
uv run python -m rundown.cli train \
    --dataset etth1 --model lstm \
    --trainer examples.custom_trainers.staged.StagedTrainer \
    --epochs-max 1
```

`staged` 不需要 `--trainer-config` (用 default stages). 想自定义:

```bash
cat > /tmp/staged.yaml <<EOF
stages:
  - {epochs: 2, lr: 0.001, freeze: false}
  - {epochs: 1, lr: 0.0005, freeze: true}
EOF

uv run python -m rundown.cli train \
    --dataset etth1 --model lstm \
    --trainer examples.custom_trainers.staged.StagedTrainer \
    --trainer-config /tmp/staged.yaml \
    --epochs-max 1
```

### 5.3 joint

```bash
uv run python -m rundown.cli train \
    --dataset etth1 --model lstm \
    --trainer examples.custom_trainers.joint.JointTrainer \
    --epochs-max 1
```

### 5.4 experiment sweep

```yaml
# configs/experiments.yaml
experiments:
  custom_sweep:
    dataset: etth1
    model: lstm
    epochs_max: 1
    variants: [small]
    trainer: examples.custom_trainers.joint.JointTrainer
    trainer_config: {alpha: 0.1, aux_model: lstm, aux_variant: small}
    metric: best_val_mse
```

```bash
uv run python -m rundown.cli experiment run custom_sweep
```

---

## 6. Caveats — 上手前请读

### 6.1 import 路径 / sys.path

`python -m rundown.cli` 把 cwd 加进 `sys.path` (CLI 默认行为), 因此
`examples.custom_trainers.X` 直接 import 即可. **直接走 `rundown` console
script** (e.g. 全局装的 `rundown` 命令) 时, `examples.custom_trainers` 不在
搜索路径里 — 你需要:

1. `pip install -e .` 让项目根可 import; 或
2. `PYTHONPATH=. rundown train ...`; 或
3. 把 trainer 拷到 `src/myproj/trainers.py`, 用 `myproj.trainers.JointTrainer`
   这种 absolute import.

L11 / L17 风险已锁定, 见 trainer-layer-plan §6.

### 6.2 代码快照

snapshot 只含 **tracked** 文件 (走 `git archive` 时). trainer.py 是
源代码 — 必须提交进 git 才能被 snapshot 抓到. dirty tree 的 trainer
改动**不会**进 snapshot. L12.

### 6.3 trainer.py 不许 import rundown

模板里的 `from rundown.trainers import TrainResult` 是唯一例外 (协议返回值).
其它 `import rundown.xxx` 是反向依赖, **禁止** — 模型 / trainer 文件
不应知道 rundown 存在. CLI / registry 反射 trainer, 不反过来. L1.

---

## 7. 文件清单

```
examples/custom_trainers/
├── __init__.py
├── common.py             # seed / forward / state / evaluate / write_test_predictions
├── custom_loss.py        # CustomLossTrainer (Huber + L1 weight)
├── custom_loss.yaml      # 默认 trainer_config
├── staged.py             # StagedTrainer (SequenceLSTM-style freeze)
├── joint.py              # JointTrainer (multi-model, prefix weights)
├── README.md             # 本文件
└── (test_custom_trainers.py 在 tests/, 与 rundown 测试同跑)
```

## 8. 复制改写

三个模板都**显式可改** — 比「框架抽象」更适合学习. 步骤:

1. `cp -r examples/custom_trainers myproj/trainers`
2. 改 `custom_loss.py` 里 `_custom_loss` 公式 / 加 metric / 加曲线;
3. 注册到 `configs/trainers.yaml`:
   ```yaml
   trainers:
     my_custom:
       target: myproj.trainers.custom_loss.MyLossTrainer
   ```
4. CLI: `--trainer my_custom`.

或者直接走 dotted path, 不用 `trainers.yaml`:

```bash
--trainer myproj.trainers.custom_loss.MyLossTrainer
```
