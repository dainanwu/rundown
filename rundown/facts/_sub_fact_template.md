# [FACT_ID]: <一句话结论标题>

**status**: pending
**created_at**: YYYY-MM-DD
**updated_at**: YYYY-MM-DD
**tags**: []

> 状态可选: `confirmed` (证实) / `pending` (待验证) / `refuted` (证伪) /
> `superseded` (被超越). 结论可被推翻可被超越, 改 status 字段 + updated_at.

## 实验设置

- dataset: `<alias>`  (splits=...)
- model: `<alias>`  (variant=..., key kwargs=...)
- epochs_max: ..., lr: ..., seed: ...

## 相关 run (链接到 manifest.yaml)

- `[<run_id>](../<run_id>/manifest.yaml)`  ← 必填

## 结论 (3-5 句话, 数字 / 改进百分比 / 失败原因)

<一句话不够. 至少写明: 测的是什么, 跟谁比, 量化结果, 是否多次复现.>

## 证据 artifacts (相对路径)

- weights: `<run_id>/weights_best.safetensors`
- predictions: `<run_id>/predictions/test_best.safetensors`
- events: `<run_id>/events.jsonl`
- code snapshot: `[code_<sha8>.zip](../_shared/code_<sha8>.zip)`

## 备注 / 局限 / 待验证假设

<optional. 这条结论的边界条件 / 后续问题 / 复现要求 (GPU vs CPU / seed 敏感度).>
