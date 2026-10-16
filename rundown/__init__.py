"""rundown: 声明式运行 · 持久产物 · 为 Agent 而生.

MVP 实现:
  - rundown.ids:        ID 校验 (alias / model_id / run_id).
  - rundown.events:     训练事件流 (events.jsonl + stderr).
  - rundown.registry:   YAML 注册表 + 反射绑定.
  - rundown.artifact:   run 目录的写/读 (manifest + weights + curves + predictions).
  - rundown.engine:     GradientStrategy.fit() (MVP 唯一策略).
  - rundown.cli:        7 个动词 (prepare / train / experiment / read-run / list-runs / compare / analyze).
  - rundown.cli_query:  list-runs / compare 的 artifact-ground-truth 查询.
  - rundown._env:       capture_env + snapshot_code (autolog host env + code snapshot).
  - rundown.facts:      fact 系统 (main_facts.yaml + 子 fact markdown, 手动保存).

Phase 2 (尚未实现, 不在本仓库):
  - AnalyticStrategy / FrozenStrategy
  - lineage DAG + CAS 2 级散列
  - optuna / TPE 集成 (extras, 已有 stub)
  - 完整 Agent 工具集 (diff-runs / tail-events / run-analyzer)
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
