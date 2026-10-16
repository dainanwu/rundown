"""rundown facts system — 保存结论与判断的事实库.

跟 artifact 自描述哲学一致: facts 是「判断」层, manifest 是「事实」层.
manifest 记「这次 run 跑了什么」; facts 记「这次 run **意义**是什么」.

设计原则 (跟 rundown 哲学对齐):
  * **手动保存** — framework 不自动捕获「这次实验意义」(不撞「拒绝 3:
    不做隐式回落」). agent/人根据实验结论自己决定写什么 fact.
  * **路径约定** — facts 统一在 out_root/_facts/ (跟 _shared/ 横向同级).
  * **两层结构** — main_facts.yaml (索引) + [FACT_ID].md (详情).
  * **状态字段** — confirmed / refuted / pending / superseded, 让事实库有
    生命周期, 结论可被推翻可被超越.

使用方式:
  1. agent/人 cp ``artifacts/_facts/_sub_fact_template.md fact_xxx.md``
  2. 按模板填 (时间 / 设置 / 结论 / 相关 run 链接 / 证据 artifacts)
  3. 把 entry 加进 ``artifacts/_facts/main_facts.yaml`` 的 facts: 列表

framework 不提供 CLI, 不自动写, 不校验 schema — 文本文件由 user/agent
直接 ``read_file`` / ``write_file`` 操作.
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Final


# 路径约定 — out_root/_facts/ 横向层, 跟 _shared/ 同级 (snapshot 池).
FACTS_DIR_NAME: Final[str] = "_facts"
MAIN_FACTS_FILENAME: Final[str] = "main_facts.yaml"


class FactStatus(str, Enum):
    """fact 生命周期状态.

    confirmed  — 证实 (有多次 run 复现 / 多个证据支持)
    refuted    — 证伪 (新 run 推翻了先前结论)
    pending    — 待验证 (单次实验, 还没复现)
    superseded — 被新 fact 超越 (结论没错, 但被更好的方法盖过)
    """
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    PENDING = "pending"
    SUPERSEDED = "superseded"


def facts_dir(out_root: Path) -> Path:
    """out_root/_facts/ 路径. caller 决定 mkdir 还是 no-op."""
    return Path(out_root) / FACTS_DIR_NAME


def main_facts_path(out_root: Path) -> Path:
    """out_root/_facts/main_facts.yaml 路径 (索引文件)."""
    return facts_dir(out_root) / MAIN_FACTS_FILENAME


def sub_fact_path(out_root: Path, fact_id: str) -> Path:
    """out_root/_facts/<fact_id>.md 路径 (子 fact 详情)."""
    return facts_dir(out_root) / f"{fact_id}.md"


# 主 fact YAML schema 注释 — agent/人 cp 后改.
# (实际模板放 _main_facts_template.yaml, 编辑器友好.)
MAIN_FACTS_SCHEMA_NOTE: Final[str] = """\
main_facts.yaml schema:
  facts:
    - id: fact_001                    # 唯一 ID (用作文件名)
      title: 一句话结论标题            # agent/人扫一眼能懂
      status: confirmed|pending|refuted|superseded
      created_at: YYYY-MM-DD
      updated_at: YYYY-MM-DD
      tags: [tag1, tag2]               # 自由分类
      path: fact_001.md                # 相对 out_root/_facts/
"""


__all__ = [
    "FACTS_DIR_NAME",
    "MAIN_FACTS_FILENAME",
    "FactStatus",
    "facts_dir",
    "main_facts_path",
    "sub_fact_path",
    "MAIN_FACTS_SCHEMA_NOTE",
]
