"""rundown facts system tests — 路径约定 + 模板内容.

facts 系统是「手动保存结论」, framework 不做 CLI / schema 校验.
测试只锁住「路径约定」+「模板 section」+「状态 enum」, 防止无意破坏.
"""
from __future__ import annotations
from pathlib import Path

import pytest

from rundown.facts import (
    FACTS_DIR_NAME,
    MAIN_FACTS_FILENAME,
    FactStatus,
    facts_dir,
    main_facts_path,
    sub_fact_path,
    MAIN_FACTS_SCHEMA_NOTE,
)


# ---------------------------------------------------------------------------
# 路径约定 — out_root/_facts/ 横向层, 跟 _shared/ 同级
# ---------------------------------------------------------------------------
class TestFactPaths:
    def test_facts_dir_under_out_root(self, tmp_path):
        assert facts_dir(tmp_path) == tmp_path / "_facts"

    def test_main_facts_path(self, tmp_path):
        assert main_facts_path(tmp_path) == tmp_path / "_facts" / "main_facts.yaml"

    def test_sub_fact_path(self, tmp_path):
        assert sub_fact_path(tmp_path, "fact_001") == tmp_path / "_facts" / "fact_001.md"
        assert sub_fact_path(tmp_path, "method_v2_etth1") == \
               tmp_path / "_facts" / "method_v2_etth1.md"

    def test_facts_dir_does_not_create(self, tmp_path):
        """helper 不创建目录 — caller 自己决定 mkdir."""
        assert not facts_dir(tmp_path).exists()


# ---------------------------------------------------------------------------
# FactStatus — 4 个状态, 字符串 enum
# ---------------------------------------------------------------------------
class TestFactStatus:
    def test_status_values(self):
        assert FactStatus.CONFIRMED.value == "confirmed"
        assert FactStatus.REFUTED.value == "refuted"
        assert FactStatus.PENDING.value == "pending"
        assert FactStatus.SUPERSEDED.value == "superseded"

    def test_status_from_string(self):
        assert FactStatus("confirmed") is FactStatus.CONFIRMED
        assert FactStatus("refuted") is FactStatus.REFUTED

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            FactStatus("unknown_state")


# ---------------------------------------------------------------------------
# 模板内容 — 防止无意破坏关键 section
# ---------------------------------------------------------------------------
class TestFactTemplates:
    def test_sub_fact_template_exists(self):
        p = Path(__file__).parent.parent / "rundown" / "facts" / "_sub_fact_template.md"
        assert p.exists(), f"sub_fact_template.md missing: {p}"

    def test_sub_fact_template_has_required_sections(self):
        """子 fact 模板必须含: status / 实验设置 / 相关 run / 结论 / 证据 artifacts.

        这些 section 是 agent/人填表的骨架, 任何丢失都让模板失效.
        """
        p = Path(__file__).parent.parent / "rundown" / "facts" / "_sub_fact_template.md"
        text = p.read_text(encoding="utf-8")
        for required_section in [
            "**status**",
            "## 实验设置",
            "## 相关 run",
            "## 结论",
            "## 证据 artifacts",
            "manifest.yaml",  # 必填相关 run 链接
            "predictions",     # 必填证据
        ]:
            assert required_section in text, (
                f"sub_fact_template.md 缺 section: {required_section!r}"
            )

    def test_sub_fact_template_placeholder(self):
        """模板含 [FACT_ID] 占位符 (明示替换位置)."""
        p = Path(__file__).parent.parent / "rundown" / "facts" / "_sub_fact_template.md"
        text = p.read_text(encoding="utf-8")
        assert "[FACT_ID]" in text

    def test_main_facts_template_exists(self):
        p = Path(__file__).parent.parent / "rundown" / "facts" / "_main_facts_template.yaml"
        assert p.exists()

    def test_main_facts_template_has_facts_key(self):
        """主 fact YAML 模板根 key 是 facts: (YAML 列表锚点)."""
        p = Path(__file__).parent.parent / "rundown" / "facts" / "_main_facts_template.yaml"
        text = p.read_text(encoding="utf-8")
        assert "facts:" in text
        # 模板 entry 含必需字段
        for required_field in ["id:", "title:", "status:", "created_at:", "updated_at:",
                               "tags:", "path:"]:
            assert required_field in text, (
                f"main_facts_template.yaml 缺字段: {required_field!r}"
            )


# ---------------------------------------------------------------------------
# 哲学边界 — facts 系统明确不做的事
# ---------------------------------------------------------------------------
class TestFactsPhilosophyBoundaries:
    """防止「fact 系统自动捕获」悄悄滑入 — 撞「拒绝 3: 不做隐式回落」."""

    def test_no_facts_module_in_cli(self):
        """cli.py 不引用 rundown.facts (framework 不做 CLI)."""
        from rundown import cli as cli_module
        cli_src = Path(cli_module.__file__).read_text(encoding="utf-8")
        assert "rundown.facts" not in cli_src, (
            "cli.py 不该引用 facts — framework 不做 fact CLI (用户手动写)"
        )

    def test_no_automatic_fact_write_in_engine(self):
        """engine.py 不写 fact 文件 (不撞「拒绝 3: 不做隐式回落」)."""
        from rundown import engine as engine_module
        engine_src = Path(engine_module.__file__).read_text(encoding="utf-8")
        # engine 不该 import facts (也不该自动写)
        assert "rundown.facts" not in engine_src
        assert "_facts" not in engine_src

    def test_main_facts_schema_documents_status_enum(self):
        """MAIN_FACTS_SCHEMA_NOTE 描述 4 个状态 (跟 FactStatus 一致)."""
        for status in ["confirmed", "pending", "refuted", "superseded"]:
            assert status in MAIN_FACTS_SCHEMA_NOTE
