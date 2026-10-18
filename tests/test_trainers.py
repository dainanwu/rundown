"""C2 — Trainer registry / resolve_trainer / --trainer / --trainer-config 集成测试.

覆盖:
    1. trainer_alias regex 校验 (valid / invalid).
    2. resolve_trainer("gradient") -> GradientTrainer, canonical "gradient".
    3. resolve_trainer("tests.fixtures.DummyTrainer") -> instance + canonical dotted path.
    4. Unknown alias -> ValueError (含可读信息).
    5. configs/trainers.yaml:
        * alias 解析 (importlib + getattr).
        * alias "gradient" shadow built-in -> ValueError (L16).
        * bad target (无 dot / 非 str) -> ValueError.
    6. ModelEntry.default_trainer 加载 + 校验 (alias / dotted / invalid).
    7. CLI custom trainer 端到端 (dotted path + --trainer-config JSON).
    8. CLI unknown trainer -> rc 2 + stderr 提到 name / available.
    9. CLI 默认 --trainer 不传 -> 内置 gradient (resolve_trainer 行为单测 +
       小 epochs 跑一次端到端).

所有测试用 try/finally + set_quiet(False) / clear_active_log() 收尾, 不污染
模块全局 (event / quiet).
"""
from __future__ import annotations
import json
import os
import textwrap
from pathlib import Path
from typing import Dict

import pytest
import yaml

from rundown.ids import assert_valid
from rundown.trainers import (
    GradientTrainer,
    TrainResult,
    load_trainer_aliases,
    resolve_trainer,
)
from rundown.events import clear_active_log, set_quiet
from tests.fixtures import DummyTrainer


# ---------------------------------------------------------------------------
# 1. trainer_alias regex
# ---------------------------------------------------------------------------

class TestTrainerAliasRegex:
    """trainer_alias 走与 dataset_alias / model_id 同款正则."""

    def test_valid_aliases(self):
        for good in ["gradient", "joint", "joint_trainer", "a_b_c_2", "abc",
                     "a" + "b" * 40]:
            assert_valid("trainer_alias", good)

    def test_invalid_aliases(self):
        for bad in ["", "1bad", "Has-Cap", "has space", "x" * 50, "ab",
                     "a" * 42, "abc.def"]:
            with pytest.raises(ValueError):
                assert_valid("trainer_alias", bad)


# ---------------------------------------------------------------------------
# 2. resolve_trainer("gradient") -> GradientTrainer
# ---------------------------------------------------------------------------

class TestResolveBuiltin:
    def test_gradient_returns_GradientTrainer(self):
        set_quiet(False)
        try:
            t, canonical = resolve_trainer("gradient")
            assert isinstance(t, GradientTrainer)
            assert canonical == "gradient"
            # 协议字段
            assert t.name == "gradient"
            assert t.required_provider_keys == ("x", "y", "warm_start")
            assert t.primary_metric == "best_val_mse"
            # validate 是 no-op
            t.validate({"train": lambda s: None, "val": lambda s: None, "test": lambda s: None})
        finally:
            clear_active_log()

    def test_gradient_validate_does_not_consume_batch(self, monkeypatch):
        """L8-衍生: GradientTrainer.validate() 不能消费 batch — 默认 CLI
        路径的 RNG 必须与 C1 一致. 用 spy 监测 provider 是否被调用.
        """
        set_quiet(False)
        try:
            called = {"n": 0}
            def spy(split):
                called["n"] += 1
                return {"x": None, "y": None, "warm_start": 0}
            providers = {"train": spy, "val": spy, "test": spy}
            t, _ = resolve_trainer("gradient")
            t.validate(providers)
            assert called["n"] == 0, f"validate() must be no-op, called {called['n']} times"
        finally:
            clear_active_log()


# ---------------------------------------------------------------------------
# 3. resolve_trainer(dotted path) -> instance + canonical dotted path
# ---------------------------------------------------------------------------

class TestResolveDottedPath:
    def test_dotted_path_to_dummy_trainer(self):
        """``tests.fixtures.DummyTrainer`` 路径 → DummyTrainer 实例, canonical = 路径."""
        set_quiet(False)
        try:
            t, canonical = resolve_trainer("tests.fixtures.DummyTrainer")
            assert isinstance(t, DummyTrainer)
            assert canonical == "tests.fixtures.DummyTrainer"
            # 协议字段对齐 (与 GradientTrainer 一致字段集, 便于 test 默认 manifest 字段)
            assert t.name == "dummy"
            assert t.required_provider_keys == ("x", "y", "warm_start")
            assert t.primary_metric == "best_val_mse"
        finally:
            clear_active_log()

    def test_dotted_path_unimportable(self):
        with pytest.raises(ValueError, match="cannot import"):
            resolve_trainer("nonexistent.module.NopeTrainer")

    def test_dotted_path_no_module_attr(self):
        # 模块存在, 但没那个 attr
        with pytest.raises(ValueError, match="no attribute"):
            resolve_trainer("tests.fixtures.NotATrainerClass")

    def test_dotted_path_missing_protocol_attrs(self):
        """类存在但缺 Trainer 协议字段 → ValueError (不是训练中途 AttributeError)."""
        with pytest.raises(ValueError, match="missing required Trainer attribute"):
            resolve_trainer("tests.fixtures.IncompleteTrainer")

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            resolve_trainer("")


# ---------------------------------------------------------------------------
# 4. Unknown alias -> ValueError
# ---------------------------------------------------------------------------

class TestUnknownAlias:
    def test_unknown_alias_raises_with_available_list(self, tmp_path, monkeypatch):
        """alias 没在 trainers.yaml 注册 -> ValueError 含可读信息.

        把 cwd 切到 tmp_path (空 configs/), 避免污染仓库自带 config.
        """
        monkeypatch.chdir(tmp_path)
        # tmp_path 没有任何 trainers.yaml → load_trainer_aliases -> {}
        with pytest.raises(ValueError) as exc_info:
            resolve_trainer("nonexistent_alias_xyz")
        msg = str(exc_info.value)
        assert "nonexistent_alias_xyz" in msg
        # 至少提到 "gradient" (built-in 永远在 available)
        assert "gradient" in msg


# ---------------------------------------------------------------------------
# 5. configs/trainers.yaml
# ---------------------------------------------------------------------------

class TestTrainersYaml:
    def _write_yaml(self, tmp_path: Path, body: str) -> Path:
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "trainers.yaml").write_text(body, encoding="utf-8")
        return cfg

    def test_yaml_alias_resolution(self, tmp_path):
        """trainers.yaml 注册的 alias 走 importlib + getattr, no-arg 实例化."""
        # 写一个会 redirect 到 DummyTrainer 的 alias.
        self._write_yaml(tmp_path, textwrap.dedent("""\
            trainers:
              my_dummy:
                target: tests.fixtures.DummyTrainer
        """))
        t, canonical = resolve_trainer("my_dummy", configs_dir=tmp_path / "configs")
        assert isinstance(t, DummyTrainer)
        assert canonical == "my_dummy"

    def test_yaml_alias_shadow_gradient_rejected(self, tmp_path):
        """alias "gradient" 试图覆盖 built-in -> ValueError (L16)."""
        self._write_yaml(tmp_path, textwrap.dedent("""\
            trainers:
              gradient:
                target: tests.fixtures.DummyTrainer
        """))
        with pytest.raises(ValueError, match="shadows built-in"):
            load_trainer_aliases(configs_dir=tmp_path / "configs")

    def test_yaml_alias_invalid_alias_name_rejected(self, tmp_path):
        """alias 不满足 trainer_alias 正则 -> ValueError."""
        self._write_yaml(tmp_path, textwrap.dedent("""\
            trainers:
              "Has-Cap":
                target: tests.fixtures.DummyTrainer
        """))
        with pytest.raises(ValueError, match="invalid trainer_alias"):
            load_trainer_aliases(configs_dir=tmp_path / "configs")

    def test_yaml_alias_bad_target_rejected(self, tmp_path):
        """target 不是 dotted path (无 dot) -> ValueError."""
        self._write_yaml(tmp_path, textwrap.dedent("""\
            trainers:
              my_alias:
                target: NotADottedPath
        """))
        with pytest.raises(ValueError, match="dotted path"):
            load_trainer_aliases(configs_dir=tmp_path / "configs")

    def test_yaml_alias_missing_target_rejected(self, tmp_path):
        self._write_yaml(tmp_path, textwrap.dedent("""\
            trainers:
              my_alias:
                # no target field
                foo: bar
        """))
        with pytest.raises(ValueError, match="dotted path"):
            load_trainer_aliases(configs_dir=tmp_path / "configs")

    def test_load_trainer_aliases_missing_file_returns_empty(self, tmp_path):
        """configs/trainers.yaml 缺 -> {} (不抛错)."""
        cfg = tmp_path / "configs"  # mkdir but no file
        cfg.mkdir()
        assert load_trainer_aliases(configs_dir=cfg) == {}

    def test_load_trainer_aliases_no_trainers_key_returns_empty(self, tmp_path):
        self._write_yaml(tmp_path, "other_key: 1\n")
        assert load_trainer_aliases(configs_dir=tmp_path / "configs") == {}

    def test_malformed_yaml_raises_instead_of_silent_empty(self, tmp_path):
        """损坏的 trainers.yaml 不能静默当空配置 — 否则 alias 会神秘消失."""
        self._write_yaml(tmp_path, "trainers: [oops\n")
        with pytest.raises(ValueError, match="invalid YAML"):
            load_trainer_aliases(configs_dir=tmp_path / "configs")

    def test_non_mapping_top_level_raises(self, tmp_path):
        self._write_yaml(tmp_path, "- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="top-level must be a mapping"):
            load_trainer_aliases(configs_dir=tmp_path / "configs")


# ---------------------------------------------------------------------------
# 6. ModelEntry.default_trainer 加载 + 校验
# ---------------------------------------------------------------------------

class TestModelEntryDefaultTrainer:
    def test_load_models_default_trainer_alias(self, tmp_path):
        from rundown.registry import load_models
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "models.yaml").write_text(textwrap.dedent("""\
            models:
              lstm:
                factory:
                  target: tests.fixtures.ToyNet
                  kwargs: {in_dim: 7, hidden: 16}
                default_trainer: gradient
        """), encoding="utf-8")
        models = load_models(cfg)
        assert models["lstm"].default_trainer == "gradient"

    def test_load_models_default_trainer_dotted(self, tmp_path):
        from rundown.registry import load_models
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "models.yaml").write_text(textwrap.dedent("""\
            models:
              lstm:
                factory:
                  target: tests.fixtures.ToyNet
                  kwargs: {in_dim: 7, hidden: 16}
                default_trainer: tests.fixtures.DummyTrainer
        """), encoding="utf-8")
        models = load_models(cfg)
        assert models["lstm"].default_trainer == "tests.fixtures.DummyTrainer"

    def test_load_models_default_trainer_none_default(self, tmp_path):
        from rundown.registry import load_models
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "models.yaml").write_text(textwrap.dedent("""\
            models:
              lstm:
                factory:
                  target: tests.fixtures.ToyNet
                  kwargs: {in_dim: 7, hidden: 16}
        """), encoding="utf-8")
        models = load_models(cfg)
        # 不写 default_trainer → None
        assert models["lstm"].default_trainer is None

    def test_model_entry_default_trainer_invalid_alias_rejected(self):
        from rundown.registry import ModelEntry
        with pytest.raises(ValueError, match="invalid trainer_alias"):
            ModelEntry(
                alias="bad", factory_target="tests.fixtures.ToyNet",
                factory_kwargs={}, default_trainer="Has-Cap",
            )

    def test_model_entry_default_trainer_invalid_dotted_rejected(self):
        from rundown.registry import ModelEntry
        # "NotADottedPath" 是 alias 形式但通不过 alias 正则
        with pytest.raises(ValueError, match="invalid trainer_alias"):
            ModelEntry(
                alias="bad", factory_target="tests.fixtures.ToyNet",
                factory_kwargs={}, default_trainer="NotADottedPath",
            )
        # "module." 不算 valid dotted (attr 为空)
        with pytest.raises(ValueError, match="dotted path"):
            ModelEntry(
                alias="bad2", factory_target="tests.fixtures.ToyNet",
                factory_kwargs={}, default_trainer="mod.",
            )
        # ".attr" (module 为空)
        with pytest.raises(ValueError, match="dotted path"):
            ModelEntry(
                alias="bad3", factory_target="tests.fixtures.ToyNet",
                factory_kwargs={}, default_trainer=".attr",
            )


# ---------------------------------------------------------------------------
# 7. CLI custom trainer end-to-end (dotted path + --trainer-config JSON)
# ---------------------------------------------------------------------------

class TestCliCustomTrainer:
    """CLI ``rundown train`` 接 ``--trainer tests.fixtures.DummyTrainer`` +
    ``--trainer-config '{"alpha": 0.5}'`` → 跑通, manifest 字段齐."""

    def _setup_configs(self, tmp_path: Path) -> Path:
        """写 configs/{datasets,models}.yaml 让 CLI 可执行. 用
        ``tests.fixtures.build_random_providers`` 作为 provider_factory
        (RandomProvider 的 registry builder, 签名跟 build_etth1_providers
        对齐). monkeypatch.chdir 之前 sys.path 里有 tests/ (默认 pytest 加了).
        """
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "datasets.yaml").write_text(textwrap.dedent("""\
            datasets:
              etth1:
                source: {kind: csv, path: examples/baseline/data/ETTh1.csv}
                target_col: OT
                provider_factory: tests.fixtures.build_random_providers
                splits: {train: 0.7, val: 0.1, test: 0.2}
        """), encoding="utf-8")
        (cfg / "models.yaml").write_text(textwrap.dedent("""\
            models:
              lstm:
                factory:
                  target: tests.fixtures.ToyNet
                  kwargs: {in_dim: 7, hidden: 16}
        """), encoding="utf-8")
        return cfg

    def test_cli_dotted_trainer_with_json_config(self, tmp_path, monkeypatch, capsys):
        """rundown train --trainer tests.fixtures.DummyTrainer
        --trainer-config '{"alpha": 0.5}' → rc 0, manifest 字段齐."""
        cfg = self._setup_configs(tmp_path)
        # RandomProvider 需要 in_dim; 透传 --kwargs in_dim=7.
        # 但 ToyNet 的 factory.kwargs in_dim=7 已经设了; RandomProvider
        # 用 kwargs["in_dim"] 构造. 这里让 provider 接受 kwargs=7 也 OK.

        monkeypatch.chdir(tmp_path)
        set_quiet(False)
        try:
            from rundown.cli import main
            rc = main([
                "train", "--dataset", "etth1", "--model", "lstm",
                "--trainer", "tests.fixtures.DummyTrainer",
                "--trainer-config", '{"alpha": 0.5}',
                "--epochs-max", "1",
                "--out-dir", str(tmp_path),
                # RandomProvider 接受 in_dim kwargs, ToyNet 已经传给 factory
                # 时 in_dim=7; provider 也用 in_dim=7. 不需要 --kwargs.
            ])
        finally:
            clear_active_log()
            set_quiet(False)

        assert rc == 0, f"CLI 失败: rc={rc}, stderr={capsys.readouterr().err}"
        # 找生成的 run_dir: tmp_path / "lstm__etth1__seed0" (没 --out-dir 时
        # 默认 artifacts/, 这里 --out-dir = tmp_path)
        run_dir = tmp_path / "lstm__etth1__seed0"
        assert (run_dir / "manifest.yaml").exists(), f"manifest 不在 {run_dir}"

        m = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        cfg_block = m["config"]
        assert cfg_block["trainer"] == "tests.fixtures.DummyTrainer"
        assert cfg_block["trainer_config"] == {"alpha": 0.5}
        assert cfg_block["trainer_contract"] == ["x", "y", "warm_start"]
        assert cfg_block["primary_metric"] == "best_val_mse"
        # products written
        assert (run_dir / "weights.safetensors").exists()
        assert (run_dir / "weights_best.safetensors").exists()
        assert (run_dir / "curves.safetensors").exists()
        assert (run_dir / "predictions" / "test_best.safetensors").exists()
        assert (run_dir / "predictions" / "test.safetensors").exists()


# ---------------------------------------------------------------------------
# 8. Unknown trainer via CLI -> rc 2 + stderr 提到 name / available
# ---------------------------------------------------------------------------

class TestCliUnknownTrainer:
    def test_unknown_trainer_alias(self, tmp_path, monkeypatch, capsys):
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "datasets.yaml").write_text(textwrap.dedent("""\
            datasets:
              etth1:
                source: {kind: csv, path: examples/baseline/data/ETTh1.csv}
                target_col: OT
                provider_factory: tests.fixtures.build_random_providers
                splits: {train: 0.7, val: 0.1, test: 0.2}
        """), encoding="utf-8")
        (cfg / "models.yaml").write_text(textwrap.dedent("""\
            models:
              lstm:
                factory:
                  target: tests.fixtures.ToyNet
                  kwargs: {in_dim: 7, hidden: 16}
        """), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        set_quiet(False)
        try:
            from rundown.cli import main
            rc = main([
                "train", "--dataset", "etth1", "--model", "lstm",
                "--trainer", "no_such_trainer_alias_xyz",
            ])
        finally:
            clear_active_log()
            set_quiet(False)
        assert rc == 2
        err = capsys.readouterr().err
        assert "no_such_trainer_alias_xyz" in err
        # 至少提到一个可用的 trainer (gradient)
        assert "gradient" in err

    def test_bad_trainer_dotted_path(self, tmp_path, monkeypatch, capsys):
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "datasets.yaml").write_text(textwrap.dedent("""\
            datasets:
              etth1:
                source: {kind: csv, path: examples/baseline/data/ETTh1.csv}
                target_col: OT
                provider_factory: tests.fixtures.build_random_providers
                splits: {train: 0.7, val: 0.1, test: 0.2}
        """), encoding="utf-8")
        (cfg / "models.yaml").write_text(textwrap.dedent("""\
            models:
              lstm:
                factory:
                  target: tests.fixtures.ToyNet
                  kwargs: {in_dim: 7, hidden: 16}
        """), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        set_quiet(False)
        try:
            from rundown.cli import main
            rc = main([
                "train", "--dataset", "etth1", "--model", "lstm",
                "--trainer", "nonexistent.module.NopeTrainer",
            ])
        finally:
            clear_active_log()
            set_quiet(False)
        assert rc == 2
        err = capsys.readouterr().err
        assert "nonexistent.module.NopeTrainer" in err
        assert "cannot import" in err


# ---------------------------------------------------------------------------
# 9. 默认 --trainer 不传 -> 内置 gradient
# ---------------------------------------------------------------------------

class TestDefaultTrainer:
    def test_resolve_gradient_when_no_arg(self):
        """不传 trainer → 模型层 default_trainer=None → CLI fallback 'gradient'.

        这部分单测: 在 trainer 选择之前, ``model_entry.default_trainer or 'gradient'``
        应该走 "gradient". 这里直接测 ``resolve_trainer('gradient')`` 行为
        (CLI 端到端会跑出 manifest, 单独测在下面 test_cli_default_trainer).
        """
        set_quiet(False)
        try:
            t, canonical = resolve_trainer("gradient")
            assert isinstance(t, GradientTrainer)
            assert canonical == "gradient"
        finally:
            clear_active_log()

    def test_cli_default_trainer_runs(self, tmp_path, monkeypatch, capsys):
        """CLI 不传 --trainer → 用内置 gradient. 小 epochs 跑一次,
        验证 manifest.config.trainer == 'gradient' (C1 行为保持)."""
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "datasets.yaml").write_text(textwrap.dedent("""\
            datasets:
              etth1:
                source: {kind: csv, path: examples/baseline/data/ETTh1.csv}
                target_col: OT
                provider_factory: tests.fixtures.build_random_providers
                splits: {train: 0.7, val: 0.1, test: 0.2}
        """), encoding="utf-8")
        (cfg / "models.yaml").write_text(textwrap.dedent("""\
            models:
              lstm:
                factory:
                  target: tests.fixtures.ToyNet
                  kwargs: {in_dim: 7, hidden: 16}
        """), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        set_quiet(False)
        try:
            from rundown.cli import main
            rc = main([
                "train", "--dataset", "etth1", "--model", "lstm",
                "--epochs-max", "1",
                "--out-dir", str(tmp_path),
            ])
        finally:
            clear_active_log()
            set_quiet(False)
        assert rc == 0, f"CLI 失败: rc={rc}, stderr={capsys.readouterr().err}"
        run_dir = tmp_path / "lstm__etth1__seed0"
        m = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        # 默认 trainer 必须是 "gradient"
        assert m["config"]["trainer"] == "gradient"
        assert m["config"]["primary_metric"] == "best_val_mse"
        assert m["config"]["trainer_contract"] == ["x", "y", "warm_start"]
        # trainer_config 默认空 dict
        assert m["config"]["trainer_config"] == {}


# ---------------------------------------------------------------------------
# 附: TrainResult 形状契约
# ---------------------------------------------------------------------------

class TestTrainResultShape:
    def test_dataclass_defaults(self):
        r = TrainResult(metrics={"a": 1})
        assert r.metrics == {"a": 1}
        assert r.predictions_primary == "test_best"
        assert r.epochs_actual is None
        assert r.early_stopped is False

    def test_dataclass_full(self):
        r = TrainResult(
            metrics={"best_val_mse": 0.5},
            predictions_primary="test",
            epochs_actual=10,
            early_stopped=True,
        )
        assert r.predictions_primary == "test"
        assert r.epochs_actual == 10
        assert r.early_stopped is True
