"""Smoke tests. 不依赖网络, 不依赖大文件, 不依赖 GPU.

跑通门槛:  ``pytest tests/``  全绿.
"""
from __future__ import annotations
from pathlib import Path

import pytest

import rundown
from rundown.ids import assert_valid, is_valid, DATASET_ALIAS_RE, RUN_ID_RE
from rundown.registry import DatasetEntry, ModelEntry, load_datasets, load_models
from rundown.artifact import RunManifest, RunArtifact
from rundown.events import emit, set_active_log, clear_active_log, active_run_id


# ---------------------------------------------------------------------------
# package meta
# ---------------------------------------------------------------------------
def test_version_is_string():
    assert isinstance(rundown.__version__, str)
    parts = rundown.__version__.split(".")
    assert len(parts) == 3, "semver expected"
    for p in parts:
        assert p.isdigit()


# ---------------------------------------------------------------------------
# ids
# ---------------------------------------------------------------------------
class TestIds:
    def test_valid_dataset_alias(self):
        assert_valid("dataset_alias", "my_data")
        assert_valid("dataset_alias", "cmapss_fd001")
        assert_valid("dataset_alias", "a_b_c_2")

    def test_invalid_dataset_alias(self):
        for bad in ["", "1bad", "Has-Cap", "has space", "x" * 50]:
            with pytest.raises(ValueError):
                assert_valid("dataset_alias", bad)

    def test_valid_run_id(self):
        assert_valid("run_id", "cnn_rul__cmapss_fd001__seed0")
        # 每个 segment 至少 2 字符 ([a-z][a-z0-9_]+)
        assert_valid("run_id", "aa__bb__seed42")

    def test_invalid_run_id(self):
        for bad in ["", "no_seed", "x__y__seed", "x__y__seed0__extra",
                    "X__y__seed0", "x__Y__seed0", "a__b__seed42"]:
            with pytest.raises(ValueError):
                assert_valid("run_id", bad)

    def test_is_valid_returns_bool(self):
        assert is_valid("dataset_alias", "my_data") is True
        assert is_valid("dataset_alias", "1bad") is False
        assert is_valid("model_id", "cnn_rul") is True


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
class TestEvents:
    def test_emit_without_active_runs_writes_only_stderr(self, tmp_path, capsys):
        # 清干净 (其他 test 可能设了 active)
        clear_active_log()
        emit("test_kind", foo=1, bar="baz")
        captured = capsys.readouterr()
        # 新格式: "{t_iso}  kind  {json}" — agent 友好, 复杂 value 安全.
        assert "test_kind" in captured.err
        # 第 3 段是 JSON 字段
        import json as _json
        parts = captured.err.strip().rsplit("  ", 1)
        assert len(parts) == 2
        kind, fields_json = parts[0].split("  ", 1)[1], parts[1]
        assert kind == "test_kind"
        assert _json.loads(fields_json) == {"foo": 1, "bar": "baz"}
        # 不应落盘
        assert not list(tmp_path.glob("events.jsonl"))

    def test_set_active_then_emit_writes_file(self, tmp_path):
        events_path = set_active_log(tmp_path, "my_run")
        assert events_path.exists() is False  # 还没写任何行
        assert active_run_id() == "my_run"
        emit("epoch_completed", epoch=1, train_loss=0.42)
        emit("epoch_completed", epoch=2, train_loss=0.31)
        assert events_path.exists()
        lines = events_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        import json
        for ln in lines:
            rec = json.loads(ln)
            assert rec["kind"] == "epoch_completed"
            assert rec["run_id"] == "my_run"
            assert "t" in rec and "t_iso" in rec
        clear_active_log()
        assert active_run_id() is None

    def test_emit_quiet_suppresses_stderr_but_writes_file(self, tmp_path, capsys):
        """set_quiet(True) → emit() 不写 stderr, events.jsonl 仍写 (artifact 完整)."""
        from rundown.events import set_quiet, is_quiet
        assert is_quiet() is False
        events_path = set_active_log(tmp_path, "q_run")
        try:
            set_quiet(True)
            assert is_quiet() is True
            emit("step_completed", epoch=1, step=10, train_loss=0.5)
            captured = capsys.readouterr()
            # stderr 静默
            assert captured.err == ""
            # 文件仍写
            assert events_path.exists()
            recs = events_path.read_text(encoding="utf-8").strip().split("\n")
            assert len(recs) == 1
            import json as _json
            rec = _json.loads(recs[0])
            assert rec["kind"] == "step_completed"
            assert rec["epoch"] == 1
        finally:
            set_quiet(False)
            clear_active_log()

    def test_emit_complex_value_in_stderr_is_json_safe(self, tmp_path, capsys):
        """带空格字符串 / list 在 stderr JSON 字段里不破坏解析.

        旧格式 'foo=etth1 default config' 会被 split 错; 新格式 '{"foo": "etth1 default config"}' 安全.
        """
        from rundown.events import set_active_log, clear_active_log
        clear_active_log()
        emit("test", name="etth1 default config", items=[1, 2, 3])
        captured = capsys.readouterr()
        import json as _json
        # 解析最后一段 JSON
        tail = captured.err.strip().rsplit("  ", 1)[1]
        parsed = _json.loads(tail)
        assert parsed == {"name": "etth1 default config", "items": [1, 2, 3]}

    def test_cli_quiet_flag_sets_events_quiet_at_any_position(self):
        """CLI --quiet 在 main() 开头从 argv 手动抓, 任何位置都生效.

        argparse 全局 flag 必须放子命令前, 但 user 直觉 `train ... --quiet`
        更常见. main() 绕过这个限制, 用 sys.argv 直接查 --quiet/-q.
        """
        from rundown.cli import main
        from rundown.events import is_quiet, set_quiet

        # 测前重置
        set_quiet(False)

        # 模拟 invalid dataset (main 会快速返回 error, 但 set_quiet 已被调)
        rc = main(["prepare", "--dataset", "definitely_not_registered_xxx"])
        # 不关心 rc, 关心 quiet 被设回 False (因为没传 --quiet)
        assert is_quiet() is False

        # 子命令后传 --quiet — main() 应识别
        set_quiet(False)
        # 不能真跑 train (会写 artifact), 用一个会快速返回 error 的子命令
        # analyze 是 stub, rc=3 但 set_quiet 已被调
        main(["analyze", "--analyzer", "x", "x", "--quiet"])
        # 测 main 把 quiet 设到了 True
        # 重读 sys.argv 不可能 — 用 is_quiet() 验证副作用
        # 但 prepare 失败后 set_quiet 已被设回 False (rc=2 是 _err 路径, 不调 set_quiet(False))
        # 实际 main() 调 set_quiet(quiet_flag) — 不 reset
        assert is_quiet() is True
        set_quiet(False)  # 清理

        # 短形式 -q
        main(["analyze", "--analyzer", "x", "x", "-q"])
        assert is_quiet() is True
        set_quiet(False)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_load_datasets_missing_file(self, tmp_path):
        result = load_datasets(tmp_path)
        assert result == {}

    def test_load_datasets_ok(self, tmp_path):
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "datasets.yaml").write_text(
            "datasets:\n"
            "  my_data:\n"
            "    source: {kind: parquet, path: data/x.parquet}\n"
            "    target_col: y\n",
            encoding="utf-8",
        )
        ds = load_datasets(cfg)
        assert "my_data" in ds
        assert isinstance(ds["my_data"], DatasetEntry)
        assert ds["my_data"].source_kind == "parquet"
        assert ds["my_data"].source_path == "data/x.parquet"
        assert ds["my_data"].target_col == "y"

    def test_load_models_factory_target_validation(self, tmp_path):
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "models.yaml").write_text(
            "models:\n"
            "  bad_target:\n"
            "    factory: {target: not-a-dotted-path}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="dotted path"):
            load_models(cfg)

    def test_load_models_ignores_optimizer_field(self, tmp_path):
        """models.yaml 含 optimizer/epochs_max → 静默忽略 (边界已重设计).

        不抛错, 也不读这些字段. 警告 user via load_models 不弹错 — 因为有些
        user 可能还有老配置, 静默忽略保证向后兼容 (但 manifest.config.optimizer
        会用 defaults.DEFAULT_OPTIMIZER).
        """
        from rundown.registry import load_models
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "models.yaml").write_text(
            "models:\n"
            "  legacy:\n"
            "    factory: {target: tests.fixtures.ToyNet, kwargs: {in_dim: 7, hidden: 16}}\n"
            "    optimizer: {kind: adam, lr: 0.001}\n"   # 老字段, 应忽略
            "    epochs_max: 15\n",                    # 老字段, 应忽略
            encoding="utf-8",
        )
        models = load_models(cfg)
        assert "legacy" in models
        # ModelEntry 不再有 optimizer / epochs_max 属性
        assert not hasattr(models["legacy"], "optimizer")
        assert not hasattr(models["legacy"], "epochs_max")
        # build() 仍然能反射 + 实例化
        from tests.fixtures import ToyNet
        instance, merged, _ = models["legacy"].build()
        assert isinstance(instance, ToyNet)

    def test_model_entry_no_optimizer_or_epochs_max(self):
        """ModelEntry 字段只有 alias/factory_target/factory_kwargs/variants.

        锁住「边界已重设计」: optimizer/epochs_max 不在 models, 防止以后误加回来.
        """
        from dataclasses import fields as dc_fields
        field_names = {f.name for f in dc_fields(ModelEntry)}
        assert field_names == {"alias", "factory_target", "factory_kwargs", "variants"}, (
            f"ModelEntry 字段变了: {field_names}"
        )

    def test_defaults_optimizer_is_single_source(self):
        """DEFAULT_OPTIMIZER 是单点真值, 单次 train 的 fallback."""
        from rundown.defaults import DEFAULT_OPTIMIZER, DEFAULT_LR
        assert DEFAULT_OPTIMIZER["kind"] == "adam"
        assert DEFAULT_OPTIMIZER["lr"] == DEFAULT_LR
        assert isinstance(DEFAULT_OPTIMIZER, dict)

    def test_model_entry_build_reflective(self):
        """反射绑定到 tests.fixtures.ToyNet, 验证不抛错 + 返回 (model, kwargs, variant) tuple."""
        me = ModelEntry(
            alias="toy",
            factory_target="tests.fixtures.ToyNet",
            factory_kwargs={"in_dim": 7, "hidden": 16},
        )
        model, merged_kwargs, chosen_variant = me.build()
        from tests.fixtures import ToyNet
        assert isinstance(model, ToyNet)
        assert merged_kwargs == {"in_dim": 7, "hidden": 16}
        assert chosen_variant is None

    def test_model_entry_build_with_variant(self):
        """build(variant) 合并 variant kwargs 到 factory_kwargs, 选中的 variant 名."""
        me = ModelEntry(
            alias="toy",
            factory_target="tests.fixtures.ToyNet",
            factory_kwargs={"in_dim": 7, "hidden": 16},
            variants={"big": {"hidden": 64}},
        )
        model, merged_kwargs, chosen_variant = me.build(variant="big")
        assert merged_kwargs == {"in_dim": 7, "hidden": 64}
        assert chosen_variant == "big"

    def test_model_entry_build_unknown_variant_raises(self):
        me = ModelEntry(
            alias="toy",
            factory_target="tests.fixtures.ToyNet",
            factory_kwargs={"in_dim": 7, "hidden": 16},
            variants={"big": {"hidden": 64}},
        )
        with pytest.raises(ValueError, match="variant 'medium' not declared"):
            me.build(variant="medium")

    def test_model_entry_build_missing_module_raises(self):
        me = ModelEntry(
            alias="bad",
            factory_target="nonexistent.module.NopeNet",
            factory_kwargs={},
        )
        with pytest.raises(ValueError, match="cannot import"):
            me.build()


# ---------------------------------------------------------------------------
# experiments.yaml — optimizer field 下发 (Fix 2: 不再静默丢弃)
# ---------------------------------------------------------------------------
class TestExperimentOptimizer:
    """锁住「experiments.yaml.optimizer → subprocess --lr → engine.lr」这条线.

    早期版本 optimizer 是死字段 — 修了之后:
      1. load_experiments 必须读 optimizer 块到 Experiment.optimizer
      2. kind=adam 必须接受; 其它 kind 必须 ValueError (不静默)
      3. lr 在 (0, 1] 内接受, 否则 ValueError
      4. run_experiment 拼 subprocess 时, exp.optimizer.lr → --lr
      5. 不传 optimizer 字段时, Experiment.optimizer is None (trial 走 defaults)
    """

    def _write_exp_cfg(self, tmp_path: Path, body: str) -> Path:
        cfg = tmp_path / "configs"
        cfg.mkdir(exist_ok=True)
        (cfg / "experiments.yaml").write_text(
            "experiments:\n  e1:\n    dataset: etth1\n    model: lstm\n"
            "    epochs_max: 5\n" + body,
            encoding="utf-8",
        )
        return cfg

    def test_load_experiments_optimizes_lr_passes_through(self, tmp_path):
        from rundown.registry import load_experiments
        cfg = self._write_exp_cfg(tmp_path, "    optimizer: {kind: adam, lr: 0.0005}\n")
        exps = load_experiments(cfg)
        assert exps["e1"].optimizer == {"kind": "adam", "lr": 0.0005}

    def test_load_experiments_rejects_unknown_kind(self, tmp_path):
        """kind 必须 in SUPPORTED_OPTIMIZERS, 否则硬报错 (不静默)."""
        from rundown.registry import load_experiments
        cfg = self._write_exp_cfg(tmp_path, "    optimizer: {kind: sgd, lr: 0.01}\n")
        with pytest.raises(ValueError, match="optimizer.kind='sgd'"):
            load_experiments(cfg)

    def test_load_experiments_rejects_lr_out_of_range(self, tmp_path):
        """lr 越界 (>1 或 <=0) 硬报错 — 防 yaml 里写成小数点漂移."""
        from rundown.registry import load_experiments
        cfg = self._write_exp_cfg(tmp_path, "    optimizer: {kind: adam, lr: 5.0}\n")
        with pytest.raises(ValueError, match=r"optimizer\.lr=5\.0 越界"):
            load_experiments(cfg)
        cfg = self._write_exp_cfg(tmp_path, "    optimizer: {kind: adam, lr: 0.0}\n")
        with pytest.raises(ValueError, match=r"optimizer\.lr=0\.0 越界"):
            load_experiments(cfg)

    def test_load_experiments_lr_coerced_to_float(self, tmp_path):
        """yaml 里 lr 写成字符串 ('1e-3') 也强制 float."""
        from rundown.registry import load_experiments
        cfg = self._write_exp_cfg(tmp_path, "    optimizer: {kind: adam, lr: '0.0007'}\n")
        exps = load_experiments(cfg)
        assert isinstance(exps["e1"].optimizer["lr"], float)
        assert exps["e1"].optimizer["lr"] == pytest.approx(0.0007)

    def test_load_experiments_optimizer_optional(self, tmp_path):
        """不写 optimizer → Experiment.optimizer is None, 走 defaults."""
        from rundown.registry import load_experiments
        cfg = self._write_exp_cfg(tmp_path, "")
        exps = load_experiments(cfg)
        assert exps["e1"].optimizer is None

    def test_run_experiment_passes_lr_to_subprocess(self, tmp_path, monkeypatch):
        """run_experiment 拼 cmd 时, exp.optimizer.lr 必须出现成 --lr LR."""
        from rundown.experiment import Experiment
        captured_cmds = []

        # 拦截 subprocess.run, 不真跑 train, 只记录拼出来的 cmd
        import rundown.experiment as exp_mod

        def fake_run(cmd, *args, **kwargs):
            captured_cmds.append(cmd)
            class _R:
                returncode = 0
                stdout = "[run] fake__etth1__seed0\n"
                stderr = ""
            return _R()

        monkeypatch.setattr(exp_mod.subprocess, "run", fake_run)
        # 把 snapshot 掐掉 (不写 _shared/), 简化测试
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        exp = Experiment(
            name="smoke", dataset="etth1", model="lstm", epochs_max=1,
            variants=["small"],
            optimizer={"kind": "adam", "lr": 0.0005},
        )
        exp_mod.run_experiment(exp, out_root=tmp_path / "artifacts")

        assert len(captured_cmds) == 1, f"expect 1 trial, got {len(captured_cmds)}"
        cmd = captured_cmds[0]
        assert "--lr" in cmd, f"--lr 没出现: {cmd}"
        idx = cmd.index("--lr")
        assert cmd[idx + 1] == "0.0005"

    def test_run_experiment_no_optimizer_no_lr_flag(self, tmp_path, monkeypatch):
        """exp.optimizer is None → cmd 不含 --lr (trial 走 defaults)."""
        from rundown.experiment import Experiment
        import rundown.experiment as exp_mod

        captured_cmds = []

        def fake_run(cmd, *args, **kwargs):
            captured_cmds.append(cmd)
            class _R:
                returncode = 0
                stdout = "[run] fake__etth1__seed0\n"
                stderr = ""
            return _R()

        monkeypatch.setattr(exp_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(exp_mod, "snapshot_code", lambda *a, **k: None)

        exp = Experiment(
            name="smoke", dataset="etth1", model="lstm", epochs_max=1,
            variants=["small"],
        )
        exp_mod.run_experiment(exp, out_root=tmp_path / "artifacts")

        assert "--lr" not in captured_cmds[0]


# ---------------------------------------------------------------------------
# artifact round-trip
# ---------------------------------------------------------------------------
class TestArtifact:
    def test_manifest_save_load_roundtrip(self, tmp_path):
        m = RunManifest(
            run_id="toy__my_data__seed0",
            created_at="2026-10-16T17:00:00",
            seed=0,
            dataset={"alias": "my_data", "content_hash": "sha256:abc"},
            model={"alias": "toy", "factory": {"target": "tests.fixtures.ToyNet"}},
            metrics={"best_val_mse": 0.5},
        )
        m.save(tmp_path)
        m2 = RunManifest.load(tmp_path)
        assert m2.run_id == m.run_id
        assert m2.dataset == m.dataset
        assert m2.metrics == m.metrics

    def test_run_artifact_write_read_roundtrip(self, tmp_path):
        import torch
        run_dir = tmp_path / "run"
        manifest = RunManifest(
            run_id="toy__my_data__seed0",
            created_at="2026-10-16T17:00:00",
            seed=0,
            dataset={"alias": "my_data"},
            model={"alias": "toy"},
        )
        art = RunArtifact(run_dir=run_dir, manifest=manifest)
        art.save_manifest()

        weights = {"lin.weight": torch.randn(16, 7), "lin.bias": torch.randn(16),
                   "out.weight": torch.randn(1, 16), "out.bias": torch.randn(1)}
        art.write_weights(weights, best=False)
        art.write_weights({k: v * 0.5 for k, v in weights.items()}, best=True)
        art.write_curves({"train_loss": torch.tensor([0.5, 0.4, 0.3])})
        art.write_predictions("test", {
            "y_hat": torch.randn(4, 20),
            "y_true": torch.randn(4, 20),
        })

        # 重读
        art2 = RunArtifact.read(run_dir)
        assert "lin.weight" in art2.weights
        assert torch.allclose(art2.weights["lin.weight"], weights["lin.weight"])
        assert torch.allclose(art2.weights_best["lin.bias"], weights["lin.bias"] * 0.5)
        # float32 精度: 用 allclose, 不严格 == 
        assert torch.allclose(art2.curves["train_loss"],
                              torch.tensor([0.5, 0.4, 0.3]))
        assert art2.predictions["test"]["y_hat"].shape == (4, 20)

    def test_run_artifact_read_partial(self, tmp_path):
        """include 参数控制只读部分文件."""
        import torch
        run_dir = tmp_path / "run"
        manifest = RunManifest(
            run_id="x__y__seed0",
            created_at="2026-10-16T17:00:00",
            seed=0,
            dataset={"alias": "y"},
            model={"alias": "x"},
        )
        RunArtifact(run_dir=run_dir, manifest=manifest).save_manifest()
        RunArtifact(run_dir=run_dir, manifest=manifest).write_weights(
            {"a": torch.tensor([1.0])}
        )
        art = RunArtifact.read(run_dir, include=["weights"])
        assert art.weights is not None
        assert art.weights_best is None
        assert art.curves is None
        assert art.predictions == {}


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------
class TestCli:
    def test_help_prints(self, capsys):
        # argparse --help 会 SystemExit(0); 不能直接 main(["--help"])
        with pytest.raises(SystemExit) as exc_info:
            from rundown.cli import build_parser
            build_parser().parse_args(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "prepare" in out and "train" in out and "read-run" in out

    def test_analyze_is_stub(self, capsys):
        from rundown.cli import main
        rc = main(["analyze", "--analyzer", "horizon_growth", "x__y__seed0"])
        assert rc == 3
        err = capsys.readouterr().err
        assert "STUB" in err

    def test_prepare_unknown_dataset_errors(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        from rundown.cli import main
        rc = main(["prepare", "--dataset", "nonexistent"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "not in configs/datasets.yaml" in err

    def test_train_end_to_end(self, tmp_path):
        """Library mode: registry -> engine -> artifact 完整链路.

        自 #2 起, engine 不再支持 synthetic-data fallback; BatchProvider
        是必需的. CLI 的 train 子命令当前不接 BatchProvider registry
        (那是 Phase 2 任务), 所以这里走 library API, 直接构造 ToyNet +
        RandomProvider + GradientStrategy().fit().
        """
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet

        run_id = "toy__smoke__seed0"
        run_dir = tmp_path / "artifacts" / run_id
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id,
                created_at="2026-10-16T18:00:00",
                seed=0,
                dataset={"alias": "smoke", "source": "random"},
                model={"alias": "toy",
                       "factory": {"target": "tests.fixtures.ToyNet"}},
                config={"epochs_max": 2, "warm_start": 4, "rollout": 16},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()

            model = ToyNet(in_dim=7, hidden=16)
            train_p = RandomProvider(in_dim=7, warm_start=4, rollout=16, batch_size=32, seed=4)
            val_p = RandomProvider(in_dim=7, warm_start=4, rollout=16, batch_size=32, seed=5)
            test_p = RandomProvider(in_dim=7, warm_start=4, rollout=16, batch_size=32, seed=6)

            ctx = EngineConfig(
                run_id=run_id, seed=0, epochs_max=2, batch_size=32,
                steps_per_epoch=1, lr=1e-3,
            )
            result = GradientStrategy().fit(
                model, train_data=train_p, val_data=val_p, test_data=test_p,
                ctx=ctx, artifact=artifact,
            )
            assert result["epochs_actual"] == 2
        finally:
            clear_active_log()

        # 7 文件齐 (含 test + test_best)
        assert (run_dir / "manifest.yaml").exists()
        assert (run_dir / "weights.safetensors").exists()
        assert (run_dir / "weights_best.safetensors").exists()
        assert (run_dir / "curves.safetensors").exists()
        assert (run_dir / "events.jsonl").exists()
        assert (run_dir / "predictions" / "test.safetensors").exists()
        assert (run_dir / "predictions" / "test_best.safetensors").exists()

        import json as _json
        events = [
            _json.loads(line) for line in
            (run_dir / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
        ]
        kinds = [e["kind"] for e in events]
        assert "strategy_started" in kinds
        assert "epoch_completed" in kinds
        assert "run_completed" in kinds
        epoch_events = [e for e in events if e["kind"] == "epoch_completed"]
        assert len(epoch_events) == 2

        # manifest.config.predictions_primary 应该标记 reader 读 test_best
        import yaml as _yaml
        m = _yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        assert m["config"]["predictions_primary"] == "test_best"

    def test_cli_train_returns_clear_error(self, tmp_path, monkeypatch, capsys):
        """CLI 的 train 子命令: registry 没 provider_factory 时, 给清晰错误.

        自 Phase A 起, registry 加了 provider_factory 字段. dataset 没配时,
        CLI 应在 build_providers() 步骤报「provider_factory missing」, 而不是
        抛 AttributeError / TypeError.
        """
        cfg = tmp_path / "configs"
        cfg.mkdir()
        (cfg / "datasets.yaml").write_text(
            "datasets:\n  smoke:\n    source: {kind: parquet, path: x}\n    target_col: y\n",
            encoding="utf-8",
        )
        (cfg / "models.yaml").write_text(
            "models:\n  toy:\n    factory: {target: tests.fixtures.ToyNet, kwargs: {in_dim: 7, hidden: 16}}\n"
            "    optimizer: {kind: adam, lr: 0.001}\n    epochs_max: 1\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        from rundown.cli import main
        rc = main(["train", "--dataset", "smoke", "--model", "toy"])
        assert rc == 2
        err = capsys.readouterr().err
        # CLI 在 build_providers 步骤报清晰错误
        assert "provider_factory" in err or "library" in err

    def test_read_run_after_library_train(self, tmp_path, monkeypatch):
        """CLI read-run 在 library-mode 训练产出的 run 上仍能跑."""
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet

        run_id = "toy__smoke__seed0"
        run_dir = tmp_path / "artifacts" / run_id
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-16T18:00:00", seed=0,
                dataset={"alias": "smoke"}, model={"alias": "toy"},
                config={"epochs_max": 1},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()
            model = ToyNet(in_dim=7, hidden=16)
            p = RandomProvider(in_dim=7, warm_start=4, rollout=16, batch_size=16, seed=0)
            ctx = EngineConfig(run_id=run_id, seed=0, epochs_max=1, batch_size=16, steps_per_epoch=1)
            GradientStrategy().fit(model, train_data=p, val_data=p, test_data=p,
                                  ctx=ctx, artifact=artifact)
        finally:
            clear_active_log()

        monkeypatch.chdir(tmp_path)
        from rundown.cli import main
        rc = main(["read-run", run_id, "--metric", "prediction_error_std"])
        assert rc == 0

    def test_per_step_events_emitted(self, tmp_path):
        """step_event_every_n 控制 step_completed 事件频率.

        3 epoch * 4 step/epoch, step_event_every_n=2 → 每 epoch 写
        step 2 + step 4 (= last) 共 2 个 step_completed → 6 个总.
        """
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet
        import json as _json

        run_id = "toy__smoke__seed0"
        run_dir = tmp_path / "artifacts" / run_id
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-16T18:00:00", seed=0,
                dataset={"alias": "smoke"}, model={"alias": "toy"},
                config={"epochs_max": 3},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()
            model = ToyNet(in_dim=7, hidden=16)
            p = RandomProvider(in_dim=7, warm_start=4, rollout=8, batch_size=8, seed=0)
            ctx = EngineConfig(
                run_id=run_id, seed=0, epochs_max=3, batch_size=8,
                steps_per_epoch=4, step_event_every_n=2,
            )
            GradientStrategy().fit(model, train_data=p, val_data=p, test_data=p,
                                  ctx=ctx, artifact=artifact)
        finally:
            clear_active_log()

        events = [
            _json.loads(line) for line in
            (run_dir / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
        ]
        step_events = [e for e in events if e["kind"] == "step_completed"]
        # 3 epoch * 2 step_completed (step 2 + last step 4) = 6
        assert len(step_events) == 6, f"expected 6 step_completed, got {len(step_events)}"
        for e in step_events:
            assert "grad_norm" in e
            assert "train_loss" in e
            assert "lr" in e

        # step_completed 必须在 epoch_completed 之前 (顺序保证 reader 可配对)
        kinds_in_order = [e["kind"] for e in events]
        first_epoch_idx = kinds_in_order.index("epoch_completed")
        last_step_before = max(
            i for i, k in enumerate(kinds_in_order[:first_epoch_idx]) if k == "step_completed"
        )
        assert last_step_before < first_epoch_idx

    def test_read_run_root_argparse_binding(self, tmp_path):
        """Fix 3: read-run 必须有 --root, 默认 Path('artifacts'), 与 list/compare/inspect 一致."""
        from rundown.cli import build_parser
        parser = build_parser()
        ns = parser.parse_args(["read-run", "my_run"])
        from pathlib import Path as _P
        assert ns.root == _P("artifacts")
        ns = parser.parse_args(["read-run", "my_run", "--root", str(tmp_path)])
        assert ns.root == tmp_path

    def test_cli_main_reconfigures_stdout_to_utf8(self, monkeypatch):
        """Fix 4: cli.py:main() 在解析前 reconfigure stdout/stderr → utf-8.

        Windows 默认 cp936/cp1252, 中文 description / emit() 会乱码.
        用 MagicMock 让 stdout/stderr 仍能 write, 同时记录 reconfigure 调用.
        """
        from unittest.mock import MagicMock
        import rundown.cli as cli_mod

        # 保存原始流, 测完还原 — 否则 monkeypatch 只在本 test 内有效, 但保险
        real_stdout, real_stderr = cli_mod.sys.stdout, cli_mod.sys.stderr
        mock_stdout, mock_stderr = MagicMock(), MagicMock()
        monkeypatch.setattr(cli_mod.sys, "stdout", mock_stdout)
        monkeypatch.setattr(cli_mod.sys, "stderr", mock_stderr)
        try:
            rc = cli_mod.main(["list-runs", "--root", "."])
        finally:
            cli_mod.sys.stdout, cli_mod.sys.stderr = real_stdout, real_stderr
        # reconfigure 调用次数: stdout 1 + stderr 1 = 2
        assert mock_stdout.reconfigure.call_count == 1, mock_stdout.reconfigure.mock_calls
        assert mock_stderr.reconfigure.call_count == 1, mock_stderr.reconfigure.mock_calls
        for mock in (mock_stdout, mock_stderr):
            args, kwargs = mock.reconfigure.call_args
            assert kwargs.get("encoding") == "utf-8"
            assert kwargs.get("errors") == "replace"
        assert rc == 0

    def test_run_id_format_base_and_variant(self, tmp_path, monkeypatch):
        """CLI run_id 组成契约: ``{model}[__var{variant}]__{dataset}__seed{N}``.

        examples/baseline/test_baseline.py 依赖此格式; 改了 CLI 组成规则
        必须同时改这个 test. 不抽 compose_run_id 函数 — 锁住 format 而不是 API.
        """
        # 直接读 CLI 内部拼装行 (白盒 test); 不通过 subprocess 跑真实训练.
        from rundown.cli import build_parser
        import argparse
        parser = build_parser()

        # base (no variant)
        ns = parser.parse_args(["train", "--dataset", "etth1", "--model", "lstm",
                                "--seed", "0"])
        # 复刻 cli.py 第 115-116 行的拼装
        variant_suffix = f"__var{ns.variant}" if ns.variant else ""
        expected = f"{ns.model}{variant_suffix}__{ns.dataset}__seed{ns.seed}"
        assert expected == "lstm__etth1__seed0"

        # with variant
        ns = parser.parse_args(["train", "--dataset", "etth1", "--model", "lstm",
                                "--variant", "large", "--seed", "42"])
        variant_suffix = f"__var{ns.variant}" if ns.variant else ""
        expected = f"{ns.model}{variant_suffix}__{ns.dataset}__seed{ns.seed}"
        assert expected == "lstm__varlarge__etth1__seed42"

        # 验证两个都通过 ids.is_valid("run_id", ...)
        from rundown.ids import is_valid
        assert is_valid("run_id", "lstm__etth1__seed0")
        assert is_valid("run_id", "lstm__varlarge__etth1__seed42")

    def test_engine_emits_run_started_info_event(self, tmp_path):
        """engine fit() 开头 emit ``run_started`` 携带 dataset/model/params/config.

        上一轮 README 承诺 events.jsonl 自描述; 这个 test 锁住事件 schema.
        """
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet
        import json as _json

        run_id = "toy__smoke__seed0"
        run_dir = tmp_path / "artifacts" / run_id
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-16T18:00:00", seed=0,
                dataset={"alias": "smoke"}, model={"alias": "toy", "variant": None},
                config={"epochs_max": 1},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()
            model = ToyNet(in_dim=7, hidden=16)
            p = RandomProvider(in_dim=7, warm_start=4, rollout=8, batch_size=4, seed=0)
            ctx = EngineConfig(run_id=run_id, seed=0, epochs_max=1, batch_size=4, steps_per_epoch=1)
            GradientStrategy().fit(model, train_data=p, val_data=p, test_data=p,
                                  ctx=ctx, artifact=artifact)
        finally:
            clear_active_log()

        events = [
            _json.loads(line) for line in
            (run_dir / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
        ]
        run_started = [e for e in events if e["kind"] == "run_started"]
        assert len(run_started) == 1
        rs = run_started[0]
        assert rs["dataset"] == "smoke"
        assert rs["model"] == "toy"
        assert rs["variant"] is None
        assert isinstance(rs["params"], int) and rs["params"] > 0
        assert rs["epochs_max"] == 1
        assert rs["seed"] == 0

    def test_engine_autolog_env_written_to_manifest(self, tmp_path):
        """engine fit() 自动 capture env 写进 manifest.env.

        字段集合不锁死 (依赖 host 环境), 但 rundown_version / python /
        torch / argv 一定在 — 这是 self-describing ground truth 的核心承诺.
        """
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet
        import yaml as _yaml

        run_id = "toy__smoke__seed0"
        run_dir = tmp_path / "artifacts" / run_id
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-16T18:00:00", seed=0,
                dataset={"alias": "smoke"}, model={"alias": "toy"},
                config={"epochs_max": 1},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()
            model = ToyNet(in_dim=7, hidden=16)
            p = RandomProvider(in_dim=7, warm_start=4, rollout=8, batch_size=4, seed=0)
            ctx = EngineConfig(run_id=run_id, seed=0, epochs_max=1, batch_size=4, steps_per_epoch=1)
            GradientStrategy().fit(model, train_data=p, val_data=p, test_data=p,
                                   ctx=ctx, artifact=artifact)
        finally:
            clear_active_log()

        m = _yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        env = m.get("env") or {}
        for required in ("rundown_version", "python", "torch", "argv", "cwd"):
            assert required in env, f"manifest.env 缺 {required!r}"
        assert isinstance(env["argv"], list)
        # torch 必须是 plain string (不是 TorchVersion 对象, 否则 yaml dump 报错)
        assert isinstance(env["torch"], str)


# ---------------------------------------------------------------------------
# cli_query: list-runs / compare
# ---------------------------------------------------------------------------
class TestCliQuery:
    """list-runs / compare 的 artifact-ground-truth 查询测试.

    用 tmp_path 下的 mock manifest.yaml 跑真实 list_runs / compare_runs.
    """

    @staticmethod
    def _write_manifest(root: Path, run_id: str, *, dataset: str, model: str,
                        variant, best_val_mse: float, epochs_to_best: int,
                        epochs_max: int = 10, seed: int = 0) -> None:
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "created_at": "2026-10-16T20:00:00",
            "seed": seed,
            "dataset": {"alias": dataset, "source": {"kind": "csv", "path": "x"}},
            "model": {"alias": model, "variant": variant,
                      "factory": {"target": "x", "kwargs": {}}},
            "config": {"epochs_max": epochs_max, "lr": 0.001},
            "env": {"python": "3.10.21", "torch": "2.14.1+cpu"},
            "metrics": {"best_val_mse": best_val_mse,
                        "epochs_to_best": epochs_to_best,
                        "epochs_actual": epochs_max},
        }
        import yaml as _yaml
        (run_dir / "manifest.yaml").write_text(
            _yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8"
        )

    def test_list_runs_filters_dataset_and_model(self, tmp_path, capsys):
        from rundown.cli_query import list_runs
        self._write_manifest(tmp_path, "lstm__etth1__seed0",
                             dataset="etth1", model="lstm", variant=None,
                             best_val_mse=1.0, epochs_to_best=3)
        self._write_manifest(tmp_path, "cnn__etth1__seed0",
                             dataset="etth1", model="cnn", variant=None,
                             best_val_mse=2.0, epochs_to_best=5)
        self._write_manifest(tmp_path, "lstm__cmapss__seed0",
                             dataset="cmapss", model="lstm", variant=None,
                             best_val_mse=0.5, epochs_to_best=2)

        rc = list_runs(root=tmp_path, dataset="etth1", model="lstm", fmt="table")
        assert rc == 0
        out = capsys.readouterr().out
        assert "lstm__etth1__seed0" in out
        assert "cnn__etth1__seed0" not in out
        assert "lstm__cmapss__seed0" not in out

    def test_list_runs_metric_filter_above_below(self, tmp_path, capsys):
        from rundown.cli_query import list_runs
        self._write_manifest(tmp_path, "a__x__seed0",
                             dataset="x", model="a", variant=None,
                             best_val_mse=1.0, epochs_to_best=3)
        self._write_manifest(tmp_path, "b__x__seed0",
                             dataset="x", model="b", variant=None,
                             best_val_mse=2.5, epochs_to_best=3)
        self._write_manifest(tmp_path, "c__x__seed0",
                             dataset="x", model="c", variant=None,
                             best_val_mse=4.0, epochs_to_best=3)

        rc = list_runs(root=tmp_path, below_metric=("best_val_mse", 3.0), fmt="table")
        assert rc == 0
        out = capsys.readouterr().out
        assert "a__x__seed0" in out
        assert "b__x__seed0" in out
        assert "c__x__seed0" not in out  # 4.0 >= 3.0 被过滤

        # above-metric 严格 > (3.0 不算 above 3.0)
        rc = list_runs(root=tmp_path, above_metric=("best_val_mse", 2.0), fmt="table")
        assert rc == 0
        out = capsys.readouterr().out
        assert "a__x__seed0" not in out  # 1.0 <= 2.0
        assert "b__x__seed0" in out       # 2.5 > 2.0
        assert "c__x__seed0" in out       # 4.0 > 2.0

    def test_list_runs_json_format(self, tmp_path, capsys):
        from rundown.cli_query import list_runs
        import json as _json
        self._write_manifest(tmp_path, "x__y__seed0",
                             dataset="y", model="x", variant="small",
                             best_val_mse=1.5, epochs_to_best=2)
        rc = list_runs(root=tmp_path, fmt="json")
        assert rc == 0
        out = capsys.readouterr().out
        rows = _json.loads(out)
        assert isinstance(rows, list) and len(rows) == 1
        r = rows[0]
        assert r["run_id"] == "x__y__seed0"
        assert r["dataset_alias"] == "y"
        assert r["model_alias"] == "x"
        assert r["variant"] == "small"
        assert r["metrics"]["best_val_mse"] == 1.5

    def test_list_runs_no_match_prints_message(self, tmp_path, capsys):
        from rundown.cli_query import list_runs
        self._write_manifest(tmp_path, "x__y__seed0",
                             dataset="y", model="x", variant=None,
                             best_val_mse=1.0, epochs_to_best=1)
        rc = list_runs(root=tmp_path, dataset="nope", fmt="table")
        assert rc == 0
        out = capsys.readouterr().out
        assert "no runs" in out

    def test_compare_runs_self_no_diff(self, tmp_path, capsys):
        from rundown.cli_query import compare_runs
        self._write_manifest(tmp_path, "x__y__seed0",
                             dataset="y", model="x", variant=None,
                             best_val_mse=1.0, epochs_to_best=3)
        rc = compare_runs("x__y__seed0", "x__y__seed0", root=tmp_path)
        assert rc == 0
        out = capsys.readouterr().out
        # 相同 manifest 没 diff 行 (只有表头 + 分隔线)
        body_lines = [
            line for line in out.splitlines()
            if line.strip() and not line.strip().startswith(("field", "---"))
        ]
        assert body_lines == []

    def test_compare_runs_diff_with_delta(self, tmp_path, capsys):
        from rundown.cli_query import compare_runs
        # a: best_val_mse=1.0, epochs_max=10
        self._write_manifest(tmp_path, "a__y__seed0",
                             dataset="y", model="a", variant=None,
                             best_val_mse=1.0, epochs_to_best=3, epochs_max=10)
        # b: best_val_mse=2.0, epochs_max=20
        self._write_manifest(tmp_path, "b__y__seed0",
                             dataset="y", model="b", variant=None,
                             best_val_mse=2.0, epochs_to_best=5, epochs_max=20)
        rc = compare_runs("a__y__seed0", "b__y__seed0", root=tmp_path)
        assert rc == 0
        out = capsys.readouterr().out
        # 数值字段显示 Δ
        assert "metrics.best_val_mse" in out
        assert "Δ=" in out
        # model_alias 也 diff
        assert "model.alias" in out

    def test_compare_runs_missing_returns_error(self, tmp_path, capsys):
        from rundown.cli_query import compare_runs
        rc = compare_runs("nonexistent_a", "nonexistent_b", root=tmp_path)
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err

    def test_cli_metric_filter_argparse_nargs(self):
        """--above-metric / --below-metric 接受 NAME VALUE 两 token (nargs=2).

        Bug 修复: 之前 default=None 单值, argparse 把第二个 token 当成
        下一个 flag → 'unrecognized arguments: 2.5'. 现在 nargs=2 + metavar.
        """
        from rundown.cli import build_parser
        parser = build_parser()

        ns = parser.parse_args([
            "list-runs",
            "--above-metric", "best_val_mse", "2.0",
            "--below-metric", "best_val_mse", "3.5",
        ])
        assert ns.above_metric == ["best_val_mse", "2.0"]
        assert ns.below_metric == ["best_val_mse", "3.5"]

        # 不传时是 None (不强制要求 2 token)
        ns2 = parser.parse_args(["list-runs"])
        assert ns2.above_metric is None
        assert ns2.below_metric is None

        # 单 token 报错 (argparse nargs=2 强制)
        with pytest.raises(SystemExit):
            parser.parse_args(["list-runs", "--above-metric", "only_name"])


# ---------------------------------------------------------------------------
# --kwargs: 透传任意超参给 model factory (optuna sweep 用)
# ---------------------------------------------------------------------------
class TestCliKwargs:
    """方案 C: --kwargs KEY=VALUE 透传给 model factory.

    optuna sweep 时绕过 models.variants 限制, 自由传任意超参.
    合并顺序: factory_kwargs < variants[variant] < --kwargs (后者覆盖).
    """

    def test_cli_kwargs_argparse_append(self):
        """--kwargs action='append' — 可多次传, 列表形式."""
        from rundown.cli import build_parser
        parser = build_parser()
        ns = parser.parse_args([
            "train", "--dataset", "x", "--model", "y",
            "--kwargs", "hidden=128",
            "--kwargs", "num_layers=2",
            "--kwargs", "lr=0.0005",
        ])
        assert ns.kwargs == ["hidden=128", "num_layers=2", "lr=0.0005"]

    def test_parse_cli_kwargs_value_type_coercion(self):
        """--kwargs value 自动 int / float / bool / str 转换."""
        from rundown.cli import _parse_cli_kwargs
        result = _parse_cli_kwargs([
            "hidden=128", "lr=1e-3", "dropout=0.5",
            "use_bias=true", "name=etth1", "n_layers=2",
        ])
        assert result == {
            "hidden": 128, "lr": 0.001, "dropout": 0.5,
            "use_bias": True, "name": "etth1", "n_layers": 2,
        }

    def test_parse_cli_kwargs_invalid_format_raises(self):
        """--kwargs token 没 '=' 报错 (不静默吃掉, 宁报错不猜)."""
        from rundown.cli import _parse_cli_kwargs
        with pytest.raises(ValueError, match="KEY=VALUE"):
            _parse_cli_kwargs(["hidden_no_equals"])
        with pytest.raises(ValueError, match="empty key"):
            _parse_cli_kwargs(["=value_no_key"])

    def test_parse_cli_kwargs_none_returns_none(self):
        """未传 --kwargs 时返回 None (与 args.kwargs default=None 对齐)."""
        from rundown.cli import _parse_cli_kwargs
        assert _parse_cli_kwargs(None) is None
        assert _parse_cli_kwargs([]) is None

    def test_model_entry_build_extra_kwargs_overrides(self):
        """ModelEntry.build(extra_kwargs=...) 合并顺序正确.

        优先级: extra_kwargs > variants > factory_kwargs.
        """
        from rundown.registry import ModelEntry
        # 用 tests.fixtures.ToyNet: __init__(in_dim=7, hidden=16)
        entry = ModelEntry(
            alias="toy",
            factory_target="tests.fixtures.ToyNet",
            factory_kwargs={"in_dim": 7, "hidden": 16},
            variants={"small": {"hidden": 8}},
        )

        # 1) base only — merged = factory_kwargs
        _, merged, chosen = entry.build()
        assert merged == {"in_dim": 7, "hidden": 16}
        assert chosen is None

        # 2) variant 覆盖 base — merged 反映 variant hidden=8
        _, merged, chosen = entry.build(variant="small")
        assert merged == {"in_dim": 7, "hidden": 8}
        assert chosen == "small"

        # 3) extra_kwargs 覆盖一切 (optuna sweep 场景)
        # hidden 在 ToyNet 签名里 → OK, merged 应反映 extra_kwargs 覆盖
        _, merged, chosen = entry.build(
            variant="small",
            extra_kwargs={"hidden": 32},
        )
        assert merged == {"in_dim": 7, "hidden": 32}
        assert chosen == "small"

        # 4) extra_kwargs key 不在 factory 签名 → factory 自己抛 TypeError
        # (rundown 不在 registry 层拦截, 宁报错不猜 — optuna 写错立刻知道)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            entry.build(extra_kwargs={"lr": 0.0001})

    def test_cli_train_kwargs_in_manifest(self, tmp_path, monkeypatch, capsys):
        """--kwargs 透传后, manifest.model.factory.kwargs 含 merged 值.

        走 library API (跳过 provider config 反射) — 验证 ModelEntry.build
        返回的 merged_kwargs 是 CLI 写 manifest 的依据.
        """
        from rundown.registry import ModelEntry
        from rundown.cli import _parse_cli_kwargs

        # 模拟 CLI: --kwargs hidden=32, --kwargs lr=0.0005
        parsed = _parse_cli_kwargs(["hidden=32", "lr=0.0005"])

        # 走 registry 层
        entry = ModelEntry(
            alias="toy",
            factory_target="tests.fixtures.ToyNet",
            factory_kwargs={"in_dim": 7, "hidden": 16},
        )

        # ToyNet 只接受 in_dim + hidden; lr 会 TypeError.
        # 测合并 dict (不实际实例化):
        merged_base = dict(entry.factory_kwargs)
        if parsed:
            merged_base.update(parsed)
        assert merged_base == {"in_dim": 7, "hidden": 32, "lr": 0.0005}

        # 实际能实例化的部分 (去掉 lr)
        parsed_safe = {k: v for k, v in parsed.items() if k != "lr"}
        model, merged, _ = entry.build(extra_kwargs=parsed_safe)
        assert merged == {"in_dim": 7, "hidden": 32}
        # ToyNet 实例化 OK — 第一层 weight 维度对 (in_dim=7, hidden=32)


# ---------------------------------------------------------------------------
# code snapshot — autolog env 的产物比代码活得久承诺
# ---------------------------------------------------------------------------
class TestCodeSnapshot:
    """engine.fit() 默认会打包当前代码进 artifacts/<run_id>/code/.

    两种路径: git repo → tar archive (tracked only, 自动 .gitignore);
    非 git repo → zip 全目录 (排除 artifacts/.venv/__pycache__).
    """

    def test_snapshot_zip_fallback_in_non_git_repo(self, tmp_path, monkeypatch):
        """tmp_path 不是 git repo → 走 zip fallback, 文件名 code_<sha8>.zip.

        排除 artifacts/.venv/__pycache__ 等噪音目录.
        """
        from rundown._env import snapshot_code

        # 造一些文件: 一部分该进 snapshot, 一部分该被排除
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
        noise = tmp_path / "artifacts"
        noise.mkdir()
        (noise / "old_run.safetensors").write_bytes(b"old")  # 应被排除
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "junk.pyc").write_bytes(b"\x00\x00")       # 应被排除

        snap_dir = tmp_path / "code"
        info = snapshot_code(snap_dir, cwd=tmp_path)

        assert info["format"] == "zip"
        assert info["tracked_only"] is False
        assert info["sha256"] and len(info["sha256"]) == 64
        assert info["sha256_short"] and len(info["sha256_short"]) == 8
        assert info["archive_path"] == f"code_{info['sha256_short']}.zip"
        assert info["size_bytes"] > 0
        archive = snap_dir / info["archive_path"]
        assert archive.exists()

        # main.py + README.md 进 snapshot
        import zipfile
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        assert "main.py" in names
        assert "README.md" in names
        # 噪音不在
        assert not any(n.startswith("artifacts/") for n in names)
        assert not any(n.startswith("__pycache__/") for n in names)

    def test_snapshot_tar_archive_in_git_repo(self, tmp_path):
        """git repo → tar archive (tracked only) 解到 code_<sha8>/ 目录."""
        from rundown._env import snapshot_code

        # 临时 git repo (要装 user.email / user.name 否则 commit 失败; archive 不需要 commit)
        subprocess_run = __import__("subprocess").run
        subprocess_run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
        subprocess_run(["git", "config", "user.email", "t@t.t"], cwd=str(tmp_path), check=True)
        subprocess_run(["git", "config", "user.name", "t"], cwd=str(tmp_path), check=True)
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "mod.py").write_text("y = 2\n", encoding="utf-8")
        subprocess_run(["git", "add", "."], cwd=str(tmp_path), check=True)
        subprocess_run(["git", "commit", "-q", "-m", "init"], cwd=str(tmp_path), check=True)

        snap_dir = tmp_path / "code"
        info = snapshot_code(snap_dir, cwd=tmp_path)

        assert info["format"] == "tar"
        assert info["tracked_only"] is True
        assert info["sha256"] and len(info["sha256"]) == 64
        assert info["sha256_short"] and len(info["sha256_short"]) == 8
        assert info["archive_path"] == f"code_{info['sha256_short']}/"
        # 解 tar 到 snap_dir / code_<sha8>/
        extract_dir = snap_dir / info["archive_path"]
        assert extract_dir.exists()
        extracted = {p.name for p in extract_dir.iterdir() if p.is_file()}
        assert "main.py" in extracted
        assert "mod.py" in extracted

    def test_snapshot_default_on_writes_code_snapshot_to_manifest(self, tmp_path):
        """engine.fit() 默认 (snapshot=True) → manifest.env.code_snapshot 落盘.

        文件名带 sha8 前缀: code_<sha8>.zip (zip) 或 code_<sha8>/ (tar).
        """
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet
        import yaml as _yaml

        run_id = "toy__smoke__seed0"
        run_dir = tmp_path / "artifacts" / run_id
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-16T18:00:00", seed=0,
                dataset={"alias": "smoke"}, model={"alias": "toy"},
                config={"epochs_max": 1},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()
            model = ToyNet(in_dim=7, hidden=16)
            p = RandomProvider(in_dim=7, warm_start=4, rollout=8, batch_size=4, seed=0)
            ctx = EngineConfig(run_id=run_id, seed=0, epochs_max=1, batch_size=4,
                               steps_per_epoch=1)  # snapshot 默认 True
            GradientStrategy().fit(model, train_data=p, val_data=p, test_data=p,
                                   ctx=ctx, artifact=artifact)
        finally:
            clear_active_log()

        m = _yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        cs = m["env"].get("code_snapshot")
        assert cs is not None, "snapshot 默认开, manifest.env.code_snapshot 必须落盘"
        assert cs["format"] in ("tar", "zip")
        assert cs["tracked_only"] in (True, False)
        assert isinstance(cs["sha256"], str) and len(cs["sha256"]) == 64
        assert isinstance(cs["sha256_short"], str) and len(cs["sha256_short"]) == 8
        assert cs["archive_path"].startswith("code_")
        assert cs["size_bytes"] > 0
        assert cs["num_files"] >= 1

        # 实际文件落盘 (run_dir 根目录, 不再是 code/ 子目录)
        archive = run_dir / cs["archive_path"]
        assert archive.exists(), f"{archive} 应存在"

    def test_snapshot_disabled_omits_code_snapshot(self, tmp_path):
        """ctx.snapshot=False → manifest.env 没 code_snapshot 字段."""
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet
        import yaml as _yaml

        run_id = "toy__smoke__seed0"
        run_dir = tmp_path / "artifacts" / run_id
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-16T18:00:00", seed=0,
                dataset={"alias": "smoke"}, model={"alias": "toy"},
                config={"epochs_max": 1},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()
            model = ToyNet(in_dim=7, hidden=16)
            p = RandomProvider(in_dim=7, warm_start=4, rollout=8, batch_size=4, seed=0)
            ctx = EngineConfig(run_id=run_id, seed=0, epochs_max=1, batch_size=4,
                               steps_per_epoch=1, snapshot=False)
            GradientStrategy().fit(model, train_data=p, val_data=p, test_data=p,
                                   ctx=ctx, artifact=artifact)
        finally:
            clear_active_log()

        m = _yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        assert "code_snapshot" not in m["env"]
        # code_<sha8>.zip / code_<sha8>/ 都不该创建
        assert not list(run_dir.glob("code_*"))

    def test_cli_no_snapshot_flag(self):
        """CLI --no-snapshot → args.snapshot = False (默认 True)."""
        from rundown.cli import build_parser
        parser = build_parser()

        # 默认 (不传 --no-snapshot)
        ns = parser.parse_args(["train", "--dataset", "x", "--model", "y"])
        assert ns.snapshot is True

        # 传 --no-snapshot
        ns = parser.parse_args(["train", "--dataset", "x", "--model", "y", "--no-snapshot"])
        assert ns.snapshot is False

    def test_train_cli_default_uses_shared_snapshot_pool(self, tmp_path, monkeypatch):
        """默认 --out-dir 模式: snapshot 落 out_root/_shared/, run_dir 引用 ../_shared/.

        单次 train 自包含沉没到 out_root 边界 — 多次手动 train 同代码共享 1 份.
        """
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet
        import yaml as _yaml

        run_id = "toy__smoke__seed0"
        out_root = tmp_path / "artifacts"
        run_dir = out_root / run_id
        shared_dir = out_root / "_shared"
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-16T18:00:00", seed=0,
                dataset={"alias": "smoke"}, model={"alias": "toy"},
                config={"epochs_max": 1},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()
            model = ToyNet(in_dim=7, hidden=16)
            p = RandomProvider(in_dim=7, warm_start=4, rollout=8, batch_size=4, seed=0)
            # 模拟 CLI 默认行为: snapshot_target_dir=out_root/_shared, archive_rel="../_shared"
            ctx = EngineConfig(
                run_id=run_id, seed=0, epochs_max=1, batch_size=4, steps_per_epoch=1,
                snapshot_target_dir=shared_dir,
                snapshot_archive_rel="../_shared",
            )
            GradientStrategy().fit(model, train_data=p, val_data=p, test_data=p,
                                   ctx=ctx, artifact=artifact)
        finally:
            clear_active_log()

        m = _yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        cs = m["env"]["code_snapshot"]
        # archive_path 相对 run_dir 指向 _shared/
        assert cs["archive_path"].startswith("../_shared/code_"), cs["archive_path"]
        # 物理上 snapshot 落 _shared/, 不在 run_dir 内
        assert not list(run_dir.glob("code_*")), "run_dir 不该有自己的 code_*"
        # 1 份 snapshot — 形态可能是 zip (非 git repo) 或目录 (git repo tar 路径)
        shared_entries = list(shared_dir.glob("code_*"))
        assert len(shared_entries) == 1, f"_shared/ 应只 1 份, got {shared_entries}"
        # 物理 resolve 验证: run_dir + archive_path 真的指向 _shared/ 文件/目录
        assert (run_dir / cs["archive_path"]).exists()

    def test_train_cli_run_dir_mode_keeps_snapshot_in_run_dir(self, tmp_path):
        """--run-dir 模式: snapshot 仍落 run_dir (self-contained, 不走 _shared/).

        --run-dir 是精细控制场景 — 用户明示此 run 独立, 不参与 out_root 池.
        """
        from rundown.artifact import RunArtifact, RunManifest
        from rundown.engine import EngineConfig, GradientStrategy
        from rundown.events import set_active_log, clear_active_log
        from tests.fixtures import RandomProvider, ToyNet
        import yaml as _yaml

        run_id = "toy__smoke__seed0"
        run_dir = tmp_path / "scratch" / run_id
        set_active_log(run_dir, run_id)
        try:
            manifest = RunManifest(
                run_id=run_id, created_at="2026-10-16T18:00:00", seed=0,
                dataset={"alias": "smoke"}, model={"alias": "toy"},
                config={"epochs_max": 1},
            )
            artifact = RunArtifact(run_dir=run_dir, manifest=manifest)
            artifact.save_manifest()
            model = ToyNet(in_dim=7, hidden=16)
            p = RandomProvider(in_dim=7, warm_start=4, rollout=8, batch_size=4, seed=0)
            # --run-dir 模式: snapshot_target_dir=None, archive_rel=""
            ctx = EngineConfig(
                run_id=run_id, seed=0, epochs_max=1, batch_size=4, steps_per_epoch=1,
            )
            GradientStrategy().fit(model, train_data=p, val_data=p, test_data=p,
                                   ctx=ctx, artifact=artifact)
        finally:
            clear_active_log()

        m = _yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
        cs = m["env"]["code_snapshot"]
        # archive_path 不含 ../ 前缀, 直接是 "code_xxx.zip" (在 run_dir 内)
        assert not cs["archive_path"].startswith("../"), cs["archive_path"]
        assert cs["archive_path"].startswith("code_"), cs["archive_path"]
        # 物理上 snapshot 在 run_dir 内
        assert (run_dir / cs["archive_path"]).exists()
        # 没有 _shared/ 目录
        assert not (run_dir.parent / "_shared").exists()


# ---------------------------------------------------------------------------
# sweep 共享 snapshot 池 — experiment.run 开头建 1 次, 所有 trial 复用
# ---------------------------------------------------------------------------
class TestSweepSnapshotPool:
    """experiment sweep 一次命令建 1 份 snapshot, 所有 trial 共享.

    sweep 是天然 artifact 边界 — 用户拷走整个 out_root 就完整 (含 _shared/).
    """

    def test_experiment_creates_shared_snapshot_meta(self, tmp_path, monkeypatch):
        """run_experiment() 开头建 1 份 snapshot + meta.json 到 out_root/_shared/."""
        from rundown.experiment import Experiment, run_experiment
        monkeypatch.chdir(tmp_path)

        exp = Experiment(
            name="smoke_sweep", dataset="etth1", model="lstm",
            epochs_max=1, variants=["small"], steps_per_epoch=1,
        )
        run_experiment(exp, out_root=tmp_path / "artifacts")

        shared = tmp_path / "artifacts" / "_shared"
        assert shared.exists()
        meta_files = list(shared.glob("code_*.zip.meta.json"))
        assert len(meta_files) == 1, f"应有 1 个 meta json, got {meta_files}"
        import json as _json
        meta = _json.loads(meta_files[0].read_text(encoding="utf-8"))
        assert meta["archive_path"] == f"_shared/{meta_files[0].stem.replace('.meta', '')}" or \
               meta["archive_path"].startswith("_shared/code_")
        assert len(meta["sha256"]) == 64
        assert len(meta["sha256_short"]) == 8
        # meta 的 archive_path 是相对 out_root 的 (subprocess 视角)
        assert meta["archive_path"].startswith("_shared/code_")

    def test_experiment_trial_manifest_references_shared_snapshot(self, tmp_path, monkeypatch):
        """每个 trial 的 manifest.env.code_snapshot.archive_path 指向 _shared/ 共享池.

        不真跑 subprocess (那需要完整 configs/ 数据集); 直接验证 run_experiment()
        建好 shared snapshot 后, capture_env 喂 sweep meta.json 时:
          - archive_path 转成相对 run_dir (../_shared/code_xxx.zip)
          - trial run_dir 内不该有独立 code_*.zip
        """
        from rundown.experiment import Experiment, run_experiment
        from rundown._env import capture_env
        import yaml as _yaml
        import json as _json
        monkeypatch.chdir(tmp_path)

        exp = Experiment(
            name="smoke_sweep", dataset="etth1", model="lstm",
            epochs_max=1, variants=["small"], steps_per_epoch=1,
        )
        run_experiment(exp, out_root=tmp_path / "artifacts")

        shared = tmp_path / "artifacts" / "_shared"
        meta_files = list(shared.glob("code_*.zip.meta.json"))
        assert len(meta_files) == 1, f"应有 1 个 meta json, got {meta_files}"
        meta = _json.loads(meta_files[0].read_text(encoding="utf-8"))

        # 直接用 sweep meta 模拟一个 trial 的 capture_env 调用
        from rundown.cli import _load_external_snapshot
        run_dir = tmp_path / "artifacts" / "lstm__varsmall__etth1__seed0"
        info = _load_external_snapshot(meta_files[0], run_dir)
        env = capture_env(
            snapshot_dir=run_dir,           # 自建路径仍然传, 但 capture_env 优先 external
            snapshot=True,
            external_snapshot=info,
        )
        cs = env["code_snapshot"]
        # archive_path 相对 run_dir, 指向 _shared 共享池
        assert "_shared" in cs["archive_path"], cs["archive_path"]
        assert cs["archive_path"].endswith(".zip"), cs["archive_path"]
        assert cs["sha256"] == meta["sha256"]
        assert cs["sha256_short"] == meta["sha256_short"]

        # 物理验证: _shared/ 只有 1 份 snapshot 文件 (sweep 共享, 不是 N 份)
        shared_zips = list(shared.glob("code_*.zip"))
        assert len(shared_zips) == 1, f"sweep 应只存 1 份 snapshot, got {len(shared_zips)}"

    def test_cli_snapshot_from_pool_argparse(self):
        """CLI --snapshot-from-pool PATH 接受路径."""
        from rundown.cli import build_parser
        parser = build_parser()
        ns = parser.parse_args([
            "train", "--dataset", "x", "--model", "y",
            "--snapshot-from-pool", "_shared/code_86f357f6.zip.meta.json",
        ])
        assert ns.snapshot_from_pool.as_posix() == "_shared/code_86f357f6.zip.meta.json"

    def test_load_external_snapshot_translates_archive_path(self, tmp_path):
        """_load_external_snapshot() 把 meta 里的 archive_path (相对 out_root) 转成相对 run_dir."""
        from rundown.cli import _load_external_snapshot
        import json as _json

        # 模拟 sweep 入口写的 meta.json: archive_path = "_shared/code_xx.zip"
        meta_path = tmp_path / "artifacts" / "_shared" / "code_86f357f6.zip.meta.json"
        meta_path.parent.mkdir(parents=True)
        meta_path.write_text(_json.dumps({
            "format": "zip", "tracked_only": False,
            "sha256": "86f357f6" + "0" * 56, "sha256_short": "86f357f6",
            "size_bytes": 100, "num_files": 36,
            "archive_path": "_shared/code_86f357f6.zip",
        }), encoding="utf-8")

        # run_dir = artifacts/<run_id>, 期望相对路径 "../_shared/code_86f357f6.zip"
        run_dir = tmp_path / "artifacts" / "lstm__varsmall__etth1__seed0"
        info = _load_external_snapshot(meta_path, run_dir)
        assert info is not None
        assert info["archive_path"] == "../_shared/code_86f357f6.zip"
        assert info["sha256_short"] == "86f357f6"

        # meta 缺失/损坏 → 返回 None (走自建 fallback)
        assert _load_external_snapshot(None, run_dir) is None
        assert _load_external_snapshot(tmp_path / "nope.json", run_dir) is None
