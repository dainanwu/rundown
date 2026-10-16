"""Run 产物: 写 / 读 manifest + weights + curves + predictions.

MVP 阶段后端: safetensors (张量) + yaml (manifest) + jsonl (events).
读路径必须能跨年读回; 写路径必须原子 (Windows 上绕 error 1224).

每个 run 目录约定布局 (rundown/cli.py 写出时也按此):
    <run_dir>/
      manifest.yaml
      weights.safetensors
      weights_best.safetensors
      curves.safetensors
      events.jsonl       (由 rundown.events 写)
      predictions/
        train.safetensors
        val.safetensors
        test.safetensors
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch
from safetensors import safe_open
from safetensors.torch import save_file as _st_save


def _atomic_save_safetensors(tensors: Dict[str, torch.Tensor], final_path: Path) -> None:
    """写 safetensors: 临时文件 + rename, 绕 Windows error 1224.

    见 PST_Model/io/save_load.py:_atomic_safetensors_save 的同款问题与解法.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    payload = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
    for attempt in range(8):
        try:
            if tmp.exists():
                tmp.unlink()
            _st_save(payload, str(tmp))
            os.replace(str(tmp), str(final_path))
            return
        except (OSError, RuntimeError, ValueError, TypeError):
            # safetensors 抛 RuntimeError (损坏) / ValueError (shape 不对); OSError (Windows share)
            # 其他异常 (KeyboardInterrupt 等) 不应该被吞 — 让它正常抛出.
            time.sleep(0.05 * (attempt + 1))
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    # 兜底
    _st_save(payload, str(final_path))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass
class RunManifest:
    run_id: str
    created_at: str
    seed: int
    dataset: Dict[str, Any]
    model:   Dict[str, Any]
    config:  Dict[str, Any] = field(default_factory=dict)
    code:    Dict[str, Any] = field(default_factory=dict)
    env:     Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    derived_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "RunManifest":
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"manifest: expected mapping, got {type(data).__name__}")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, run_dir: Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.yaml"
        path.write_text(self.to_yaml(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, run_dir: Path) -> "RunManifest":
        path = Path(run_dir) / "manifest.yaml"
        if not path.exists():
            raise FileNotFoundError(f"manifest.yaml not found in {run_dir}")
        return cls.from_yaml(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 读路径
# ---------------------------------------------------------------------------
@dataclass
class RunArtifact:
    run_dir: Path
    manifest: RunManifest
    weights:     Optional[Dict[str, torch.Tensor]] = None
    weights_best: Optional[Dict[str, torch.Tensor]] = None
    curves:      Optional[Dict[str, torch.Tensor]] = None
    predictions: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    @classmethod
    def read(
        cls,
        run_dir: Path,
        *,
        include: Optional[List[str]] = None,
    ) -> "RunArtifact":
        """读一个 run. include 控制读哪些文件; None = 全部."""
        run_dir = Path(run_dir)
        if not (run_dir / "manifest.yaml").exists():
            raise FileNotFoundError(
                f"no manifest.yaml in {run_dir}; not a rundown run?"
            )
        manifest = RunManifest.load(run_dir)
        want = set(include) if include else {
            "weights", "weights_best", "curves", "predictions",
        }

        def _load(p: Path) -> Dict[str, torch.Tensor]:
            if not p.exists():
                return {}
            with safe_open(str(p), framework="pt", device="cpu") as f:
                return {k: f.get_tensor(k) for k in f.keys()}

        out = cls(run_dir=run_dir, manifest=manifest)
        if "weights" in want:
            out.weights = _load(run_dir / "weights.safetensors")
        if "weights_best" in want:
            out.weights_best = _load(run_dir / "weights_best.safetensors")
        if "curves" in want:
            out.curves = _load(run_dir / "curves.safetensors")
        if "predictions" in want:
            preds_dir = run_dir / "predictions"
            if preds_dir.exists():
                for f in sorted(preds_dir.glob("*.safetensors")):
                    out.predictions[f.stem] = _load(f)
        return out

    # ----- 写路径 -----
    def save_manifest(self) -> Path:
        return self.manifest.save(self.run_dir)

    def write_weights(self, state: Dict[str, torch.Tensor], *, best: bool = False) -> Path:
        name = "weights_best.safetensors" if best else "weights.safetensors"
        path = Path(self.run_dir) / name
        _atomic_save_safetensors(state, path)
        if best:
            self.weights_best = {k: v.detach().cpu() for k, v in state.items()}
        else:
            self.weights = {k: v.detach().cpu() for k, v in state.items()}
        return path

    def write_curves(self, curves: Dict[str, torch.Tensor]) -> Path:
        path = Path(self.run_dir) / "curves.safetensors"
        _atomic_save_safetensors(curves, path)
        self.curves = {k: v.detach().cpu() for k, v in curves.items()}
        return path

    def write_predictions(self, split: str, tensors: Dict[str, torch.Tensor]) -> Path:
        preds_dir = Path(self.run_dir) / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        path = preds_dir / f"{split}.safetensors"
        _atomic_save_safetensors(tensors, path)
        self.predictions[split] = {k: v.detach().cpu() for k, v in tensors.items()}
        return path


__all__ = ["RunManifest", "RunArtifact", "_atomic_save_safetensors"]
