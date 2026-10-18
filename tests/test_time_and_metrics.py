"""Tests for the three surgical fixes:

  A. cli._compute_metric — missing y_hat / y_true raises ValueError (not TypeError)
  B. cli._utc_now_iso — UTC ISO 8601 with offset, second precision
  C. inspect_run._parse_created_at — naive → local tz; aware → as-is; bad → None

No network, no GPU, no big files.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import torch

from rundown.cli import _compute_metric, _utc_now_iso
from rundown.inspect_run import _parse_created_at


# ============================================================================
# A. _compute_metric — ValueError on missing y_hat / y_true
# ============================================================================

class TestComputeMetricPredictionErrorStd:
    def test_missing_keys_raises_value_error_not_type_error(self) -> None:
        # Regression: previously `err = predictions.get("y_hat") - predictions.get("y_true")`
        # ran unconditionally and raised TypeError ("unsupported operand types: NoneType - NoneType").
        with pytest.raises(ValueError) as ei:
            _compute_metric(
                {"y_hat": torch.zeros(2, 3)},
                "prediction_error_std",
                None,
            )
        msg = str(ei.value)
        assert "y_hat" in msg and "y_true" in msg, (
            f"error message should mention y_hat / y_true, got: {msg!r}"
        )

    def test_missing_y_true_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _compute_metric(
                {"y_true": torch.zeros(2, 3)},
                "prediction_error_std",
                None,
            )

    def test_happy_path_zero_error(self) -> None:
        t = torch.zeros(4, 5)
        out = _compute_metric(
            {"y_hat": t, "y_true": t},
            "prediction_error_std",
            None,
        )
        assert "total" in out
        assert out["total"] == 0.0


# ============================================================================
# B. _utc_now_iso — UTC ISO 8601 with offset
# ============================================================================

class TestUtcNowIso:
    def test_parses_with_fromisoformat_and_is_utc(self) -> None:
        s = _utc_now_iso()
        parsed = datetime.fromisoformat(s)
        assert parsed.tzinfo is not None, (
            f"expected tzinfo attached, got naive string {s!r}"
        )
        assert parsed.utcoffset() == timedelta(0), (
            f"expected UTC offset (0), got {parsed.utcoffset()} from {s!r}"
        )

    def test_seconds_precision(self) -> None:
        # ISO with offset should not contain sub-second digits
        s = _utc_now_iso()
        # Strip offset suffix "...+00:00" then the dot-and-fraction
        head = s.rsplit("+", 1)[0].rsplit("-", 1)[0]
        assert "." not in head, f"expected second precision, got {s!r}"


# ============================================================================
# C. _parse_created_at — naive → local tz, aware → as-is, bad → None
# ============================================================================

class TestParseCreatedAt:
    def test_naive_string_attaches_local_tz(self) -> None:
        ts = _parse_created_at("2026-10-18T08:00:00")
        assert ts is not None
        assert ts.tzinfo is not None, "naive input should become aware"
        expected_offset = datetime.now().astimezone().utcoffset()
        assert ts.utcoffset() == expected_offset, (
            f"naive timestamp should carry local tz, "
            f"got offset={ts.utcoffset()} vs local={expected_offset}"
        )
        # Wall-clock value must be preserved (8:00 stays 8:00 local).
        assert ts.hour == 8 and ts.minute == 0

    def test_aware_utc_keeps_offset(self) -> None:
        ts = _parse_created_at("2026-10-18T08:00:00+00:00")
        assert ts is not None
        assert ts.tzinfo is not None
        assert ts.utcoffset() == timedelta(0), (
            f"aware UTC string must keep its UTC offset, got {ts.utcoffset()}"
        )
        assert ts.hour == 8 and ts.minute == 0

    def test_aware_non_utc_keeps_offset(self) -> None:
        # +08:00 offset must NOT be re-interpreted as local — stays at +08:00.
        ts = _parse_created_at("2026-10-18T08:00:00+08:00")
        assert ts is not None
        assert ts.utcoffset() == timedelta(hours=8)

    def test_unparseable_returns_none(self) -> None:
        assert _parse_created_at("not-a-date") is None

    def test_none_returns_none(self) -> None:
        assert _parse_created_at(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_created_at("") is None

    def test_non_string_returns_none(self) -> None:
        assert _parse_created_at(12345) is None
        assert _parse_created_at([]) is None
