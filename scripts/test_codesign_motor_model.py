"""Tests for the actuator mass/cost models used by GA co-design."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from codesign_motor_model import (  # noqa: E402
  DEFAULT_ALPHA,
  DEFAULT_CATALOG,
  MotorCostModel,
  powerlaw_k,
)


def test_powerlaw_anchored_at_100nm():
  m = MotorCostModel(kind="powerlaw", alpha=DEFAULT_ALPHA)
  assert m.joint_mass(100.0) == pytest.approx(1.2, rel=1e-6)


def test_powerlaw_is_sublinear():
  """Doubling torque should less-than-double mass (torque density improves)."""
  m = MotorCostModel(kind="powerlaw")
  m50, m100 = m.joint_mass(50.0), m.joint_mass(100.0)
  assert m100 < 2.0 * m50
  # And strictly increasing.
  assert m100 > m50


def test_linear_matches_powerlaw_at_anchor():
  lin = MotorCostModel(kind="linear")
  pw = MotorCostModel(kind="powerlaw")
  assert lin.joint_mass(100.0) == pytest.approx(pw.joint_mass(100.0), rel=1e-6)
  # Linear over-weighs large motors relative to the power law.
  assert lin.joint_mass(200.0) > pw.joint_mass(200.0)


def test_powerlaw_k_consistent():
  k = powerlaw_k(0.75)
  assert k * 100.0**0.75 == pytest.approx(1.2, rel=1e-9)


def test_catalog_selects_cheapest_covering_sku():
  m = MotorCostModel(kind="catalog")
  # 70 Nm demand -> QDD-90 (first SKU with peak >= 70).
  assert m._select_sku(70.0).name == "QDD-90"
  # Exact boundary picks that SKU.
  assert m._select_sku(60.0).name == "QDD-60"
  # Above the catalog snaps to the largest.
  assert m._select_sku(500.0).name == DEFAULT_CATALOG[-1].name


def test_catalog_mass_is_monotone_nondecreasing():
  m = MotorCostModel(kind="catalog")
  taus = np.linspace(40, 200, 40)
  masses = [m.joint_mass(t) for t in taus]
  assert all(masses[i + 1] >= masses[i] for i in range(len(masses) - 1))


def test_design_mass_sums_over_joints():
  m = MotorCostModel(kind="powerlaw")
  tau = np.array([100.0, 100.0, 100.0])
  assert m.design_mass(tau) == pytest.approx(3 * m.joint_mass(100.0), rel=1e-9)


def test_distinct_sku_count():
  m = MotorCostModel(kind="catalog")
  # Two joints on QDD-40, two mapping to QDD-90 -> 2 distinct SKUs.
  tau = np.array([35.0, 40.0, 70.0, 85.0])
  assert m.n_distinct_skus(tau) == 2
  assert m.sku_names(tau) == ["QDD-40", "QDD-40", "QDD-90", "QDD-90"]


def test_unknown_kind_raises():
  with pytest.raises(ValueError):
    MotorCostModel(kind="bogus")
