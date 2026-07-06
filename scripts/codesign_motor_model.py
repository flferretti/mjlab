"""Physically-grounded actuator mass/cost models for motor co-design.

Motivation
----------
The GA co-design (``codesign_ga.py``) originally used a *linear* proxy for the
hardware cost of a design: ``cum_tau = sum(tau_g)``, i.e. it treated a motor's
mass/size as directly proportional to its peak torque. That is convenient but
physically wrong. For brushless (BLDC) / quasi-direct-drive (QDD) actuators the
mass scales *sub-linearly* with peak torque:

    m(tau) ~= k * tau ** alpha,   alpha ~= 0.7 - 0.8

so torque density (Nm/kg) slowly improves with size. Using a linear proxy
therefore over-penalizes large motors and biases the Pareto front toward many
small motors that in reality weigh almost as much as a few big ones.

This module provides three interchangeable cost models, all returning an
estimated **mass in kg** (a physically meaningful, additive quantity):

* ``linear``   -- mass proportional to peak torque (reproduces the legacy
  behavior up to a constant; kept for backwards-compatible comparisons).
* ``powerlaw`` -- mass = k * tau ** alpha, the grounded default (alpha=0.75).
* ``catalog``  -- snap each joint to the cheapest real SKU whose peak torque
  covers the demand; mass and the number of *distinct* motor SKUs come straight
  from the catalog (no post-hoc clustering needed).

The catalog model is the most realistic for procurement: it yields exact masses,
an exact bill-of-materials (distinct SKU count), and optionally a monetary cost.

All models operate on a *per-joint* torque vector (both left/right joints of a
symmetric pair are counted, matching the physical hardware that must be bought).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Power-law anchor.
# ---------------------------------------------------------------------------
# Anchored so a 100 Nm-peak humanoid QDD actuator weighs ~1.2 kg, which is
# representative of T-Motor AK-/RMD-class and MIT-mini-cheetah-derived units.
# k = mass_anchor / tau_anchor ** alpha.
DEFAULT_ALPHA: float = 0.75
_TAU_ANCHOR_NM: float = 100.0
_MASS_ANCHOR_KG: float = 1.2


def powerlaw_k(alpha: float = DEFAULT_ALPHA) -> float:
  """Return the power-law constant k anchored to a 100 Nm / 1.2 kg actuator."""
  return _MASS_ANCHOR_KG / (_TAU_ANCHOR_NM**alpha)


# ---------------------------------------------------------------------------
# Discrete motor catalog (representative humanoid QDD SKUs).
# ---------------------------------------------------------------------------
# Masses follow the anchored power law (alpha=0.75) rounded to plausible values;
# continuous torque ~40% of peak (thermal), rated speed decreasing with size.
# These are stand-ins for a real vendor BoM -- swap in datasheet values when a
# concrete catalog is chosen. Peak torques span the trained GENE tau range
# (40-200 Nm) so any searched design maps onto a SKU.
@dataclass(frozen=True)
class MotorSKU:
  name: str
  tau_peak: float  # Nm, short-term peak torque (sizing bound used here)
  tau_cont: float  # Nm, thermal continuous torque
  omega_max: float  # rad/s, no-load / rated speed
  mass: float  # kg
  price: float = 0.0  # optional monetary cost (currency units)


DEFAULT_CATALOG: tuple[MotorSKU, ...] = (
  MotorSKU("QDD-40", tau_peak=40.0, tau_cont=16.0, omega_max=30.0, mass=0.60),
  MotorSKU("QDD-60", tau_peak=60.0, tau_cont=24.0, omega_max=26.0, mass=0.82),
  MotorSKU("QDD-90", tau_peak=90.0, tau_cont=36.0, omega_max=22.0, mass=1.11),
  MotorSKU("QDD-120", tau_peak=120.0, tau_cont=48.0, omega_max=20.0, mass=1.38),
  MotorSKU("QDD-160", tau_peak=160.0, tau_cont=64.0, omega_max=18.0, mass=1.72),
  MotorSKU("QDD-200", tau_peak=200.0, tau_cont=80.0, omega_max=16.0, mass=2.02),
)


# ---------------------------------------------------------------------------
# Cost model.
# ---------------------------------------------------------------------------
@dataclass
class MotorCostModel:
  """Maps a per-joint peak-torque vector to a hardware cost (mass, SKU count).

  Parameters
  ----------
  kind:
    ``"linear"``, ``"powerlaw"`` or ``"catalog"``.
  alpha:
    Power-law exponent (only used by ``powerlaw``). ~0.7-0.8 for BLDC/QDD.
  catalog:
    SKU list used by ``catalog`` (defaults to :data:`DEFAULT_CATALOG`).
  linear_density:
    kg per Nm for the ``linear`` model (default anchored to the 100 Nm unit so
    linear and power-law agree at the anchor point).
  """

  kind: str = "powerlaw"
  alpha: float = DEFAULT_ALPHA
  catalog: tuple[MotorSKU, ...] = DEFAULT_CATALOG
  linear_density: float = _MASS_ANCHOR_KG / _TAU_ANCHOR_NM

  def __post_init__(self) -> None:
    if self.kind not in ("linear", "powerlaw", "catalog"):
      raise ValueError(f"unknown cost model kind: {self.kind!r}")
    if self.kind == "catalog":
      # Ascending peak torque simplifies "cheapest SKU that covers demand".
      self.catalog = tuple(sorted(self.catalog, key=lambda s: s.tau_peak))

  # -- per-joint mass -------------------------------------------------------
  def joint_mass(self, tau: float) -> float:
    """Estimated mass (kg) of a single actuator sized for peak torque ``tau``."""
    if self.kind == "linear":
      return self.linear_density * max(tau, 0.0)
    if self.kind == "powerlaw":
      return powerlaw_k(self.alpha) * max(tau, 0.0) ** self.alpha
    return self._select_sku(tau).mass

  def _select_sku(self, tau: float) -> MotorSKU:
    """Cheapest (smallest) SKU whose peak torque covers ``tau``.

    Demands above the largest SKU snap to the largest SKU (the search bounds are
    clamped to the catalog range upstream, so this is only a safety net).
    """
    for sku in self.catalog:  # ascending tau_peak
      if sku.tau_peak >= tau:
        return sku
    return self.catalog[-1]

  # -- design-level aggregation --------------------------------------------
  def design_mass(self, tau_per_joint: np.ndarray) -> float:
    """Total actuator mass (kg) summed over every physical joint."""
    return float(sum(self.joint_mass(float(t)) for t in np.asarray(tau_per_joint)))

  def design_price(self, tau_per_joint: np.ndarray) -> float:
    """Total monetary price (catalog only; 0 for continuous models)."""
    if self.kind != "catalog":
      return 0.0
    return float(sum(self._select_sku(float(t)).price for t in tau_per_joint))

  def sku_names(self, tau_per_joint: np.ndarray) -> list[str]:
    """SKU chosen per joint (catalog only)."""
    if self.kind != "catalog":
      return []
    return [self._select_sku(float(t)).name for t in tau_per_joint]

  def n_distinct_skus(self, tau_per_joint: np.ndarray) -> int:
    """Number of *distinct* SKUs in the bill-of-materials (catalog only).

    For continuous models this is undefined here (use the clustering-based
    ``count_motor_types`` in ``codesign_ga.py`` instead), so we return 0.
    """
    if self.kind != "catalog":
      return 0
    return len(set(self.sku_names(tau_per_joint)))
