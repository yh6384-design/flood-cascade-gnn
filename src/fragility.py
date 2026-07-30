"""
Fragility: turning flood depth into substation failure.

A fragility curve gives the probability that a component reaches or exceeds a
damage state, conditional on the intensity of the hazard at its location. The
standard functional form in earthquake and flood engineering (and the one used
in HAZUS) is lognormal:

    P(fail | depth d) = Phi( ln(d / theta) / beta )

    theta = median capacity  (the depth at which failure is 50% likely)
    beta  = logarithmic standard deviation (how uncertain that capacity is)

This is the same object as a vulnerability curve in a catastrophe model: the
function that converts hazard intensity into damage. The difference is only
the output - damage ratio there, failure probability here.

Median capacities below are illustrative, chosen so that higher-voltage
substations are assumed to be better protected (larger sites, more elevated
equipment pads). They are placeholders for real asset data, not published
values, and the README says so.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def median_capacity_m(vn_kv: np.ndarray) -> np.ndarray:
    """Median inundation depth a substation survives, by voltage class."""
    theta = np.full_like(vn_kv, 1.2, dtype=float)  # default / distribution level
    theta = np.where(vn_kv >= 100.0, 1.6, theta)  # sub-transmission
    theta = np.where(vn_kv >= 300.0, 2.2, theta)  # bulk transmission
    return theta


def failure_probability(depth_m: np.ndarray, vn_kv: np.ndarray, beta: float = 0.45) -> np.ndarray:
    """
    Lognormal fragility. Dry sites (depth <= 0) have probability exactly zero;
    the curve is only defined for positive intensity.
    """
    theta = median_capacity_m(vn_kv)
    p = np.zeros_like(depth_m, dtype=float)
    wet = depth_m > 0
    if wet.any():
        p[wet] = norm.cdf(np.log(depth_m[wet] / theta[wet]) / beta)
    return p


def sample_failures(
    depth_m: np.ndarray, vn_kv: np.ndarray, rng: np.random.Generator, beta: float = 0.45
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw the initial (direct) substation failures for one event.

    Returns (boolean failure mask, failure probability per bus). The
    probabilities are kept because they are a useful model input: they tell the
    GNN how close a surviving neighbour was to failing.
    """
    p = failure_probability(depth_m, vn_kv, beta=beta)
    return rng.random(len(p)) < p, p


if __name__ == "__main__":
    d = np.array([0.0, 0.5, 1.0, 1.6, 2.5, 4.0, 8.0])
    for kv in (138.0, 345.0):
        p = failure_probability(d, np.full_like(d, kv))
        print(f"{kv:.0f} kV:", " ".join(f"{x:.0f}m->{y:.2f}" for x, y in zip(d, p)))
