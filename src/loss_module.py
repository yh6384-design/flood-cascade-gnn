"""
From de-energised substations to an insured loss distribution.

The simulator (`simulate.py`) answers a physical question: which substations go
dark in each flood event. This module answers the insurance question sitting on
top of it: what does that outage cost an insured, and what does the
distribution of that cost look like across the whole event set.

The peril modelled here is business interruption (BI) from loss of power, the
exposure a conventional flood CAT model does not see because it prices direct
property damage, not network-propagated outage.

The chain
---------
    de-energised bus  ->  daily BI value at risk  ->  outage duration
        (from Y)          (exposure assumption)      (restoration model)
        ->  indemnified loss (waiting period, max indemnity period)
        ->  event loss  ->  event loss table  ->  OEP curve, AAL, PML

Everything downstream of `event_losses` is standard catastrophe-model output,
just denominated in dollars instead of megawatts.

Insurance mechanics represented
-------------------------------
* Exposure: each bus carries a daily BI value proportional to its served load
  (`value_per_mw_day`). A bus only contributes loss in events where it is
  de-energised.
* Outage duration: a per-event restoration time, lognormal, with a median that
  scales with how much of the system went down (a bigger cascade takes longer
  to restore). Stated as an assumption, not a calibrated figure.
* Waiting period: a time deductible. Outage shorter than this indemnifies zero.
* Maximum indemnity period: the policy caps how many days of BI it will pay.

Assumptions and limitations
---------------------------
* Exposure is synthetic and load-proportional. No real insured-value schedule
  is used. `value_per_mw_day` is an illustrative rate, tune it to a book.
* Duration is modelled at the event level, not per bus. A real study would let
  restoration time vary by asset and by how deep in the cascade a bus sits.
* One deterministic `value_per_mw_day` and one duration model; no correlation
  structure beyond what the physical cascade already induces.

The point of the module is the pipeline and the insurance framing, not the
absolute dollar figures, which are only as good as the exposure assumptions
fed in.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class ExposureParams:
    value_per_mw_day: float = 120_000.0  # daily BI value at risk per MW of served load
    # Interpretation: a bus serving 100 MW carries $12.0m/day of BI exposure.
    # Illustrative; replace with a real value-per-MW derived from the book.


@dataclass
class DurationParams:
    median_days_min: float = 1.0    # restoration time for the smallest cascade
    median_days_max: float = 21.0   # restoration time when ~all load is down
    sigma_ln: float = 0.5           # lognormal shape (spread of restoration time)


@dataclass
class PolicyTerms:
    waiting_period_days: float = 1.0        # time deductible
    max_indemnity_days: float = 30.0        # BI cap


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def bus_daily_value(bus_load_mw: np.ndarray, exp: ExposureParams) -> np.ndarray:
    """Per-bus daily BI value at risk (n_bus,)."""
    return bus_load_mw.astype(float) * exp.value_per_mw_day


def event_durations(
    outage_frac: np.ndarray, rng: np.random.Generator, dur: DurationParams
) -> np.ndarray:
    """
    Draw a restoration time (days) for each event.

    The lognormal median is interpolated between `median_days_min` and
    `median_days_max` by the fraction of system load that ended up down, so a
    near-total blackout takes materially longer to restore than a local outage.
    """
    med = dur.median_days_min + (dur.median_days_max - dur.median_days_min) * np.clip(
        outage_frac, 0.0, 1.0
    )
    # lognormal with the given median (median = exp(mu)) and shape sigma_ln
    mu = np.log(np.maximum(med, 1e-6))
    return rng.lognormal(mean=mu, sigma=dur.sigma_ln)


def indemnified_days(duration_days: np.ndarray, pol: PolicyTerms) -> np.ndarray:
    """Apply waiting period and max indemnity period to a duration."""
    payable = np.clip(duration_days - pol.waiting_period_days, 0.0, None)
    return np.minimum(payable, pol.max_indemnity_days)


def event_losses(
    energised_state,
    bus_load_mw: np.ndarray,
    durations_days: np.ndarray,
    exp: ExposureParams,
    pol: PolicyTerms,
) -> np.ndarray:
    """
    Ground-up insured BI loss per event (n_events,).

    `energised_state` is (n_events, n_bus). It can be either:
      * hard 0/1 labels (the simulator's Y, or a surrogate's thresholded
        prediction), or
      * probabilities in [0, 1] (a surrogate's raw output), in which case the
        loss is the expected loss under those probabilities.
    Passing probabilities is what lets the trained GNN price the book directly,
    without re-running the physical cascade for every scenario.
    """
    state = np.asarray(energised_state, dtype=float)          # (E, B)
    daily = bus_daily_value(bus_load_mw, exp)                 # (B,)
    ind_days = indemnified_days(durations_days, pol)          # (E,)
    # loss = sum_b [ down_b * daily_value_b ] * indemnified_days_event
    per_event_daily_value = state @ daily                     # (E,)
    return per_event_daily_value * ind_days


# ---------------------------------------------------------------------------
# Catastrophe-model summaries
# ---------------------------------------------------------------------------

def oep_curve(event_loss: np.ndarray):
    """
    Occurrence exceedance probability curve.

    Returns (loss_sorted_desc, exceedance_prob). With one modelled occurrence
    per event, the annual OEP and the per-event exceedance coincide here; a
    multi-event-year model would group by year first.
    """
    losses = np.sort(event_loss)[::-1]
    n = len(losses)
    exceed = (np.arange(1, n + 1)) / (n + 1)
    return losses, exceed


def pml_at(event_loss: np.ndarray, return_periods=(100, 250)) -> dict:
    """PML (loss) at the given return periods, read off the OEP curve."""
    losses, exceed = oep_curve(event_loss)
    out = {}
    for rp in return_periods:
        target = 1.0 / rp
        # first loss whose exceedance probability is <= target
        idx = np.searchsorted(exceed, target)
        idx = min(idx, len(losses) - 1)
        out[f"pml_{rp}yr"] = float(losses[idx])
    return out


def loss_metrics(event_loss: np.ndarray, return_periods=(100, 250)) -> dict:
    aal = float(event_loss.mean())
    sd = float(event_loss.std())
    m = {
        "events": int(len(event_loss)),
        "aal": aal,
        "loss_sd": sd,
        "loss_cov": float(sd / aal) if aal > 0 else float("nan"),
        "mean_loss_given_event": float(event_loss[event_loss > 0].mean())
        if (event_loss > 0).any()
        else 0.0,
        "prob_nonzero": float((event_loss > 0).mean()),
    }
    m.update(pml_at(event_loss, return_periods))
    return m


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def losses_from_scenarios(
    npz_path: Path,
    exp: ExposureParams,
    dur: DurationParams,
    pol: PolicyTerms,
    seed: int = 7,
    use_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """
    Load a scenario set and return the per-event insured loss array.

    If `use_probabilities` is given (n_events, n_bus), losses are computed from
    it (the surrogate path). Otherwise the simulator's hard Y labels are used.
    """
    d = np.load(npz_path, allow_pickle=True)
    Y = d["Y"]
    bus_load_mw = d["bus_load_mw"]
    load_shed_mw = d["meta"][:, list(d["meta_cols"]).index("load_shed_mw")]
    outage_frac = load_shed_mw / bus_load_mw.sum()

    rng = np.random.default_rng(seed)
    durations = event_durations(outage_frac, rng, dur)

    state = use_probabilities if use_probabilities is not None else Y
    return event_losses(state, bus_load_mw, durations, exp, pol)


def main() -> None:
    ap = argparse.ArgumentParser(description="Insured BI loss from the flood-cascade scenario set.")
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios.npz"))
    ap.add_argument("--value-per-mw-day", type=float, default=ExposureParams.value_per_mw_day)
    ap.add_argument("--waiting-days", type=float, default=PolicyTerms.waiting_period_days)
    ap.add_argument("--max-indemnity-days", type=float, default=PolicyTerms.max_indemnity_days)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("artifacts/event_losses.npy"))
    args = ap.parse_args()

    exp = ExposureParams(value_per_mw_day=args.value_per_mw_day)
    dur = DurationParams()
    pol = PolicyTerms(waiting_period_days=args.waiting_days, max_indemnity_days=args.max_indemnity_days)

    event_loss = losses_from_scenarios(args.scenarios, exp, dur, pol, seed=args.seed)
    metrics = loss_metrics(event_loss)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, event_loss)
    summary = {"assumptions": {**asdict(exp), **asdict(dur), **asdict(pol)}, "metrics": metrics}
    (args.out.parent / "loss_summary.json").write_text(json.dumps(summary, indent=2))

    print("insured BI loss summary  (illustrative exposure)")
    for k, v in metrics.items():
        if isinstance(v, float) and abs(v) >= 1000:
            print(f"  {k:24s} {v:,.0f}")
        else:
            print(f"  {k:24s} {v}")
    print(f"\nwrote {args.out} and loss_summary.json")


if __name__ == "__main__":
    main()
