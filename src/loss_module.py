"""
An insurance reading of the simulation: load shed to economic loss.

The cascade model already produces the object a catastrophe model produces: an
exceedance curve over outcomes, with the tail made explicit. The only thing
between that and a catastrophe *loss* curve is the choice of unit. This module
takes the last small step, valuing the energy not served in each event at a
value-of-lost-load rate:

    load shed (MW)  x  outage duration (h)   =  energy not served (MWh)
    energy not served  x  VOLL ($/MWh)       =  event loss ($)

That is the whole model. One economic assumption (VOLL) and a restoration-time
model, both stated here. The absolute dollars scale with VOLL; the shape of the
curve is the cascade's, unchanged. It exists so the technical result above can
be read as, and priced like, a catastrophe model, not to be a pricing tool in
its own right.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1F3864"
RED = "#B23A48"
BLUE = "#4C72B0"


@dataclass
class Assumptions:
    voll_per_mwh: float = 2_500.0     # value of lost load; illustrative
    restore_min_h: float = 3.0        # restoration time for the smallest cascade
    restore_max_h: float = 96.0       # restoration time when ~all load is down
    sigma_ln: float = 0.5             # lognormal spread of restoration time


def outage_duration_h(outage_frac, rng, a: Assumptions):
    """Per-event restoration time (hours), lognormal, longer for bigger cascades."""
    med = a.restore_min_h + (a.restore_max_h - a.restore_min_h) * np.clip(outage_frac, 0, 1)
    return rng.lognormal(mean=np.log(np.maximum(med, 1e-6)), sigma=a.sigma_ln)


def event_losses(load_shed_mw, duration_h, a: Assumptions):
    """Economic loss per event: energy not served, valued at VOLL."""
    energy_not_served_mwh = np.asarray(load_shed_mw, float) * duration_h
    return energy_not_served_mwh * a.voll_per_mwh


def ep_curve(losses):
    s = np.sort(losses)[::-1]
    return s, np.arange(1, len(s) + 1) / (len(s) + 1)


def pml(losses, return_period):
    s, ep = ep_curve(losses)
    return float(s[min(np.searchsorted(ep, 1.0 / return_period), len(s) - 1)])


def metrics(losses):
    return {
        "aal": float(losses.mean()),
        "pml_100yr": pml(losses, 100),
        "pml_250yr": pml(losses, 250),
        "cov": float(losses.std() / losses.mean()) if losses.mean() else float("nan"),
    }


def price_layer(losses, attachment, limit, target_lr=0.60):
    """One illustrative excess-of-loss layer, priced on a burning-cost basis."""
    ceded = np.clip(losses - attachment, 0.0, limit)
    ceded_aal = float(ceded.mean())
    return {
        "attachment": attachment,
        "limit": limit,
        "prob_attach": float((ceded > 0).mean()),
        "ceded_aal": ceded_aal,
        "premium_60lr": ceded_aal / target_lr,
    }


def losses_from_scenarios(npz_path, a: Assumptions, seed=7, load_shed_override=None):
    """
    Per-event economic loss from a scenario set.

    `load_shed_override` lets the trained surrogate stand in for the simulator:
    pass its predicted load-shed-per-event and the same loss curve comes out with
    no cascade re-run.
    """
    d = np.load(npz_path, allow_pickle=True)
    total = d["bus_load_mw"].sum()
    load_shed = load_shed_override if load_shed_override is not None \
        else d["meta"][:, list(d["meta_cols"]).index("load_shed_mw")]
    rng = np.random.default_rng(seed)
    dur = outage_duration_h(load_shed / total, rng, a)
    return event_losses(load_shed, dur, a)


def fig_loss_curve(losses, layer, out: Path):
    s, ep = ep_curve(losses)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(s / 1e6, ep, color=NAVY, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("event loss ($m)")
    ax.set_ylabel("P(exceedance) per event")
    ax.set_title("The exceedance curve, priced")
    ax.grid(alpha=0.3, which="both")
    if layer is not None:
        ax.axvspan(layer["attachment"] / 1e6, (layer["attachment"] + layer["limit"]) / 1e6,
                   color=BLUE, alpha=0.14,
                   label="XL layer \\${:.0f}m xs \\${:.0f}m".format(layer["limit"]/1e6, layer["attachment"]/1e6))
    for rp, style in ((100, "--"), (250, "-.")):
        v = pml(losses, rp)
        ax.axvline(v / 1e6, color=RED, ls=style, lw=1.1, label="1-in-{}: \\${:,.0f}m".format(rp, v/1e6))
    ax.legend(fontsize=8)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Economic-loss reading of the flood-cascade scenarios.")
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios.npz"))
    ap.add_argument("--voll", type=float, default=Assumptions.voll_per_mwh)
    ap.add_argument("--attachment", type=float, default=400e6)
    ap.add_argument("--limit", type=float, default=400e6)
    ap.add_argument("--fig", type=Path, default=Path("figures/06_loss_curve.png"))
    args = ap.parse_args()

    a = Assumptions(voll_per_mwh=args.voll)
    losses = losses_from_scenarios(args.scenarios, a)
    m = metrics(losses)
    layer = price_layer(losses, args.attachment, args.limit)
    fig_loss_curve(losses, layer, args.fig)

    print("economic loss  (illustrative VOLL = ${:,.0f}/MWh)".format(a.voll_per_mwh))
    for k, v in m.items():
        print("  {:12s} {:,.0f}".format(k, v) if abs(v) >= 100 else "  {:12s} {:.2f}".format(k, v))
    print("\nlayer ${:.0f}m xs ${:.0f}m:  attaches {:.0f}% of events,  ceded AAL ${:.0f}m,  premium ${:.0f}m at 60% LR".format(
        layer["limit"]/1e6, layer["attachment"]/1e6, layer["prob_attach"]*100,
        layer["ceded_aal"]/1e6, layer["premium_60lr"]/1e6))
    print("wrote", args.fig)


if __name__ == "__main__":
    main()
