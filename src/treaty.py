"""
Pricing an excess-of-loss reinsurance layer on the modelled loss distribution.

`loss_module.py` turns the physical cascade into an event loss table: one
ground-up insured loss per event. This module does what a reinsurance pricing
actuary does with such a table, apply a per-occurrence excess-of-loss (XL)
structure and price it.

    event loss table  ->  ceded loss per event (attachment, limit)
        ->  ceded AAL, layer statistics
        ->  technical premium (target loss ratio, and RAROC risk load)

The structure and the pricing are the same objects used to price CAT XL treaties
on a vendor-model event loss table; only the source of the loss table differs.

Pricing bases provided
----------------------
* Burning-cost / target loss ratio: premium = ceded AAL / target loss ratio.
  The simplest market benchmark.
* RAROC / capital-based: premium = ceded AAL + expense + risk load, where the
  risk load is the cost of the capital the layer consumes. Capital is proxied
  by (layer PML at a chosen return period minus ceded AAL), and charged at a
  target return on capital. This mirrors RAROC-based technical pricing.

Assumptions and limitations
---------------------------
* Per-occurrence only. No reinstatements, no aggregate deductible or limit, no
  annual aggregation (each event is treated as its own occurrence).
* Capital is a single-return-period proxy, not a full economic-capital model.
* Expense and target ratios are illustrative inputs, not market quotes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class LayerTerms:
    attachment: float          # retention below which the layer pays nothing
    limit: float               # width of the layer

    @property
    def exhaustion(self) -> float:
        return self.attachment + self.limit


@dataclass
class PricingParams:
    target_loss_ratio: float = 0.60     # for the burning-cost premium
    expense_ratio: float = 0.10         # brokerage + internal expense, of premium
    capital_return_period: int = 250    # return period defining required capital
    target_roc: float = 0.15            # required return on allocated capital


def cede(event_loss: np.ndarray, terms: LayerTerms) -> np.ndarray:
    """Ceded loss to the layer, per event."""
    return np.clip(event_loss - terms.attachment, 0.0, terms.limit)


def layer_statistics(event_loss: np.ndarray, terms: LayerTerms) -> dict:
    ceded = cede(event_loss, terms)
    into_layer = ceded > 0
    full_limit = ceded >= terms.limit - 1e-9
    ceded_aal = float(ceded.mean())
    ceded_sd = float(ceded.std())
    return {
        "attachment": terms.attachment,
        "limit": terms.limit,
        "exhaustion": terms.exhaustion,
        "ceded_aal": ceded_aal,
        "ceded_sd": ceded_sd,
        "ceded_cov": float(ceded_sd / ceded_aal) if ceded_aal > 0 else float("nan"),
        "prob_attach": float(into_layer.mean()),
        "prob_exhaust": float(full_limit.mean()),
        "loss_on_line": float(ceded_aal / terms.limit) if terms.limit > 0 else float("nan"),
        "ceded_max": float(ceded.max()),
    }


def _pml(event_loss: np.ndarray, return_period: int) -> float:
    losses = np.sort(event_loss)[::-1]
    n = len(losses)
    exceed = np.arange(1, n + 1) / (n + 1)
    idx = min(np.searchsorted(exceed, 1.0 / return_period), n - 1)
    return float(losses[idx])


def price_layer(event_loss: np.ndarray, terms: LayerTerms, pp: PricingParams) -> dict:
    stats = layer_statistics(event_loss, terms)
    ceded = cede(event_loss, terms)
    ceded_aal = stats["ceded_aal"]

    # Burning-cost / target-loss-ratio premium.
    tlr_premium = ceded_aal / pp.target_loss_ratio if pp.target_loss_ratio > 0 else float("nan")

    # RAROC / capital-based premium.
    layer_pml = min(_pml(ceded, pp.capital_return_period), terms.limit)
    capital = max(layer_pml - ceded_aal, 0.0)
    risk_load = capital * pp.target_roc
    raroc_premium = (ceded_aal + risk_load) / (1.0 - pp.expense_ratio)

    return {
        **stats,
        "target_loss_ratio": pp.target_loss_ratio,
        "premium_target_lr": float(tlr_premium),
        "capital_return_period": pp.capital_return_period,
        "allocated_capital": float(capital),
        "risk_load": float(risk_load),
        "premium_raroc": float(raroc_premium),
        "rate_on_line_raroc": float(raroc_premium / terms.limit) if terms.limit > 0 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Price an XL layer on the modelled loss table.")
    ap.add_argument("--event-losses", type=Path, default=Path("artifacts/event_losses.npy"))
    ap.add_argument("--attachment", type=float, required=True)
    ap.add_argument("--limit", type=float, required=True)
    ap.add_argument("--target-loss-ratio", type=float, default=PricingParams.target_loss_ratio)
    ap.add_argument("--target-roc", type=float, default=PricingParams.target_roc)
    ap.add_argument("--out", type=Path, default=Path("artifacts/treaty_pricing.json"))
    args = ap.parse_args()

    event_loss = np.load(args.event_losses)
    terms = LayerTerms(attachment=args.attachment, limit=args.limit)
    pp = PricingParams(target_loss_ratio=args.target_loss_ratio, target_roc=args.target_roc)

    pricing = price_layer(event_loss, terms, pp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out).write_text(json.dumps({"terms": asdict(terms), "pricing": pricing}, indent=2))

    label = f"{terms.limit:,.0f} xs {terms.attachment:,.0f}"
    print(f"XL layer  {label}")
    for k, v in pricing.items():
        if isinstance(v, float) and abs(v) >= 1000:
            print(f"  {k:24s} {v:,.0f}")
        else:
            print(f"  {k:24s} {v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
