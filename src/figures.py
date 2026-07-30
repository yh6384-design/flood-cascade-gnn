"""
Figures. Five of them, each answering one question.

  1 network_map        where is the grid, where is the river, what floods
  2 cascade_example    what a single bad event does, round by round
  3 exceedance_curve   how bad can it get, and how often
  4 model_comparison   does the graph help
  5 criticality        which substations matter most

Figure 3 is an exceedance probability curve: for each level of load shed, the
annual-equivalent probability of exceeding it. This is the standard output of
a catastrophe model (the EP curve an insurer prices off) applied here to
megawatts lost instead of dollars lost. It is the natural way to summarise a
simulation-driven risk model, and it is what turns "the grid might fail" into
a number a planner can act on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

NAVY = "#1F3864"
RED = "#B23A48"
BLUE = "#4C72B0"
GREY = "#999999"


def _edges_xy(xy, ei):
    seen, segs = set(), []
    for a, b in zip(ei[0], ei[1]):
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        segs.append([xy[a], xy[b]])
    return segs


def fig_network_map(z, out: Path):
    xy, ei = z["xy_km"], z["edge_index"]
    elev, river = z["elevation_m"], z["river_xy_km"]
    depth = z["X"][:, :, 0]
    ever_wet = (depth > 0).mean(axis=0)  # fraction of events inundating each bus

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax in axes:
        ax.add_collection(LineCollection(_edges_xy(xy, ei), colors=GREY, linewidths=0.7, alpha=0.6))
        ax.plot(river[:, 0], river[:, 1], color=BLUE, lw=2.4, alpha=0.85, zorder=1, label="river")
        ax.set_xlabel("km east")
        ax.set_aspect("equal")

    s0 = axes[0].scatter(xy[:, 0], xy[:, 1], c=elev, cmap="terrain", s=46,
                         edgecolors="k", linewidths=0.4, zorder=3)
    fig.colorbar(s0, ax=axes[0], label="ground elevation (m)")
    axes[0].set_title("IEEE 118-bus, georeferenced onto the study domain")
    axes[0].set_ylabel("km north")

    s1 = axes[1].scatter(xy[:, 0], xy[:, 1], c=ever_wet, cmap="Blues", s=46,
                         vmin=0, edgecolors="k", linewidths=0.4, zorder=3)
    fig.colorbar(s1, ax=axes[1], label="fraction of events inundated")
    axes[1].set_title("Flood exposure across the event set")
    axes[0].legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_cascade_example(z, out: Path):
    xy, ei, river = z["xy_km"], z["edge_index"], z["river_xy_km"]
    meta, X, Y = z["meta"], z["X"], z["Y"]
    # Pick an illustrative event: the largest cascade that is NOT a total
    # blackout, so the figure shows a propagation front rather than a dark map.
    n_bus = Y.shape[1]
    gap = meta[:, 2] - meta[:, 1]
    partial = meta[:, 2] < 0.75 * n_bus
    e = int(np.argmax(np.where(partial, gap, -1)))

    init = X[e, :, 2] > 0.5
    final = Y[e] > 0
    secondary = final & ~init

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.add_collection(LineCollection(_edges_xy(xy, ei), colors=GREY, linewidths=0.7, alpha=0.55))
    ax.plot(river[:, 0], river[:, 1], color=BLUE, lw=2.4, alpha=0.8)

    ax.scatter(xy[~final, 0], xy[~final, 1], c="white", edgecolors="k",
               s=42, linewidths=0.5, label="energised", zorder=3)
    ax.scatter(xy[secondary, 0], xy[secondary, 1], c="#E8A33D", edgecolors="k",
               s=64, linewidths=0.5, label="secondary (cascade)", zorder=4)
    ax.scatter(xy[init, 0], xy[init, 1], c=RED, edgecolors="k", marker="s",
               s=74, linewidths=0.5, label="direct flood damage", zorder=5)

    shed = meta[e, 3] / z["bus_load_mw"].sum()
    ax.set_title(f"Event {e}: {int(meta[e, 1])} substations flooded  ->  "
                 f"{int(meta[e, 2])} de-energised ({shed:.0%} of load)")
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_exceedance(z, out: Path):
    meta = z["meta"]
    total = z["bus_load_mw"].sum()
    shed = np.sort(meta[:, 3])[::-1]
    n = len(shed)
    # Exceedance probability per event; multiply by events/year for an annual EP.
    ep = np.arange(1, n + 1) / n

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    axes[0].plot(shed, ep, color=NAVY, lw=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("load shed (MW)")
    axes[0].set_ylabel("P(exceedance) per event")
    axes[0].set_title("Exceedance probability curve")
    axes[0].grid(alpha=0.3, which="both")
    for q, style in ((0.5, ":"), (0.1, "--"), (0.01, "-.")):
        v = np.quantile(meta[:, 3], 1 - q)
        axes[0].axvline(v, color=RED, ls=style, lw=1,
                        label=f"{int(q * 100)}% event: {v:.0f} MW ({v / total:.0%})")
    axes[0].legend(fontsize=8)

    axes[1].scatter(meta[:, 1], meta[:, 2], s=14, alpha=0.45, color=NAVY,
                    edgecolors="none")
    lim = max(meta[:, 2].max(), meta[:, 1].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], color=GREY, ls="--", lw=1,
                 label="no cascade (y = x)")
    axes[1].set_xlabel("substations flooded directly")
    axes[1].set_ylabel("substations de-energised in total")
    axes[1].set_title("Cascade amplification")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_model_comparison(pred_path: Path, out: Path):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    p = np.load(pred_path)
    y, init = p["y"], p["initial"]
    sec = ~init

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, mask, title in (
        (axes[0], np.ones_like(y, dtype=bool), "All substations"),
        (axes[1], sec, "Substations NOT hit directly by the flood"),
    ):
        for name, colour in (("gnn", NAVY), ("mlp", RED)):
            pr, rc, _ = precision_recall_curve(y[mask], p[name][mask])
            ap = average_precision_score(y[mask], p[name][mask])
            ax.plot(rc, pr, color=colour, lw=2,
                    label=f"{'GraphSAGE' if name == 'gnn' else 'MLP (no graph)'}  AP={ap:.3f}")
        ax.axhline(y[mask].mean(), color=GREY, ls="--", lw=1,
                   label=f"base rate={y[mask].mean():.3f}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")

    fig.suptitle("Predicting which substations go dark", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_criticality(z, out: Path, top: int = 20):
    Y, X = z["Y"], z["X"]
    init = X[:, :, 2] > 0.5
    p_final = Y.mean(axis=0)
    p_secondary = (Y.astype(bool) & ~init).mean(axis=0)
    p_direct = init.mean(axis=0)

    order = np.argsort(-p_final)[:top]
    idx = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.barh(idx, p_direct[order], color=RED, label="direct flood damage")
    ax.barh(idx, p_secondary[order], left=p_direct[order], color="#E8A33D",
            label="secondary (cascade)")
    ax.set_yticks(idx)
    ax.set_yticklabels([f"bus {b}" for b in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("probability of being de-energised, per event")
    ax.set_title(f"Top {top} substations by outage probability")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render all figures.")
    ap.add_argument("--data", type=Path, default=Path("data/scenarios.npz"))
    ap.add_argument("--preds", type=Path, default=Path("artifacts/test_predictions.npz"))
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.data, allow_pickle=True)

    fig_network_map(z, args.outdir / "01_network_map.png")
    fig_cascade_example(z, args.outdir / "02_cascade_example.png")
    fig_exceedance(z, args.outdir / "03_exceedance_curve.png")
    fig_criticality(z, args.outdir / "05_criticality.png")
    if args.preds.exists():
        fig_model_comparison(args.preds, args.outdir / "04_model_comparison.png")
    else:
        print(f"(skipping model comparison: {args.preds} not found)")

    for p in sorted(args.outdir.glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()
