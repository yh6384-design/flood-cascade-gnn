"""
Scenario generation: run the whole chain many times and save the dataset.

For each simulated flood event we record, for every substation, the inputs a
model could plausibly see before the cascade happens, and the outcome we want
to predict.

    features (per bus, per event)
      0 flood_depth_m        hazard intensity at the site
      1 fail_prob            fragility output at that depth
      2 initially_failed     direct flood damage (0/1)
      3 elevation_m          terrain
      4 dist_to_river_km     terrain
      5 vn_kv                voltage class
      6 load_mw              demand at the bus
      7 gen_mw               generation at the bus
      8 degree               topological
      9 betweenness          topological
     10 event_severity_m     one scalar broadcast to every node

    label (per bus, per event)
      de-energised at the end of the cascade (0/1)

The point of the exercise is the gap between features 0-2 (what the flood did
directly) and the label (what the grid did afterwards). A model that only
looks at a node in isolation can learn the first. Learning the second requires
knowing who the node's neighbours are, which is what the graph is for.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cascade import branch_ratings, simulate_cascade
from fragility import sample_failures
from hazard import HazardParams, river_projection, sample_event
from network import build_grid, edge_index

FEATURE_NAMES = [
    "flood_depth_m",
    "fail_prob",
    "initially_failed",
    "elevation_m",
    "dist_to_river_km",
    "vn_kv",
    "load_mw",
    "gen_mw",
    "degree",
    "betweenness",
    "event_severity_m",
]


def run(n_events: int, seed: int, headroom: float, out_path: Path) -> dict:
    grid = build_grid()
    proj = river_projection(grid)
    ratings = branch_ratings(grid, headroom=headroom)
    hp = HazardParams()
    rng = np.random.default_rng(seed)

    n_bus = len(grid.net.bus)
    X = np.zeros((n_events, n_bus, len(FEATURE_NAMES)), dtype=np.float32)
    Y = np.zeros((n_events, n_bus), dtype=np.int8)
    meta = np.zeros((n_events, 6), dtype=np.float32)  # see meta_cols below

    # Static per-bus features, identical in every event.
    static = np.column_stack(
        [
            grid.elevation_m,
            grid.dist_to_river_km,
            grid.vn_kv,
            grid.bus_load_mw,
            grid.bus_gen_mw,
            grid.degree,
            grid.betweenness,
        ]
    ).astype(np.float32)

    t0 = time.time()
    for e in range(n_events):
        ev = sample_event(grid, rng, hp, proj)
        failed, p_fail = sample_failures(ev["depth_m"], grid.vn_kv, rng)
        res = simulate_cascade(grid, failed, ratings)

        X[e, :, 0] = ev["depth_m"]
        X[e, :, 1] = p_fail
        X[e, :, 2] = failed
        X[e, :, 3:10] = static
        X[e, :, 10] = ev["severity_m"]
        Y[e] = res.final_failed.astype(np.int8)

        meta[e] = [
            ev["severity_m"],
            failed.sum(),
            res.final_failed.sum(),
            res.load_shed_mw,
            res.rounds,
            float(res.diverged),
        ]

        if (e + 1) % 200 == 0:
            rate = (e + 1) / (time.time() - t0)
            print(f"  {e + 1}/{n_events} events  ({rate:.0f}/s)", flush=True)

    ei = edge_index(grid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        Y=Y,
        meta=meta,
        edge_index=ei,
        feature_names=np.array(FEATURE_NAMES),
        meta_cols=np.array(
            ["severity_m", "n_initial", "n_final", "load_shed_mw", "rounds", "diverged"]
        ),
        lonlat=grid.lonlat,
        xy_km=grid.xy_km,
        elevation_m=grid.elevation_m,
        river_xy_km=grid.river_xy_km,
        bus_load_mw=grid.bus_load_mw,
        headroom=np.array([headroom]),
    )

    secondary = (Y == 1) & (X[:, :, 2] == 0)
    summary = {
        "events": n_events,
        "buses": n_bus,
        "mean_initial_failures": float(meta[:, 1].mean()),
        "mean_final_failures": float(meta[:, 2].mean()),
        "mean_load_shed_frac": float(meta[:, 3].mean() / grid.bus_load_mw.sum()),
        "p99_load_shed_frac": float(np.quantile(meta[:, 3], 0.99) / grid.bus_load_mw.sum()),
        "positive_rate": float(Y.mean()),
        "secondary_failure_rate": float(secondary.mean()),
        "diverged": int(meta[:, 5].sum()),
        "seconds": round(time.time() - t0, 1),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the flood-cascade scenario set.")
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--headroom", type=float, default=2.5,
                    help="branch rating = headroom x base-case flow")
    ap.add_argument("--out", type=Path, default=Path("data/scenarios.npz"))
    args = ap.parse_args()

    print(f"simulating {args.events} flood events (headroom={args.headroom})")
    s = run(args.events, args.seed, args.headroom, args.out)
    print("\nsummary")
    for k, v in s.items():
        print(f"  {k:24s} {v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
