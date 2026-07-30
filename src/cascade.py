"""
Cascading failure engine.

The physical story: a flood knocks out some substations directly. Power that
used to flow through those substations has to go somewhere else. Some of the
remaining lines are now carrying more than they are rated for, so their
protection trips them. That removes more paths, which pushes flow onto yet
other lines, and so on. Separately, any part of the grid that ends up with no
electrical path back to a generating source simply goes dark.

The loop below is the standard rule-based DC cascade model:

    1. Take substations out of service (flood damage, round 0).
    2. Anything now disconnected from the source is de-energised.
    3. Solve a DC power flow on what is left.
    4. Trip every branch loaded above its rating.
    5. If anything tripped, go back to step 2.

Simplifications, stated plainly
-------------------------------
* DC power flow: real power only, no voltage or reactive power, no losses.
  Standard for contingency screening, and the usual choice for cascade studies.
* A single external grid acts as the reference. Any island without a path back
  to it is treated as lost rather than islanded and re-dispatched. This
  overstates load shed relative to a system with black-start and islanding
  capability, and understates the difficulty of the surviving island's
  frequency control. It is a simplification, not a claim.
* Line ratings are not published with the IEEE 118-bus case (the shipped
  `max_i_ka` values are placeholders), so ratings are derived from the intact
  base case: rating = max(headroom x |base case flow|, floor). This is a common
  convention for test systems and the headroom factor is the single most
  important knob in the model - see `sensitivity` in the README.
* Protection is deterministic and instantaneous. No relay timing, no operator
  action, no load shedding scheme.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandapower as pp
import pandapower.topology as top

from network import Grid


@dataclass
class CascadeResult:
    final_failed: np.ndarray  # bool (n_bus,) de-energised at the end
    initial_failed: np.ndarray  # bool (n_bus,) direct flood damage only
    tripped_lines: np.ndarray  # bool (n_line,)
    tripped_trafos: np.ndarray  # bool (n_trafo,)
    load_shed_mw: float
    load_shed_frac: float
    rounds: int
    diverged: bool = False
    history: list = field(default_factory=list)  # failed-count after each round


def branch_ratings(grid: Grid, headroom: float = 2.5, floor_mw: float = 40.0):
    """
    Derive thermal ratings from the intact base case.

    Every branch is given a rating equal to `headroom` times the power it
    carries when nothing has failed, with a floor so that lightly-loaded
    branches are not absurdly fragile. Lower headroom = a more brittle system.
    """
    net = copy.deepcopy(grid.net)
    pp.rundcpp(net)
    line = np.maximum(np.abs(net.res_line.p_from_mw.to_numpy()) * headroom, floor_mw)
    trafo = np.maximum(np.abs(net.res_trafo.p_hv_mw.to_numpy()) * headroom, floor_mw)
    return line, trafo


def _deenergise(net, dead: np.ndarray, bus_ids: np.ndarray) -> None:
    """Take a set of buses, and everything attached to them, out of service."""
    dead_ids = set(bus_ids[dead].tolist())
    net.bus.loc[net.bus.index.isin(dead_ids), "in_service"] = False
    for table, col in (("load", "bus"), ("gen", "bus"), ("sgen", "bus"), ("shunt", "bus")):
        if table in net and len(net[table]):
            net[table].loc[net[table][col].isin(dead_ids), "in_service"] = False
    # A branch with a dead end is itself dead.
    if len(net.line):
        bad = net.line.from_bus.isin(dead_ids) | net.line.to_bus.isin(dead_ids)
        net.line.loc[bad, "in_service"] = False
    if len(net.trafo):
        bad = net.trafo.hv_bus.isin(dead_ids) | net.trafo.lv_bus.isin(dead_ids)
        net.trafo.loc[bad, "in_service"] = False


def simulate_cascade(
    grid: Grid,
    initial_failed: np.ndarray,
    ratings: tuple[np.ndarray, np.ndarray] | None = None,
    max_rounds: int = 25,
) -> CascadeResult:
    """Run the cascade to a fixed point and report what is left standing."""
    net = copy.deepcopy(grid.net)
    bus_ids = grid.net.bus.index.to_numpy()
    n_bus = len(bus_ids)
    pos = {int(b): i for i, b in enumerate(bus_ids)}

    line_rating, trafo_rating = ratings if ratings is not None else branch_ratings(grid)

    dead = initial_failed.copy()
    tripped_line = np.zeros(len(net.line), dtype=bool)
    tripped_trafo = np.zeros(len(net.trafo), dtype=bool)
    history = []
    diverged = False
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        # --- apply current damage state --------------------------------
        net.bus["in_service"] = True
        for t in ("load", "gen", "sgen", "shunt"):
            if t in net and len(net[t]):
                net[t]["in_service"] = True
        net.line["in_service"] = ~tripped_line
        net.trafo["in_service"] = ~tripped_trafo
        _deenergise(net, dead, bus_ids)

        # --- connectivity: anything cut off from the source is dark -----
        try:
            unsupplied = top.unsupplied_buses(net)
        except Exception:
            unsupplied = set()
        if unsupplied:
            newly = np.zeros(n_bus, dtype=bool)
            for b in unsupplied:
                if int(b) in pos:
                    newly[pos[int(b)]] = True
            if (newly & ~dead).any():
                dead |= newly
                _deenergise(net, dead, bus_ids)

        history.append(int(dead.sum()))

        if dead.all():
            break

        # --- DC power flow on the surviving, connected system -----------
        try:
            pp.rundcpp(net)
        except Exception:
            diverged = True
            break

        # --- overload check --------------------------------------------
        flow_l = np.abs(net.res_line.p_from_mw.to_numpy())
        flow_t = np.abs(net.res_trafo.p_hv_mw.to_numpy())
        flow_l = np.nan_to_num(flow_l)
        flow_t = np.nan_to_num(flow_t)

        over_l = (flow_l > line_rating) & net.line.in_service.to_numpy() & ~tripped_line
        over_t = (flow_t > trafo_rating) & net.trafo.in_service.to_numpy() & ~tripped_trafo

        if not over_l.any() and not over_t.any():
            break  # fixed point reached

        tripped_line |= over_l
        tripped_trafo |= over_t

    lost_mw = float(grid.bus_load_mw[dead].sum())
    total_mw = float(grid.bus_load_mw.sum())

    return CascadeResult(
        final_failed=dead,
        initial_failed=initial_failed.copy(),
        tripped_lines=tripped_line,
        tripped_trafos=tripped_trafo,
        load_shed_mw=lost_mw,
        load_shed_frac=lost_mw / total_mw if total_mw else 0.0,
        rounds=rounds,
        diverged=diverged,
        history=history,
    )


if __name__ == "__main__":
    from network import build_grid

    grid = build_grid()
    ratings = branch_ratings(grid)
    rng = np.random.default_rng(0)

    # Knock out the three highest-load buses and watch what happens.
    seed_buses = np.argsort(-grid.bus_load_mw)[:3]
    init = np.zeros(len(grid.net.bus), dtype=bool)
    init[seed_buses] = True

    r = simulate_cascade(grid, init, ratings)
    print(f"initial failures : {r.initial_failed.sum()}")
    print(f"final  failures  : {r.final_failed.sum()}")
    print(f"branches tripped : {r.tripped_lines.sum()} lines, {r.tripped_trafos.sum()} trafos")
    print(f"load shed        : {r.load_shed_mw:.0f} MW ({r.load_shed_frac:.1%})")
    print(f"rounds           : {r.rounds}  history={r.history}")
