"""
Power network construction and georeferencing.

The IEEE 118-bus case is a synthetic test system: it has a realistic electrical
topology but no real-world location. pandapower ships it with a schematic layout
(net.bus.geo) used for one-line drawings. We affine-map that layout onto a real
geographic bounding box so that a spatially-correlated hazard field can be
applied to it in a meaningful way.

This is a modelling assumption, not a claim about any real grid. See README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.networks as ppn

# Bounding box for the synthetic study region (a ~120 km x 110 km domain).
LON_MIN, LON_MAX = -75.00, -73.60
LAT_MIN, LAT_MAX = 40.40, 41.40

# Deg -> km conversion at ~41N, used to work in a local planar (km) frame.
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 84.0


@dataclass
class Grid:
    """Everything downstream needs to know about the network."""

    net: "pp.pandapowerNet"  # pandapower model (electrical)
    graph: nx.Graph  # NetworkX view (topological)
    xy_km: np.ndarray  # (n_bus, 2) planar coordinates in km
    lonlat: np.ndarray  # (n_bus, 2) lon/lat, for maps
    elevation_m: np.ndarray  # (n_bus,) ground elevation, metres
    dist_to_river_km: np.ndarray  # (n_bus,)
    bus_load_mw: np.ndarray  # (n_bus,) attached load
    bus_gen_mw: np.ndarray  # (n_bus,) attached generation capacity
    vn_kv: np.ndarray  # (n_bus,) nominal voltage
    degree: np.ndarray  # (n_bus,) node degree
    betweenness: np.ndarray  # (n_bus,) betweenness centrality
    river_xy_km: np.ndarray  # (n_pts, 2) river centreline
    slack_bus: int


def _layout_xy(net) -> np.ndarray:
    """Pull the schematic layout out of net.bus.geo (GeoJSON strings)."""
    pts = []
    for g in net.bus.geo:
        if isinstance(g, str):
            pts.append(json.loads(g)["coordinates"])
        elif isinstance(g, dict):
            pts.append(g["coordinates"])
        else:  # pragma: no cover - older pandapower fallback
            pts.append([np.nan, np.nan])
    xy = np.asarray(pts, dtype=float)
    # A handful of test cases have missing coordinates; fill with a spring layout.
    if np.isnan(xy).any():
        g = nx.Graph()
        g.add_nodes_from(net.bus.index)
        g.add_edges_from(zip(net.line.from_bus, net.line.to_bus))
        pos = nx.spring_layout(g, seed=0)
        for i, b in enumerate(net.bus.index):
            if np.isnan(xy[i]).any():
                xy[i] = pos[b]
    return xy


def _to_lonlat(xy: np.ndarray) -> np.ndarray:
    """Affine-map arbitrary layout units onto the study bounding box."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    unit = (xy - lo) / (hi - lo)
    lon = LON_MIN + unit[:, 0] * (LON_MAX - LON_MIN)
    lat = LAT_MIN + unit[:, 1] * (LAT_MAX - LAT_MIN)
    return np.column_stack([lon, lat])


def _to_km(lonlat: np.ndarray) -> np.ndarray:
    x = (lonlat[:, 0] - LON_MIN) * KM_PER_DEG_LON
    y = (lonlat[:, 1] - LAT_MIN) * KM_PER_DEG_LAT
    return np.column_stack([x, y])


def _river_centreline(n_pts: int = 240, seed: int = 7) -> np.ndarray:
    """
    A meandering river running roughly south-west to north-east across the
    domain, in the planar km frame. Deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    width_km = (LON_MAX - LON_MIN) * KM_PER_DEG_LON
    height_km = (LAT_MAX - LAT_MIN) * KM_PER_DEG_LAT
    t = np.linspace(0.0, 1.0, n_pts)
    x = t * width_km
    # Base diagonal trend plus two meander harmonics with random phase.
    phase = rng.uniform(0, 2 * np.pi, size=2)
    y = (
        0.20 * height_km
        + 0.55 * height_km * t
        + 0.07 * height_km * np.sin(2 * np.pi * 1.5 * t + phase[0])
        + 0.04 * height_km * np.sin(2 * np.pi * 3.5 * t + phase[1])
    )
    return np.column_stack([x, y])


def _distance_to_polyline(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    """
    Perpendicular distance from each point to a polyline, computed segment-wise.
    Vectorised over points; loops over the (few hundred) segments.
    """
    a, b = line[:-1], line[1:]  # segment endpoints
    seg = b - a  # (n_seg, 2)
    seg_len2 = np.einsum("ij,ij->i", seg, seg)
    seg_len2 = np.where(seg_len2 == 0, 1e-12, seg_len2)

    best = np.full(len(pts), np.inf)
    for k in range(len(a)):
        w = pts - a[k]  # (n_pts, 2)
        t = np.clip((w @ seg[k]) / seg_len2[k], 0.0, 1.0)
        proj = a[k] + t[:, None] * seg[k]
        d = np.linalg.norm(pts - proj, axis=1)
        best = np.minimum(best, d)
    return best


def _synthetic_elevation(dist_km: np.ndarray, xy_km: np.ndarray, seed: int = 11) -> np.ndarray:
    """
    Synthetic DEM. Terrain rises away from the river (a floodplain), with smooth
    spatial noise on top so elevation is not a pure function of river distance.

    Elevation is measured relative to the river bed (0 m at the channel).
    """
    rng = np.random.default_rng(seed)
    # Concave rise: steep near the channel, flattening out on the plain.
    base = 9.0 * np.sqrt(np.maximum(dist_km, 0.0))
    # Two smooth Fourier bumps for terrain texture (deterministic given seed).
    kx, ky = rng.uniform(0.02, 0.06, size=2)
    px, py = rng.uniform(0, 2 * np.pi, size=2)
    texture = 6.0 * np.sin(kx * xy_km[:, 0] + px) * np.cos(ky * xy_km[:, 1] + py)
    jitter = rng.normal(0.0, 1.5, size=len(dist_km))
    return np.maximum(base + texture + jitter, 0.2)


def build_grid(seed: int = 7) -> Grid:
    """Load IEEE 118-bus, georeference it, and attach terrain + graph features."""
    net = ppn.case118()

    # --- geometry -------------------------------------------------------
    lonlat = _to_lonlat(_layout_xy(net))
    xy_km = _to_km(lonlat)
    river = _river_centreline(seed=seed)
    dist = _distance_to_polyline(xy_km, river)
    elev = _synthetic_elevation(dist, xy_km, seed=seed + 4)

    # --- topology -------------------------------------------------------
    g = nx.Graph()
    g.add_nodes_from(net.bus.index.tolist())
    for f, t in zip(net.line.from_bus, net.line.to_bus):
        g.add_edge(int(f), int(t), kind="line")
    for f, t in zip(net.trafo.hv_bus, net.trafo.lv_bus):
        g.add_edge(int(f), int(t), kind="trafo")

    n = len(net.bus)
    idx = {b: i for i, b in enumerate(net.bus.index)}

    deg = np.zeros(n)
    for b, d in g.degree():
        deg[idx[b]] = d
    btw_raw = nx.betweenness_centrality(g)
    btw = np.array([btw_raw[b] for b in net.bus.index])

    # --- electrical attributes per bus ----------------------------------
    load = np.zeros(n)
    for b, p in zip(net.load.bus, net.load.p_mw):
        load[idx[int(b)]] += float(p)
    gen = np.zeros(n)
    for b, p in zip(net.gen.bus, net.gen.p_mw):
        gen[idx[int(b)]] += float(p)

    slack = int(net.ext_grid.bus.iloc[0])

    return Grid(
        net=net,
        graph=g,
        xy_km=xy_km,
        lonlat=lonlat,
        elevation_m=elev,
        dist_to_river_km=dist,
        bus_load_mw=load,
        bus_gen_mw=gen,
        vn_kv=net.bus.vn_kv.to_numpy(dtype=float),
        degree=deg,
        betweenness=btw,
        river_xy_km=river,
        slack_bus=slack,
    )


def edge_index(grid: Grid) -> np.ndarray:
    """
    (2, 2*E) array of directed edges in both directions, in *positional* index
    space (0..n_bus-1), which is what PyTorch Geometric expects.
    """
    idx = {b: i for i, b in enumerate(grid.net.bus.index)}
    src, dst = [], []
    for u, v in grid.graph.edges():
        a, b = idx[u], idx[v]
        src += [a, b]
        dst += [b, a]
    return np.array([src, dst], dtype=np.int64)


if __name__ == "__main__":
    grid = build_grid()
    print(f"buses={len(grid.net.bus)} edges={grid.graph.number_of_edges()}")
    print(f"elevation  min={grid.elevation_m.min():.1f} m  max={grid.elevation_m.max():.1f} m")
    print(f"river dist min={grid.dist_to_river_km.min():.1f} km max={grid.dist_to_river_km.max():.1f} km")
    print(f"total load={grid.bus_load_mw.sum():.0f} MW")
