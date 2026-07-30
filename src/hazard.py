"""
Stochastic flood event set.

This is the catastrophe-modelling half of the project. Because there is no
useful historical record of flood-driven cascading grid failure, we do not fit
a hazard model to data. We *generate* one: a synthetic event set in which each
event has a random severity, a random affected reach of the river, and a
physically-motivated spatial decay away from the channel.

Each event produces a flood depth at every substation. That is exactly the
role a hazard footprint plays in a commercial CAT model (RMS/AIR): event set ->
intensity field -> intensity at each exposure location.

Swapping in real hazard data
----------------------------
`depths_from_raster()` reads a real flood depth GeoTIFF (e.g. a FEMA NFHL
depth grid, a First Street layer, or a hydraulic model output) and samples it
at the substation coordinates. The rest of the pipeline is unchanged: it only
ever sees a vector of depths per event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from network import Grid


@dataclass
class HazardParams:
    """Parameters of the stochastic event set. All lengths in metres or km."""

    severity_median_m: float = 25.0  # median peak water surface rise at the channel
    severity_log_sigma: float = 0.45  # lognormal dispersion of severity
    decay_length_km: float = 16.0  # lateral e-folding distance from the channel
    reach_frac_mean: float = 0.28  # mean flooded reach half-length, as a
    #                                fraction of total river length
    reach_frac_sd: float = 0.10


def river_projection(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    For every bus, return (perpendicular distance to the river in km,
    normalised arc-length position of the closest point on the river in [0, 1]).

    The arc-length position is what lets an event flood only part of the river
    rather than all of it, which is what produces spatially localised events.
    """
    line = grid.river_xy_km
    a, b = line[:-1], line[1:]
    seg = b - a
    seg_len = np.linalg.norm(seg, axis=1)
    seg_len2 = np.where(seg_len**2 == 0, 1e-12, seg_len**2)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    pts = grid.xy_km
    best_d = np.full(len(pts), np.inf)
    best_s = np.zeros(len(pts))
    for k in range(len(a)):
        w = pts - a[k]
        t = np.clip((w @ seg[k]) / seg_len2[k], 0.0, 1.0)
        proj = a[k] + t[:, None] * seg[k]
        d = np.linalg.norm(pts - proj, axis=1)
        hit = d < best_d
        best_d[hit] = d[hit]
        best_s[hit] = (cum[k] + t[hit] * seg_len[k]) / total
    return best_d, best_s


def sample_event(
    grid: Grid,
    rng: np.random.Generator,
    params: HazardParams,
    proj: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """
    Draw one flood event and return the depth (metres of standing water) at
    every substation, plus the event parameters for the record.
    """
    dist_km, arc = river_projection(grid) if proj is None else proj

    # 1. Severity: peak water surface elevation at the channel, lognormal.
    #    Heavy right tail is deliberate: rare events are the ones that matter.
    mu = np.log(params.severity_median_m)
    severity = float(rng.lognormal(mean=mu, sigma=params.severity_log_sigma))

    # 2. Which reach of the river floods. Centre uniform, half-length gamma.
    centre = float(rng.uniform(0.08, 0.92))
    shape = (params.reach_frac_mean / params.reach_frac_sd) ** 2
    scale = params.reach_frac_sd**2 / params.reach_frac_mean
    half_len = float(np.clip(rng.gamma(shape, scale), 0.05, 0.9))

    # 3. Along-river taper: full severity at the centre of the reach, smoothly
    #    to zero at its ends (raised cosine).
    u = np.clip(np.abs(arc - centre) / half_len, 0.0, 1.0)
    along = np.cos(0.5 * np.pi * u) ** 2

    # 4. Lateral decay away from the channel.
    lateral = np.exp(-dist_km / params.decay_length_km)

    # 5. Water surface elevation, then depth above ground.
    wse = severity * along * lateral
    depth = np.maximum(wse - grid.elevation_m, 0.0)

    return {
        "depth_m": depth,
        "severity_m": severity,
        "reach_centre": centre,
        "reach_half_len": half_len,
    }


def depths_from_raster(grid: Grid, raster_path: str, band: int = 1) -> np.ndarray:
    """
    Adapter for a REAL flood depth raster (FEMA NFHL depth grid, hydraulic model
    output, etc). Samples the raster at each substation's lon/lat and returns
    depth in metres. Requires `rasterio`, which is an optional dependency.

    Everything downstream of this function is identical whether depths come
    from the synthetic event set or from a real hazard layer.
    """
    import rasterio  # optional dependency
    from rasterio.warp import transform as warp_transform

    with rasterio.open(raster_path) as src:
        lon, lat = grid.lonlat[:, 0], grid.lonlat[:, 1]
        xs, ys = warp_transform("EPSG:4326", src.crs, lon.tolist(), lat.tolist())
        vals = np.array([v[band - 1] for v in src.sample(zip(xs, ys))], dtype=float)
        nodata = src.nodata
    if nodata is not None:
        vals = np.where(vals == nodata, 0.0, vals)
    return np.nan_to_num(np.maximum(vals, 0.0))


if __name__ == "__main__":
    from network import build_grid

    grid = build_grid()
    proj = river_projection(grid)
    rng = np.random.default_rng(0)
    p = HazardParams()
    wet = []
    for _ in range(2000):
        ev = sample_event(grid, rng, p, proj)
        wet.append((ev["depth_m"] > 0).sum())
    wet = np.array(wet)
    print(f"buses inundated per event: mean={wet.mean():.1f} "
          f"median={np.median(wet):.0f} p90={np.quantile(wet, 0.9):.0f} max={wet.max()}")
    print(f"events with zero inundation: {(wet == 0).mean():.1%}")
