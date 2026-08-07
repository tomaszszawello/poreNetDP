#!/usr/bin/env python3
"""Post-process poreNetDP simulations.

The script reads the text files written by ``data_2.py`` and by the spatial
metrics extension:

``params.txt`` (10 columns)
    time, primary solid fraction, pressure drop, porosity,
    secondary solid fraction, cumulative B consumption/export balance,
    cumulative C export/balance, cumulative dissolved edge volume,
    cumulative precipitated edge volume, cumulative D consumption.

``spatial_metrics.txt`` (optional; 9 columns)
    time, x_A50, x_A50/L, x_Q20, x_Q20/L, max(|q_e|)/Q_in,
    number of A50 grains, number of Q20 edges, Q_in.

``profiles.txt`` (optional)
    First row: number of edges crossing each axial slice.
    Following rows: number of edges needed to carry 50% of the slice flow.

``profiles_phi.txt`` (optional)
    Axial porosity profiles at successive saved snapshots.

For each run, the script writes processed CSV files, a JSON summary, and plots.
When several runs are supplied, it also writes aggregate tables and comparison
plots.

Examples
--------
Process one run::

    python postprocess_porenet.py path/to/run

Process all immediate run directories inside ``results``::

    python postprocess_porenet.py results

Search recursively and compare all runs::

    python postprocess_porenet.py results --recursive --output analysis

Use cumulative injected volume instead of time on plots::

    python postprocess_porenet.py results --recursive --x-axis throughput

Notes
-----
Under constant imposed flow, relative conductivity is ``K/K0 = dp0/dp``.
When ``Q_in`` is available in ``spatial_metrics.txt``, ``--conductivity-mode
auto`` uses ``(Q/dp)/(Q0/dp0)``; this also works for constant-flow runs.

Besides the coarse hydraulic label, the script assigns every run to one of
eight manuscript regimes: compact clogging, channelized clogging, braided
channeling with net closure, persistent wormholing, tip-branching
exploration, pathway-switching exploration, distributed oscillatory
competition, or front-propagating replacement.  The transparent rule set
combines conductivity evolution, axial dissolution penetration, retreat and
re-advance of the significant-flow front, the slice flow-focusing index, the
number of flow-carrying branches, and (when available) the axial porosity
profile.  The resulting labels are operational and are accompanied by the
metrics, thresholds, confidence, and rationale used to assign them. For batch
runs, ``Da_eff`` and ``cd_in`` are read from the ``config.txt`` located beside
that run's ``params.txt``. The script additionally writes a three-regime
Singurindy--Berkowitz-style map, the full eight-regime map, and a heatmap of
replaced volume in ``(1/Da_eff, 1/cd_in)`` space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import find_peaks, savgol_filter


PARAM_COLUMNS = (
    "time",
    "primary_solid_fraction",
    "pressure_drop",
    "porosity",
    "secondary_solid_fraction",
    "B_balance_cumulative",
    "C_balance_cumulative",
    "dissolved_edge_volume_cumulative",
    "precipitated_edge_volume_cumulative",
    "D_balance_cumulative",
)

SPATIAL_COLUMNS = (
    "time",
    "x_A50",
    "x_A50_over_L",
    "x_Q20",
    "x_Q20_over_L",
    "max_abs_q_over_Qin",
    "n_A50_grains",
    "n_Q20_edges",
    "Q_in",
)

PROFILE_TIME_CANDIDATES = (
    "profile_times.txt",
    "profiles_times.txt",
    "slice_times.txt",
)


MANUSCRIPT_REGIMES = (
    "compact_clogging",
    "channelized_clogging",
    "braided_channeling_with_net_closure",
    "persistent_wormholing",
    "tip_branching_exploration",
    "pathway_switching_exploration",
    "distributed_oscillatory_competition",
    "front_propagating_replacement",
)


THREE_REGIMES = (
    "dissolution_dominated",
    "competition",
    "precipitation_dominated",
)

THREE_REGIME_LABELS = {
    "dissolution_dominated": "dissolution dominated",
    "competition": "competition",
    "precipitation_dominated": "precipitation dominated",
}

THREE_REGIME_CODES = {
    "dissolution_dominated": "D",
    "competition": "C",
    "precipitation_dominated": "P",
}

MANUSCRIPT_REGIME_LABELS = {
    regime: regime.replace("_", " ") for regime in MANUSCRIPT_REGIMES
}

MANUSCRIPT_REGIME_CODES = {
    "compact_clogging": "CC",
    "channelized_clogging": "ChC",
    "braided_channeling_with_net_closure": "BC",
    "persistent_wormholing": "W",
    "tip_branching_exploration": "TBE",
    "pathway_switching_exploration": "PSE",
    "distributed_oscillatory_competition": "DOC",
    "front_propagating_replacement": "FPR",
}

REPLACEMENT_METRIC_SPECS = {
    "secondary_final": (
        "replaced_volume_fraction_final",
        r"final secondary-mineral volume fraction $V_E/V_{\mathrm{tot}}$",
    ),
    "secondary_added": (
        "replaced_volume_added_fraction_final",
        r"secondary-mineral volume added $\Delta V_E/V_{\mathrm{tot}}$",
    ),
    "primary_removed": (
        "primary_removed_volume_fraction_final",
        r"primary-mineral volume removed $\Delta V_A/V_{\mathrm{tot}}$",
    ),
    "primary_removed_normalized": (
        "primary_removed_normalized_final",
        r"fraction of initial primary mineral removed $\Delta V_A/V_A^0$",
    ),
}

_PHASE_SELECTOR_ALIASES = {
    "run", "folder", "name", "case", "simulation", "id", "path", "directory"
}
_PHASE_DAEFF_ALIASES = {
    "daeff", "daeffective", "effectiveda", "effectivedamkohler",
    "damkohlereffective", "daeff0"
}
_PHASE_CDIN_ALIASES = {
    "cdin", "cdinlet", "dininlet", "din", "inletcd", "inletd"
}


@dataclass(frozen=True)
class Settings:
    conductivity_mode: str
    smooth_window: int
    smooth_polyorder: int
    peak_prominence: float
    peak_distance: int
    net_log_threshold: float
    oscillation_threshold: float
    breakthrough_threshold: float
    deep_penetration_threshold: float
    clogged_conductivity_threshold: float
    front_smooth_window: int
    front_retreat_prominence: float
    front_event_distance: int
    front_recovery_fraction: float
    front_nonmonotonicity_threshold: float
    front_persistence_threshold: float
    front_profile_score_threshold: float
    front_profile_min_amplitude: float
    focusing_threshold: float
    focusing_local_threshold: float
    focusing_late_fraction: float
    focusing_axial_margin: float
    dominant_edge_fraction_threshold: float
    braided_n50_threshold: float
    braided_slice_fraction_threshold: float
    x_axis: str
    formats: tuple[str, ...]
    dpi: int
    include_profiles: bool
    include_phase_diagrams: bool
    phase_axis_scale: str
    phase_cell_labels: bool
    replacement_metric: str


@dataclass
class RunResult:
    name: str
    source_dir: Path
    output_dir: Path
    params: dict[str, np.ndarray]
    spatial: dict[str, np.ndarray] | None
    processed_spatial: dict[str, np.ndarray] | None
    processed: dict[str, np.ndarray]
    metrics: dict[str, Any]
    focusing_profiles: np.ndarray | None
    porosity_profiles: np.ndarray | None
    profile_axis: np.ndarray | None
    profile_times: np.ndarray | None
    focusing_summary: dict[str, np.ndarray] | None


def _load_numeric_table(path: Path, expected_columns: int | None = None) -> np.ndarray:
    """Load a whitespace-separated numeric table and always return 2-D data."""
    try:
        data = np.loadtxt(path, comments="#", ndmin=2)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric data in {path}: {exc}") from exc

    if data.size == 0:
        raise ValueError(f"{path} is empty")
    if expected_columns is not None and data.shape[1] < expected_columns:
        raise ValueError(
            f"{path} has {data.shape[1]} columns; expected at least {expected_columns}"
        )
    return np.asarray(data, dtype=float)


def _table_to_dict(data: np.ndarray, columns: Sequence[str]) -> dict[str, np.ndarray]:
    return {name: np.asarray(data[:, i], dtype=float) for i, name in enumerate(columns)}


def _sort_and_deduplicate(table: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sort by time and retain the last record for duplicate times."""
    time = np.asarray(table["time"], dtype=float)
    finite = np.isfinite(time)
    if not np.any(finite):
        raise ValueError("Time column contains no finite values")

    filtered = {key: np.asarray(value)[finite] for key, value in table.items()}
    order = np.argsort(filtered["time"], kind="stable")
    filtered = {key: value[order] for key, value in filtered.items()}

    times = filtered["time"]
    # Reversing before np.unique makes it straightforward to keep the last row.
    _, reverse_indices = np.unique(times[::-1], return_index=True)
    keep = np.sort(times.size - 1 - reverse_indices)
    return {key: value[keep] for key, value in filtered.items()}


def load_params(run_dir: Path) -> dict[str, np.ndarray]:
    data = _load_numeric_table(run_dir / "params.txt", len(PARAM_COLUMNS))
    table = _table_to_dict(data[:, : len(PARAM_COLUMNS)], PARAM_COLUMNS)
    return _sort_and_deduplicate(table)


def load_spatial(run_dir: Path) -> dict[str, np.ndarray] | None:
    path = run_dir / "spatial_metrics.txt"
    if not path.exists():
        return None
    data = _load_numeric_table(path, len(SPATIAL_COLUMNS))
    table = _table_to_dict(data[:, : len(SPATIAL_COLUMNS)], SPATIAL_COLUMNS)
    return _sort_and_deduplicate(table)


def load_matrix_if_present(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        data = _load_numeric_table(path)
    except ValueError as exc:
        warnings.warn(str(exc))
        return None
    return data


def load_profile_times(run_dir: Path, expected_length: int) -> np.ndarray | None:
    """Load optional physical times associated with profile snapshots."""
    for filename in PROFILE_TIME_CANDIDATES:
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            values = np.asarray(np.loadtxt(path, comments="#"), dtype=float).ravel()
        except (OSError, ValueError) as exc:
            warnings.warn(f"Could not parse {path}: {exc}")
            continue
        values = values[np.isfinite(values)]
        if values.size != expected_length:
            warnings.warn(
                f"Ignoring {path}: found {values.size} profile times, "
                f"expected {expected_length}"
            )
            continue
        return values
    return None


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return np.asarray(x[mask], dtype=float), np.asarray(y[mask], dtype=float)


def interpolate_series(
    target_x: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate finite values, using endpoint values outside range."""
    source_x, source_y = _finite_xy(source_x, source_y)
    result = np.full_like(np.asarray(target_x, dtype=float), np.nan, dtype=float)
    if source_x.size == 0:
        return result
    if source_x.size == 1:
        result[:] = source_y[0]
        return result
    order = np.argsort(source_x)
    source_x = source_x[order]
    source_y = source_y[order]
    result[:] = np.interp(target_x, source_x, source_y)
    return result


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with a zero initial value."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    if x.size < 2:
        return out
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    increments[~np.isfinite(increments)] = 0.0
    out[1:] = np.cumsum(increments)
    return out


def choose_savgol_window(n: int, requested: int, polyorder: int) -> int:
    """Return a valid odd Savitzky--Golay window, or zero if smoothing is unsafe."""
    if n < polyorder + 2:
        return 0
    if requested <= 0:
        requested = min(21, max(5, int(round(n * 0.05))))
    if requested % 2 == 0:
        requested += 1
    maximum = n if n % 2 == 1 else n - 1
    window = min(requested, maximum)
    if window <= polyorder:
        window = polyorder + 1
        if window % 2 == 0:
            window += 1
    return window if window <= maximum else 0


def smooth_log_conductivity(
    conductivity: np.ndarray,
    requested_window: int,
    polyorder: int,
) -> tuple[np.ndarray, int]:
    conductivity = np.asarray(conductivity, dtype=float)
    if np.any(~np.isfinite(conductivity)) or np.any(conductivity <= 0):
        raise ValueError("Relative conductivity must be finite and positive")
    log_k = np.log(conductivity)
    window = choose_savgol_window(log_k.size, requested_window, polyorder)
    if window == 0:
        return log_k.copy(), 0
    return savgol_filter(log_k, window_length=window, polyorder=polyorder), window


def derive_conductivity(
    params: Mapping[str, np.ndarray],
    spatial: Mapping[str, np.ndarray] | None,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return K/K0, Q interpolated to params times, and the formula used."""
    time = np.asarray(params["time"], dtype=float)
    pressure = np.asarray(params["pressure_drop"], dtype=float)
    if pressure.size == 0 or np.any(~np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError("pressure_drop must contain positive finite values")

    q_at_time = np.full_like(time, np.nan, dtype=float)
    q_available = False
    if spatial is not None:
        q_at_time = interpolate_series(time, spatial["time"], spatial["Q_in"])
        q_available = bool(np.all(np.isfinite(q_at_time)) and np.all(q_at_time > 0))

    use_variable = mode == "variable-flow" or (mode == "auto" and q_available)
    if use_variable:
        if not q_available:
            raise ValueError(
                "Variable-flow conductivity requested, but positive Q_in data are unavailable"
            )
        hydraulic_conductance = q_at_time / pressure
        conductivity = hydraulic_conductance / hydraulic_conductance[0]
        formula = "(Q_in/dp)/(Q_in[0]/dp[0])"
    else:
        conductivity = pressure[0] / pressure
        formula = "dp[0]/dp (constant imposed flow)"

    return conductivity, q_at_time, formula


def threshold_crossing_time(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float:
    """First linearly interpolated upward crossing; NaN if never reached."""
    time, values = _finite_xy(np.asarray(time), np.asarray(values))
    if time.size == 0:
        return math.nan
    above = values >= threshold
    if not np.any(above):
        return math.nan
    index = int(np.argmax(above))
    if index == 0:
        return float(time[0])
    x0, x1 = time[index - 1], time[index]
    y0, y1 = values[index - 1], values[index]
    if y1 == y0:
        return float(x1)
    fraction = (threshold - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def normalized_total_variation(values: np.ndarray) -> tuple[float, float]:
    """Return total variation and 1-|net change|/TV."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0, 0.0
    total_variation = float(np.sum(np.abs(np.diff(values))))
    net = float(values[-1] - values[0])
    if total_variation <= np.finfo(float).eps:
        return total_variation, 0.0
    index = 1.0 - abs(net) / total_variation
    return total_variation, float(np.clip(index, 0.0, 1.0))


def _contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals spanning contiguous True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def smooth_bounded_series(
    values: np.ndarray,
    requested_window: int,
    polyorder: int,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[np.ndarray, int]:
    """Smooth each finite segment independently.

    Missing values are intentionally not bridged.  For x_Q20 this prevents a
    temporary absence of any edge carrying 20% of inlet flow from being
    interpreted as an axial retreat to the inlet.
    """
    raw = np.asarray(values, dtype=float)
    result = np.full_like(raw, np.nan, dtype=float)
    windows: list[int] = []
    finite = np.isfinite(raw)
    for start, stop in _contiguous_true_segments(finite):
        segment = raw[start:stop]
        window = choose_savgol_window(segment.size, requested_window, polyorder)
        windows.append(window)
        if window:
            result[start:stop] = savgol_filter(
                segment,
                window_length=window,
                polyorder=polyorder,
            )
        else:
            result[start:stop] = segment
    if lower is not None:
        result = np.where(np.isfinite(result), np.maximum(result, lower), result)
    if upper is not None:
        result = np.where(np.isfinite(result), np.minimum(result, upper), result)
    return result, max(windows, default=0)


def _late_slice(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        return values
    count = max(1, int(math.ceil(values.size * fraction)))
    return values[-count:]


def analyze_spatial_metrics(
    spatial: Mapping[str, np.ndarray] | None,
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None, np.ndarray]:
    """Calculate penetration and significant-flow-front diagnostics.

    ``x_A50/L`` is treated as a reaction-penetration front. Its monotonicity
    and persistence are used as evidence for front-propagating replacement.

    A completed Q20 reorganization event is a significant trough in the
    smoothed ``x_Q20/L`` trajectory followed by re-advance. The Q20 analysis
    is performed only during intervals in which at least one edge satisfies
    the Q20 criterion, so temporary defocusing is not mistaken for retreat to
    the inlet.
    """
    if spatial is None:
        return {"spatial_metrics_available": False}, None, np.array([], dtype=int)

    processed = {
        key: np.asarray(value, dtype=float).copy()
        for key, value in spatial.items()
    }
    t = processed["time"]
    xa_raw = processed["x_A50_over_L"]
    xq_raw = processed["x_Q20_over_L"]
    max_fraction = processed["max_abs_q_over_Qin"]
    n_a = processed["n_A50_grains"]
    n_q = processed["n_Q20_edges"]
    q = processed["Q_in"]

    # Smooth A50 and quantify whether the penetration front advances
    # persistently or repeatedly retreats.
    xa_masked = np.where(np.isfinite(xa_raw), xa_raw, np.nan)
    xa_smooth, xa_window = smooth_bounded_series(
        xa_masked,
        settings.front_smooth_window,
        settings.smooth_polyorder,
        lower=0.0,
        upper=1.0,
    )
    xa_retreat_increment = np.zeros_like(xa_smooth)
    xa_advance_increment = np.zeros_like(xa_smooth)
    xa_drawdown = np.full_like(xa_smooth, np.nan)
    xa_total_variation = 0.0
    xa_max_drawdown = 0.0
    for seg_start, seg_stop in _contiguous_true_segments(np.isfinite(xa_smooth)):
        segment = xa_smooth[seg_start:seg_stop]
        if segment.size == 0:
            continue
        if segment.size >= 2:
            delta = np.diff(segment)
            xa_retreat_increment[seg_start + 1:seg_stop] = np.maximum(-delta, 0.0)
            xa_advance_increment[seg_start + 1:seg_stop] = np.maximum(delta, 0.0)
            xa_total_variation += float(np.sum(np.abs(delta)))
            local_drawdown = np.maximum.accumulate(segment) - segment
            xa_drawdown[seg_start:seg_stop] = local_drawdown
            xa_max_drawdown = max(xa_max_drawdown, float(np.max(local_drawdown)))
        else:
            xa_drawdown[seg_start] = 0.0

    finite_xa = xa_smooth[np.isfinite(xa_smooth)]
    if finite_xa.size >= 2 and xa_total_variation > np.finfo(float).eps:
        xa_net_advance = float(finite_xa[-1] - finite_xa[0])
        xa_nonmonotonicity = float(
            np.clip(1.0 - abs(xa_net_advance) / xa_total_variation, 0.0, 1.0)
        )
    else:
        xa_net_advance = 0.0
        xa_nonmonotonicity = 0.0
    xa_cumulative_retreat = np.cumsum(xa_retreat_increment)
    xa_cumulative_advance = np.cumsum(xa_advance_increment)
    xa_final_smoothed = float(finite_xa[-1]) if finite_xa.size else math.nan
    xa_max_smoothed = float(np.max(finite_xa)) if finite_xa.size else math.nan
    xa_persistence = (
        xa_final_smoothed / xa_max_smoothed
        if math.isfinite(xa_final_smoothed)
        and math.isfinite(xa_max_smoothed)
        and xa_max_smoothed > np.finfo(float).eps
        else math.nan
    )

    q20_present = (
        (np.isfinite(n_q) & (n_q > 0))
        | (
            np.isfinite(max_fraction)
            & (max_fraction >= settings.dominant_edge_fraction_threshold - 1e-12)
        )
    )
    xq_masked = np.where(q20_present & np.isfinite(xq_raw), xq_raw, np.nan)
    xq_smooth, front_window = smooth_bounded_series(
        xq_masked,
        settings.front_smooth_window,
        settings.smooth_polyorder,
        lower=0.0,
        upper=1.0,
    )

    retreat_increment = np.zeros_like(xq_smooth)
    advance_increment = np.zeros_like(xq_smooth)
    drawdown = np.full_like(xq_smooth, np.nan)
    event_marker = np.zeros_like(xq_smooth)
    event_retreat = np.zeros_like(xq_smooth)
    event_recovery = np.zeros_like(xq_smooth)

    events: list[int] = []
    strengths: list[float] = []
    retreat_amplitudes: list[float] = []
    recovery_amplitudes: list[float] = []
    total_variation = 0.0
    max_drawdown = 0.0

    for seg_start, seg_stop in _contiguous_true_segments(np.isfinite(xq_smooth)):
        segment = xq_smooth[seg_start:seg_stop]
        if segment.size == 0:
            continue
        if segment.size >= 2:
            delta = np.diff(segment)
            retreat_increment[seg_start + 1:seg_stop] = np.maximum(-delta, 0.0)
            advance_increment[seg_start + 1:seg_stop] = np.maximum(delta, 0.0)
            total_variation += float(np.sum(np.abs(delta)))
            local_drawdown = np.maximum.accumulate(segment) - segment
            drawdown[seg_start:seg_stop] = local_drawdown
            max_drawdown = max(max_drawdown, float(np.max(local_drawdown)))
        else:
            drawdown[seg_start] = 0.0

        if segment.size < 3:
            continue
        candidates, properties = find_peaks(
            -segment,
            prominence=settings.front_retreat_prominence,
            distance=max(1, int(settings.front_event_distance)),
        )
        left_bases = np.asarray(
            properties.get("left_bases", np.zeros(candidates.size, dtype=int)),
            dtype=int,
        )
        right_bases = np.asarray(
            properties.get("right_bases", np.zeros(candidates.size, dtype=int)),
            dtype=int,
        )
        for i, local_trough in enumerate(candidates):
            left = int(left_bases[i])
            right = int(right_bases[i])
            retreat = float(segment[left] - segment[local_trough])
            recovery = float(segment[right] - segment[local_trough])
            if retreat < settings.front_retreat_prominence:
                continue
            if recovery < settings.front_recovery_fraction * settings.front_retreat_prominence:
                continue
            trough = seg_start + int(local_trough)
            events.append(trough)
            retreat_amplitudes.append(retreat)
            recovery_amplitudes.append(recovery)
            strengths.append(min(retreat, recovery))
            event_marker[trough] = 1.0
            event_retreat[trough] = retreat
            event_recovery[trough] = recovery

    cumulative_retreat = np.cumsum(retreat_increment)
    cumulative_advance = np.cumsum(advance_increment)
    event_indices = np.asarray(sorted(set(events)), dtype=int)

    finite_front = xq_smooth[np.isfinite(xq_smooth)]
    if finite_front.size >= 2 and total_variation > np.finfo(float).eps:
        front_net = float(finite_front[-1] - finite_front[0])
        q_nonmonotonicity = float(
            np.clip(1.0 - abs(front_net) / total_variation, 0.0, 1.0)
        )
    else:
        q_nonmonotonicity = 0.0

    q20_loss_events = (
        int(np.count_nonzero(q20_present[:-1] & ~q20_present[1:]))
        if q20_present.size > 1
        else 0
    )
    q20_reappearance_events = (
        int(np.count_nonzero(~q20_present[:-1] & q20_present[1:]))
        if q20_present.size > 1
        else 0
    )
    final_q20_present = bool(q20_present[-1]) if q20_present.size else False
    if final_q20_present and np.isfinite(drawdown[-1]):
        ending_drawdown = float(drawdown[-1])
    else:
        ending_drawdown = math.nan

    q_mean = float(np.nanmean(q)) if np.any(np.isfinite(q)) else math.nan
    q_std = float(np.nanstd(q)) if np.any(np.isfinite(q)) else math.nan
    q_cv = q_std / abs(q_mean) if math.isfinite(q_mean) and abs(q_mean) > 0 else math.nan
    late_fraction = _late_slice(max_fraction, settings.focusing_late_fraction)

    processed.update(
        {
            "x_A50_over_L_smoothed": xa_smooth,
            "x_A50_retreat_increment": xa_retreat_increment,
            "x_A50_advance_increment": xa_advance_increment,
            "x_A50_cumulative_retreat": xa_cumulative_retreat,
            "x_A50_cumulative_advance": xa_cumulative_advance,
            "x_A50_drawdown_from_running_max": xa_drawdown,
            "Q20_present": q20_present.astype(float),
            "x_Q20_over_L_smoothed": xq_smooth,
            "x_Q20_retreat_increment": retreat_increment,
            "x_Q20_advance_increment": advance_increment,
            "x_Q20_cumulative_retreat": cumulative_retreat,
            "x_Q20_cumulative_advance": cumulative_advance,
            "x_Q20_drawdown_from_running_max": drawdown,
            "x_Q20_reorganization_event": event_marker,
            "x_Q20_event_retreat_amplitude": event_retreat,
            "x_Q20_event_recovery_amplitude": event_recovery,
        }
    )

    xq_final_smoothed = float(finite_front[-1]) if finite_front.size else math.nan
    xq_max_smoothed = float(np.max(finite_front)) if finite_front.size else math.nan
    metrics: dict[str, Any] = {
        "spatial_metrics_available": True,
        "front_savgol_window_used": front_window,
        "A50_savgol_window_used": xa_window,
        "x_A50_over_L_final": safe_float(xa_raw[-1]),
        "x_A50_over_L_final_smoothed": safe_float(xa_final_smoothed),
        "x_A50_over_L_max": safe_float(np.nanmax(xa_raw)),
        "x_A50_over_L_max_smoothed": safe_float(xa_max_smoothed),
        "x_A50_total_variation": float(xa_total_variation),
        "x_A50_nonmonotonicity_index": float(xa_nonmonotonicity),
        "x_A50_net_advance_over_L": float(xa_net_advance),
        "x_A50_cumulative_retreat_over_L": safe_float(xa_cumulative_retreat[-1]),
        "x_A50_cumulative_advance_over_L": safe_float(xa_cumulative_advance[-1]),
        "x_A50_max_drawdown_from_previous_reach_over_L": float(xa_max_drawdown),
        "x_A50_final_to_max_ratio": safe_float(xa_persistence),
        "x_Q20_over_L_final": safe_float(xq_raw[-1]),
        "x_Q20_over_L_final_smoothed": safe_float(xq_final_smoothed),
        "x_Q20_over_L_max": safe_float(np.nanmax(xq_raw)),
        "x_Q20_over_L_max_smoothed": safe_float(xq_max_smoothed),
        "x_Q20_total_variation": float(total_variation),
        "x_Q20_nonmonotonicity_index": q_nonmonotonicity,
        "x_Q20_cumulative_retreat_over_L": safe_float(cumulative_retreat[-1]),
        "x_Q20_cumulative_advance_over_L": safe_float(cumulative_advance[-1]),
        "x_Q20_max_drawdown_from_previous_reach_over_L": float(max_drawdown),
        "x_Q20_ending_drawdown_from_max_over_L": safe_float(ending_drawdown),
        "x_Q20_completed_retreat_recovery_events": int(event_indices.size),
        "x_Q20_completed_reorganization_strength_over_L": float(np.sum(strengths)) if strengths else 0.0,
        "x_Q20_largest_completed_reorganization_over_L": float(np.max(strengths)) if strengths else 0.0,
        "x_Q20_largest_retreat_amplitude_over_L": float(np.max(retreat_amplitudes)) if retreat_amplitudes else 0.0,
        "x_Q20_largest_recovery_amplitude_over_L": float(np.max(recovery_amplitudes)) if recovery_amplitudes else 0.0,
        "x_Q20_first_reorganization_time": safe_float(t[event_indices[0]]) if event_indices.size else math.nan,
        "x_Q20_last_reorganization_time": safe_float(t[event_indices[-1]]) if event_indices.size else math.nan,
        "Q20_presence_fraction": float(np.mean(q20_present)),
        "Q20_final_present": final_q20_present,
        "Q20_loss_events": q20_loss_events,
        "Q20_reappearance_events": q20_reappearance_events,
        "time_x_A50_reaches_half_length": threshold_crossing_time(t, xa_smooth, 0.5),
        "time_x_A50_reaches_breakthrough_threshold": threshold_crossing_time(
            t, xa_smooth, settings.breakthrough_threshold
        ),
        "time_x_Q20_reaches_breakthrough_threshold": threshold_crossing_time(
            t, xq_smooth, settings.breakthrough_threshold
        ),
        "max_edge_flow_fraction_final": safe_float(max_fraction[-1]),
        "max_edge_flow_fraction_max": safe_float(np.nanmax(max_fraction)),
        "max_edge_flow_fraction_late_mean": safe_float(np.nanmean(late_fraction)),
        "n_A50_grains_final": safe_float(n_a[-1]),
        "n_A50_grains_max": safe_float(np.nanmax(n_a)),
        "n_Q20_edges_final": safe_float(n_q[-1]),
        "n_Q20_edges_max": safe_float(np.nanmax(n_q)),
        "fraction_snapshots_with_Q20_edge": float(np.mean(q20_present)),
        "Q_in_mean": q_mean,
        "Q_in_coefficient_of_variation": q_cv,
    }
    return metrics, processed, event_indices

def conductivity_metrics(
    time: np.ndarray,
    conductivity: np.ndarray,
    smooth_log_k: np.ndarray,
    settings: Settings,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Calculate conductivity statistics and significant extrema."""
    log_k = np.log(conductivity)
    total_variation, oscillation_index = normalized_total_variation(smooth_log_k)
    net_log_change = float(smooth_log_k[-1] - smooth_log_k[0])

    distance = max(1, int(settings.peak_distance))
    peak_indices, peak_props = find_peaks(
        smooth_log_k,
        prominence=settings.peak_prominence,
        distance=distance,
    )
    trough_indices, trough_props = find_peaks(
        -smooth_log_k,
        prominence=settings.peak_prominence,
        distance=distance,
    )
    n_extrema = int(peak_indices.size + trough_indices.size)

    if net_log_change > settings.net_log_threshold:
        terminal_state = "opening"
    elif net_log_change < -settings.net_log_threshold:
        terminal_state = "closure"
    else:
        terminal_state = "balanced"

    if n_extrema >= 2 and oscillation_index >= settings.oscillation_threshold:
        trajectory = "oscillatory"
    elif n_extrema >= 1 and oscillation_index >= 0.5 * settings.oscillation_threshold:
        trajectory = "nonmonotonic"
    else:
        trajectory = "monotonic-like"

    running_max = np.maximum.accumulate(smooth_log_k)
    running_min = np.minimum.accumulate(smooth_log_k)
    max_drawdown = float(np.max(running_max - smooth_log_k))
    max_recovery = float(np.max(smooth_log_k - running_min))

    metrics: dict[str, Any] = {
        "hydraulic_response": f"{trajectory}_{terminal_state}",
        "hydraulic_terminal_state": terminal_state,
        "hydraulic_trajectory": trajectory,
        "conductivity_final_over_initial": float(conductivity[-1]),
        "conductivity_min_over_initial": float(np.min(conductivity)),
        "conductivity_max_over_initial": float(np.max(conductivity)),
        "conductivity_final_log_change": net_log_change,
        "conductivity_log_total_variation": total_variation,
        "conductivity_oscillation_index": oscillation_index,
        "conductivity_significant_extrema": n_extrema,
        "conductivity_significant_peaks": int(peak_indices.size),
        "conductivity_significant_troughs": int(trough_indices.size),
        "conductivity_peak_to_peak_log": float(np.ptp(smooth_log_k)),
        "conductivity_max_log_drawdown": max_drawdown,
        "conductivity_max_log_recovery": max_recovery,
        "time_of_min_conductivity": float(time[int(np.argmin(conductivity))]),
        "time_of_max_conductivity": float(time[int(np.argmax(conductivity))]),
    }
    if peak_indices.size:
        metrics["largest_peak_prominence_log"] = float(
            np.max(peak_props["prominences"])
        )
    else:
        metrics["largest_peak_prominence_log"] = 0.0
    if trough_indices.size:
        metrics["largest_trough_prominence_log"] = float(
            np.max(trough_props["prominences"])
        )
    else:
        metrics["largest_trough_prominence_log"] = 0.0

    return metrics, peak_indices, trough_indices


def _single_step_profile_score(
    values: np.ndarray,
    axis: np.ndarray,
    margin: float,
) -> tuple[float, float, float]:
    """Return best two-plateau step-fit score, split position, and contrast.

    The score is the fraction of variance explained by a single axial split.
    It is used only as optional support for a coherent replacement front; a
    small porosity signal is treated as unavailable rather than as evidence
    against front propagation.
    """
    values = np.asarray(values, dtype=float)
    axis = np.asarray(axis, dtype=float)
    mask = np.isfinite(values) & np.isfinite(axis)
    mask &= (axis >= margin) & (axis <= 1.0 - margin)
    y = values[mask]
    x = axis[mask]
    if y.size < 8:
        return math.nan, math.nan, math.nan
    total_sse = float(np.sum((y - np.mean(y)) ** 2))
    if total_sse <= np.finfo(float).eps:
        return math.nan, math.nan, math.nan

    minimum_side = max(3, int(math.ceil(0.08 * y.size)))
    best_sse = math.inf
    best_split = -1
    best_contrast = math.nan
    for split in range(minimum_side, y.size - minimum_side + 1):
        left = y[:split]
        right = y[split:]
        sse = float(
            np.sum((left - np.mean(left)) ** 2)
            + np.sum((right - np.mean(right)) ** 2)
        )
        if sse < best_sse:
            best_sse = sse
            best_split = split
            best_contrast = float(np.mean(left) - np.mean(right))

    if best_split <= 0 or best_split >= y.size:
        return math.nan, math.nan, math.nan
    score = float(np.clip(1.0 - best_sse / total_sse, 0.0, 1.0))
    split_position = float(0.5 * (x[best_split - 1] + x[best_split]))
    return score, split_position, best_contrast


def process_profiles(
    run_dir: Path,
    settings: Settings,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, np.ndarray] | None,
    dict[str, Any],
]:
    """Load axial profiles and summarize focusing, braiding, and front shape."""
    profiles = load_matrix_if_present(run_dir / "profiles.txt")
    phi = load_matrix_if_present(run_dir / "profiles_phi.txt")
    metrics: dict[str, Any] = {
        "flow_focusing_profiles_available": False,
        "porosity_profiles_available": phi is not None,
        "porosity_front_profile_supported": False,
    }

    focusing: np.ndarray | None = None
    profile_axis: np.ndarray | None = None
    profile_times: np.ndarray | None = None
    focusing_summary: dict[str, np.ndarray] | None = None

    if profiles is not None and profiles.shape[0] >= 2:
        total_edges = np.asarray(profiles[0], dtype=float)
        n50 = np.asarray(profiles[1:], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            focusing = 1.0 - 2.0 * n50 / total_edges[np.newaxis, :]
        focusing[:, total_edges <= 0] = np.nan
        # Discreteness can produce small negative values for odd edge counts.
        focusing = np.clip(focusing, 0.0, 1.0)
        profile_axis = np.linspace(0.0, 1.0, focusing.shape[1] + 2)[1:-1]
        profile_times = load_profile_times(run_dir, focusing.shape[0])

        margin = float(np.clip(settings.focusing_axial_margin, 0.0, 0.49))
        central_mask = (
            (profile_axis >= margin)
            & (profile_axis <= 1.0 - margin)
        )
        if not np.any(central_mask):
            central_mask = np.ones_like(profile_axis, dtype=bool)
        central = focusing[:, central_mask]
        central_n50 = n50[:, central_mask]

        mean_by_snapshot = np.nanmean(central, axis=1)
        median_by_snapshot = np.nanmedian(central, axis=1)
        p90_by_snapshot = np.nanpercentile(central, 90.0, axis=1)
        max_by_snapshot = np.nanmax(central, axis=1)
        fraction_above_by_snapshot = np.nanmean(
            central >= settings.focusing_local_threshold,
            axis=1,
        )
        n50_mean_by_snapshot = np.nanmean(central_n50, axis=1)
        n50_median_by_snapshot = np.nanmedian(central_n50, axis=1)
        n50_p90_by_snapshot = np.nanpercentile(central_n50, 90.0, axis=1)
        multichannel_fraction_by_snapshot = np.nanmean(
            central_n50 >= settings.braided_n50_threshold,
            axis=1,
        )

        coordinate = (
            profile_times.copy()
            if profile_times is not None
            else np.arange(focusing.shape[0], dtype=float)
        )
        focusing_summary = {
            "profile_coordinate": coordinate,
            "flow_focusing_mean": mean_by_snapshot,
            "flow_focusing_median": median_by_snapshot,
            "flow_focusing_p90": p90_by_snapshot,
            "flow_focusing_max": max_by_snapshot,
            "flow_focusing_fraction_x_above_local_threshold": fraction_above_by_snapshot,
            "flow_n50_mean": n50_mean_by_snapshot,
            "flow_n50_median": n50_median_by_snapshot,
            "flow_n50_p90": n50_p90_by_snapshot,
            "flow_multichannel_fraction_x": multichannel_fraction_by_snapshot,
        }

        late_count = max(
            1,
            int(math.ceil(focusing.shape[0] * settings.focusing_late_fraction)),
        )
        late = central[-late_count:]
        late_n50 = central_n50[-late_count:]
        late_snapshot_means = mean_by_snapshot[-late_count:]
        late_snapshot_fractions = fraction_above_by_snapshot[-late_count:]
        late_multichannel_fractions = multichannel_fraction_by_snapshot[-late_count:]
        focus_tv, focus_nonmonotonicity = normalized_total_variation(
            mean_by_snapshot
        )

        metrics.update(
            {
                "flow_focusing_profiles_available": True,
                "flow_focusing_profile_times_available": profile_times is not None,
                "flow_focusing_number_of_snapshots": int(focusing.shape[0]),
                "flow_focusing_axial_margin": margin,
                "flow_focusing_final_mean": safe_float(mean_by_snapshot[-1]),
                "flow_focusing_final_median": safe_float(median_by_snapshot[-1]),
                "flow_focusing_final_p90": safe_float(p90_by_snapshot[-1]),
                "flow_focusing_final_max": safe_float(max_by_snapshot[-1]),
                "flow_focusing_max_over_all_profiles": safe_float(np.nanmax(focusing)),
                "flow_focusing_mean_max_over_time": safe_float(np.nanmax(mean_by_snapshot)),
                "flow_focusing_late_mean": safe_float(np.nanmean(late)),
                "flow_focusing_late_snapshot_mean": safe_float(
                    np.nanmean(late_snapshot_means)
                ),
                "flow_focusing_late_p90": safe_float(
                    np.nanpercentile(late, 90.0)
                ),
                "flow_focusing_late_fraction_x_above_local_threshold": safe_float(
                    np.nanmean(late_snapshot_fractions)
                ),
                "flow_focusing_final_fraction_x_above_local_threshold": safe_float(
                    fraction_above_by_snapshot[-1]
                ),
                "flow_focusing_mean_total_variation": focus_tv,
                "flow_focusing_mean_nonmonotonicity_index": focus_nonmonotonicity,
                "flow_n50_final_mean": safe_float(n50_mean_by_snapshot[-1]),
                "flow_n50_final_median": safe_float(n50_median_by_snapshot[-1]),
                "flow_n50_final_p90": safe_float(n50_p90_by_snapshot[-1]),
                "flow_n50_late_mean": safe_float(np.nanmean(late_n50)),
                "flow_n50_late_median": safe_float(np.nanmedian(late_n50)),
                "flow_n50_late_p90": safe_float(np.nanpercentile(late_n50, 90.0)),
                "flow_multichannel_fraction_x_final": safe_float(
                    multichannel_fraction_by_snapshot[-1]
                ),
                "flow_multichannel_fraction_x_late": safe_float(
                    np.nanmean(late_multichannel_fractions)
                ),
            }
        )

    if phi is not None:
        if profile_axis is None or profile_axis.size != phi.shape[1]:
            profile_axis = np.linspace(0.0, 1.0, phi.shape[1] + 2)[1:-1]
        metrics.update(
            {
                "porosity_profile_number_of_snapshots": int(phi.shape[0]),
                "porosity_profile_final_mean": safe_float(np.nanmean(phi[-1])),
                "porosity_profile_final_min": safe_float(np.nanmin(phi[-1])),
                "porosity_profile_final_max": safe_float(np.nanmax(phi[-1])),
            }
        )
        if phi.shape[0] >= 2:
            delta_phi = np.asarray(phi[-1] - phi[0], dtype=float)
            finite_delta = delta_phi[np.isfinite(delta_phi)]
            if finite_delta.size:
                amplitude = float(
                    np.nanpercentile(finite_delta, 95.0)
                    - np.nanpercentile(finite_delta, 5.0)
                )
            else:
                amplitude = math.nan
            step_score, step_position, step_contrast = _single_step_profile_score(
                delta_phi,
                profile_axis,
                settings.focusing_axial_margin,
            )
            profile_supported = bool(
                math.isfinite(amplitude)
                and amplitude >= settings.front_profile_min_amplitude
                and math.isfinite(step_score)
                and step_score >= settings.front_profile_score_threshold
            )
            metrics.update(
                {
                    "porosity_front_profile_signal_amplitude": safe_float(amplitude),
                    "porosity_front_step_score": safe_float(step_score),
                    "porosity_front_position_over_L": safe_float(step_position),
                    "porosity_front_upstream_minus_downstream_change": safe_float(step_contrast),
                    "porosity_front_profile_supported": profile_supported,
                }
            )

    return (
        focusing,
        phi,
        profile_axis,
        profile_times,
        focusing_summary,
        metrics,
    )

def build_processed_series(
    params: Mapping[str, np.ndarray],
    spatial: Mapping[str, np.ndarray] | None,
    settings: Settings,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray]:
    time = np.asarray(params["time"], dtype=float)
    conductivity, q_at_time, formula = derive_conductivity(
        params, spatial, settings.conductivity_mode
    )
    smooth_log_k, window = smooth_log_conductivity(
        conductivity, settings.smooth_window, settings.smooth_polyorder
    )
    smooth_k = np.exp(smooth_log_k)

    primary = np.asarray(params["primary_solid_fraction"], dtype=float)
    secondary = np.asarray(params["secondary_solid_fraction"], dtype=float)
    porosity = np.asarray(params["porosity"], dtype=float)

    primary_removed = primary[0] - primary
    primary_removed_normalized = np.divide(
        primary_removed,
        primary[0],
        out=np.full_like(primary_removed, np.nan),
        where=abs(primary[0]) > np.finfo(float).eps,
    )
    secondary_added = secondary - secondary[0]
    porosity_change = porosity - porosity[0]
    replacement_ratio = np.divide(
        secondary_added,
        primary_removed,
        out=np.full_like(primary_removed, np.nan),
        where=primary_removed > 1e-14,
    )
    solid_balance_residual = porosity_change - (primary_removed - secondary_added)

    if np.all(np.isfinite(q_at_time)):
        throughput = cumulative_trapezoid(np.abs(q_at_time), time)
    else:
        throughput = np.full_like(time, np.nan)

    if time.size >= 2:
        log_k_rate = np.gradient(smooth_log_k, time)
    else:
        log_k_rate = np.zeros_like(time)

    processed: dict[str, np.ndarray] = {
        **{key: np.asarray(value, dtype=float) for key, value in params.items()},
        "relative_conductivity": conductivity,
        "relative_conductivity_smoothed": smooth_k,
        "log_relative_conductivity_smoothed": smooth_log_k,
        "d_log_conductivity_dt": log_k_rate,
        "Q_in_interpolated": q_at_time,
        "cumulative_injected_volume": throughput,
        "primary_removed_fraction_since_first_record": primary_removed,
        "primary_removed_fraction_normalized": primary_removed_normalized,
        "secondary_added_fraction_since_first_record": secondary_added,
        "porosity_change_since_first_record": porosity_change,
        "secondary_to_primary_removed_ratio": replacement_ratio,
        "solid_volume_balance_residual": solid_balance_residual,
    }

    metrics, peaks, troughs = conductivity_metrics(
        time, conductivity, smooth_log_k, settings
    )
    metrics.update(
        {
            "conductivity_formula": formula,
            "savgol_window_used": window,
            "time_initial": float(time[0]),
            "time_final": float(time[-1]),
            "number_of_saved_steps": int(time.size),
            "porosity_initial": float(porosity[0]),
            "porosity_final": float(porosity[-1]),
            "porosity_change": float(porosity_change[-1]),
            "primary_solid_fraction_initial": float(primary[0]),
            "primary_solid_fraction_final": float(primary[-1]),
            "secondary_solid_fraction_initial": float(secondary[0]),
            "secondary_solid_fraction_final": float(secondary[-1]),
            "primary_removed_fraction_final": float(primary_removed[-1]),
            "primary_removed_volume_fraction_final": float(primary_removed[-1]),
            "primary_removed_normalized_final": float(primary_removed_normalized[-1]),
            "secondary_added_fraction_final": float(secondary_added[-1]),
            "replaced_volume_fraction_final": float(secondary[-1]),
            "replaced_volume_added_fraction_final": float(secondary_added[-1]),
            "secondary_to_primary_removed_ratio_final": safe_float(replacement_ratio[-1]),
            "solid_volume_balance_residual_max_abs": safe_float(
                np.nanmax(np.abs(solid_balance_residual))
            ),
            "cumulative_injected_volume_final": safe_float(throughput[-1]),
        }
    )
    return processed, metrics, peaks, troughs


def classify_operational_regime(
    metrics: Mapping[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """Assign exactly one of the eight manuscript regimes.

    The decision tree separates hydraulic response, morphology, and pathway
    dynamics while returning a single manuscript label. Missing diagnostics
    do not create extra labels; instead they reduce classification confidence
    and are recorded in the rationale.
    """
    trajectory = str(metrics.get("hydraulic_trajectory", "unknown"))
    terminal = str(metrics.get("hydraulic_terminal_state", "unknown"))
    spatial_available = bool(metrics.get("spatial_metrics_available", False))
    profiles_available = bool(metrics.get("flow_focusing_profiles_available", False))

    k_final = safe_float(metrics.get("conductivity_final_over_initial"))
    oscillatory = trajectory == "oscillatory"
    nonmonotonic = trajectory in {"oscillatory", "nonmonotonic"}
    net_closure = terminal == "closure"
    net_opening = terminal == "opening"
    net_balanced = terminal == "balanced"

    xa_max = safe_float(metrics.get("x_A50_over_L_max_smoothed"))
    if not math.isfinite(xa_max):
        xa_max = safe_float(metrics.get("x_A50_over_L_max"))
    xa_final = safe_float(metrics.get("x_A50_over_L_final_smoothed"))
    if not math.isfinite(xa_final):
        xa_final = safe_float(metrics.get("x_A50_over_L_final"))
    xa_nonmonotonicity = safe_float(metrics.get("x_A50_nonmonotonicity_index"))
    xa_persistence = safe_float(metrics.get("x_A50_final_to_max_ratio"))

    deep = bool(math.isfinite(xa_max) and xa_max >= settings.deep_penetration_threshold)
    throughgoing = bool(math.isfinite(xa_max) and xa_max >= settings.breakthrough_threshold)
    a50_monotonic = bool(
        math.isfinite(xa_nonmonotonicity)
        and xa_nonmonotonicity <= settings.front_nonmonotonicity_threshold
    )
    a50_persistent = bool(
        math.isfinite(xa_persistence)
        and xa_persistence >= settings.front_persistence_threshold
    )

    focusing_score = math.nan
    focusing_source = "unavailable"
    strong_focusing: bool | None = None
    if profiles_available:
        focusing_score = safe_float(metrics.get("flow_focusing_late_mean"))
        focusing_source = "late_mean_slice_focusing_index"
        if math.isfinite(focusing_score):
            strong_focusing = focusing_score >= settings.focusing_threshold
    elif spatial_available:
        focusing_score = safe_float(metrics.get("max_edge_flow_fraction_late_mean"))
        focusing_source = "late_mean_max_edge_flow_fraction_proxy"
        if math.isfinite(focusing_score):
            strong_focusing = (
                focusing_score >= settings.dominant_edge_fraction_threshold
            )

    weak_focusing = strong_focusing is False

    n50_late = safe_float(metrics.get("flow_n50_late_mean"))
    multichannel_fraction = safe_float(
        metrics.get("flow_multichannel_fraction_x_late")
    )
    braided_evidence = bool(
        profiles_available
        and math.isfinite(n50_late)
        and math.isfinite(multichannel_fraction)
        and n50_late >= settings.braided_n50_threshold
        and multichannel_fraction >= settings.braided_slice_fraction_threshold
    )

    q20_final = safe_float(metrics.get("x_Q20_over_L_final_smoothed"))
    q20_max = safe_float(metrics.get("x_Q20_over_L_max_smoothed"))
    q20_final_present = bool(metrics.get("Q20_final_present", False))
    q20_presence_fraction = safe_float(metrics.get("Q20_presence_fraction"))
    event_count = int(metrics.get("x_Q20_completed_retreat_recovery_events", 0) or 0)
    reorganization_strength = safe_float(
        metrics.get("x_Q20_completed_reorganization_strength_over_L")
    )
    q20_loss_events = int(metrics.get("Q20_loss_events", 0) or 0)
    q20_reappearance_events = int(metrics.get("Q20_reappearance_events", 0) or 0)
    temporary_q20_loss = q20_loss_events > 0 and q20_reappearance_events > 0
    completed_reorganization = event_count > 0
    switching_evidence = completed_reorganization or temporary_q20_loss

    flow_deep_final = bool(
        q20_final_present
        and math.isfinite(q20_final)
        and q20_final >= settings.deep_penetration_threshold
    )
    flow_throughgoing_final = bool(
        q20_final_present
        and math.isfinite(q20_final)
        and q20_final >= settings.breakthrough_threshold
    )
    flow_ever_throughgoing = bool(
        math.isfinite(q20_max) and q20_max >= settings.breakthrough_threshold
    )
    terminal_q20_loss = bool(
        math.isfinite(q20_presence_fraction)
        and q20_presence_fraction > 0.2
        and not q20_final_present
    )
    ending_drawdown = safe_float(metrics.get("x_Q20_ending_drawdown_from_max_over_L"))
    terminal_q20_retreat = bool(
        math.isfinite(ending_drawdown)
        and ending_drawdown >= settings.front_retreat_prominence
        and not completed_reorganization
    )
    hydraulic_failure = bool(
        (math.isfinite(k_final) and k_final <= settings.clogged_conductivity_threshold)
        or terminal_q20_loss
        or (flow_ever_throughgoing and not q20_final_present)
    )

    front_profile_supported = bool(
        metrics.get("porosity_front_profile_supported", False)
    )
    front_kinematic_evidence = bool(deep and a50_monotonic and a50_persistent)
    front_candidate = bool(
        front_kinematic_evidence
        and weak_focusing
        and not braided_evidence
    )
    strong_front_candidate = bool(front_candidate and front_profile_supported)

    # Exact eight-regime decision tree. A clearly step-like porosity profile can
    # identify a front even if its conductivity is non-monotonic. Otherwise,
    # sustained conductivity oscillations are classified by focusing and Q20
    # retreat/re-advance before terminal hydraulic fate is considered.
    fit = "exact"
    rule = ""
    if oscillatory:
        if strong_front_candidate:
            regime = "front_propagating_replacement"
            family = "front_replacement"
            rule = "oscillatory response but independent front-profile evidence is strong"
        elif weak_focusing:
            regime = "distributed_oscillatory_competition"
            family = "distributed_competition"
            rule = "sustained conductivity oscillations with weak/distributed flow focusing"
        elif switching_evidence:
            regime = "pathway_switching_exploration"
            family = "exploration"
            rule = "sustained oscillations with focused flow and Q20 retreat--re-advance"
        elif strong_focusing is True:
            regime = "tip_branching_exploration"
            family = "exploration"
            rule = "sustained oscillations with focused flow but no large axial Q20 retreat"
        elif switching_evidence:
            regime = "pathway_switching_exploration"
            family = "exploration"
            rule = "Q20 switching evidence is present but focusing data are unavailable"
            fit = "approximate"
        else:
            regime = "distributed_oscillatory_competition"
            family = "distributed_competition"
            rule = "oscillatory response with unresolved focusing; distributed class used conservatively"
            fit = "fallback"
    elif net_closure:
        if not deep:
            regime = "compact_clogging"
            family = "clogging"
            rule = "net hydraulic closure before substantial A50 penetration"
        elif braided_evidence and flow_deep_final and q20_final_present:
            regime = "braided_channeling_with_net_closure"
            family = "braided_channeling"
            rule = "net closure with deep persistent flow and multiple N50 branches"
        elif front_candidate:
            regime = "front_propagating_replacement"
            family = "front_replacement"
            rule = "deep, persistent, weakly focused A50 advance despite net hydraulic closure"
            if not front_profile_supported:
                fit = "approximate"
        else:
            regime = "channelized_clogging"
            family = "clogging"
            rule = "deep reaction penetration with net closure and no persistent braided signature"
            if not hydraulic_failure and q20_final_present:
                fit = "approximate"
    elif net_opening:
        if front_candidate:
            regime = "front_propagating_replacement"
            family = "front_replacement"
            rule = "deep, monotonic, weakly focused A50 advance with net opening"
            if not front_profile_supported:
                fit = "approximate"
        elif strong_focusing is True and (throughgoing or flow_ever_throughgoing or flow_throughgoing_final):
            regime = "persistent_wormholing"
            family = "wormholing"
            rule = "net opening with a focused pathway reaching breakthrough"
        elif strong_focusing is True:
            regime = "persistent_wormholing"
            family = "wormholing"
            rule = "net opening with strong flow focusing but no recorded breakthrough"
            fit = "approximate"
        elif weak_focusing:
            regime = "front_propagating_replacement"
            family = "front_replacement"
            rule = "net opening is spatially distributed rather than wormhole-focused"
            fit = "approximate"
        else:
            regime = "persistent_wormholing"
            family = "wormholing"
            rule = "net opening with unresolved spatial data; wormholing is the closest class"
            fit = "fallback"
    else:  # balanced hydraulic response
        if front_candidate:
            regime = "front_propagating_replacement"
            family = "front_replacement"
            rule = "deep, monotonic, weakly focused replacement with balanced conductivity"
            if not front_profile_supported:
                fit = "approximate"
        elif strong_focusing is True and (throughgoing or flow_ever_throughgoing or flow_deep_final):
            regime = "persistent_wormholing"
            family = "wormholing"
            rule = "a persistent focused pathway is hydraulically stabilized near a plateau"
            fit = "approximate"
        else:
            regime = "front_propagating_replacement"
            family = "front_replacement"
            rule = "balanced non-oscillatory response without evidence for a dominant wormhole"
            fit = "fallback" if not deep else "approximate"

    if not spatial_available:
        fit = "fallback"
    elif not profiles_available and regime in {
        "braided_channeling_with_net_closure",
        "tip_branching_exploration",
        "distributed_oscillatory_competition",
        "front_propagating_replacement",
    } and fit == "exact":
        fit = "approximate"

    if regime not in MANUSCRIPT_REGIMES:
        raise RuntimeError(f"Unexpected manuscript regime: {regime}")

    if oscillatory:
        singurindy_projection = "competition_oscillatory"
    elif nonmonotonic:
        singurindy_projection = "competition_nonmonotonic"
    elif terminal == "closure":
        singurindy_projection = "precipitation_dominated_closure"
    elif terminal == "opening":
        singurindy_projection = "dissolution_dominated_opening"
    else:
        singurindy_projection = "balanced_or_passivated"

    if spatial_available and profiles_available:
        confidence = "high"
    elif spatial_available:
        confidence = "medium"
    else:
        confidence = "low"
    if fit == "approximate" and confidence == "high":
        confidence = "medium"
    elif fit == "fallback":
        confidence = "low"
    if regime == "pathway_switching_exploration" and event_count == 1:
        confidence = "medium" if confidence == "high" else confidence
    if regime == "front_propagating_replacement" and not front_profile_supported:
        confidence = "medium" if confidence == "high" else confidence

    rationale_parts = [
        f"rule={rule}",
        f"hydraulic trajectory={trajectory}",
        f"net hydraulic state={terminal}",
    ]
    if math.isfinite(k_final):
        rationale_parts.append(f"Kf/K0={k_final:.3g}")
    if math.isfinite(xa_max):
        rationale_parts.append(f"max x_A50/L={xa_max:.3g}")
    if math.isfinite(xa_nonmonotonicity):
        rationale_parts.append(f"A50 nonmonotonicity={xa_nonmonotonicity:.3g}")
    if math.isfinite(xa_persistence):
        rationale_parts.append(f"A50 final/max={xa_persistence:.3g}")
    if strong_focusing is not None:
        rationale_parts.append(
            f"focusing={'strong' if strong_focusing else 'weak'} "
            f"({focusing_score:.3g}, {focusing_source})"
        )
    if math.isfinite(n50_late):
        rationale_parts.append(f"late mean N50={n50_late:.3g}")
    if math.isfinite(multichannel_fraction):
        rationale_parts.append(
            f"late multichannel slice fraction={multichannel_fraction:.3g}"
        )
    rationale_parts.append(f"completed Q20 retreat-recovery events={event_count}")
    if math.isfinite(reorganization_strength):
        rationale_parts.append(
            f"Q20 reorganization strength/L={reorganization_strength:.3g}"
        )
    if terminal_q20_retreat:
        rationale_parts.append("Q20 ends in significant unrecovered retreat")
    if temporary_q20_loss:
        rationale_parts.append(
            f"Q20 criterion is lost and reappears "
            f"({q20_loss_events} losses, {q20_reappearance_events} reappearances)"
        )
    if front_profile_supported:
        rationale_parts.append("axial porosity profile supports a coherent front")
    if hydraulic_failure:
        rationale_parts.append("terminal hydraulic-access loss criterion is met")

    return {
        "manuscript_regime": regime,
        # Backward-compatible alias used by existing plots and tables.
        "operational_regime": regime,
        "operational_regime_with_hydraulic_fate": f"{regime}__net_{terminal}",
        "regime_family": family,
        "regime_definition_fit": fit,
        "singurindy_compatible_projection": singurindy_projection,
        "classification_confidence": confidence,
        "classification_rationale": "; ".join(rationale_parts),
        "deep_dissolution_penetration": deep,
        "throughgoing_dissolution_penetration": throughgoing,
        "A50_front_monotonic": a50_monotonic,
        "A50_front_persistent": a50_persistent,
        "front_kinematic_evidence": front_kinematic_evidence,
        "front_like_candidate": front_candidate,
        "front_profile_supported": front_profile_supported,
        "strong_flow_focusing": strong_focusing,
        "flow_focusing_score": focusing_score,
        "flow_focusing_score_source": focusing_source,
        "braided_flow_evidence": braided_evidence,
        "flow_Q20_deep_at_final_time": flow_deep_final,
        "flow_Q20_throughgoing_at_final_time": flow_throughgoing_final,
        "flow_Q20_ever_throughgoing": flow_ever_throughgoing,
        "completed_Q20_reorganization": completed_reorganization,
        "temporary_Q20_loss_and_reappearance": temporary_q20_loss,
        "terminal_Q20_loss": terminal_q20_loss,
        "terminal_Q20_retreat": terminal_q20_retreat,
        "terminal_hydraulic_failure": hydraulic_failure,
        "classification_threshold_deep_penetration": settings.deep_penetration_threshold,
        "classification_threshold_breakthrough": settings.breakthrough_threshold,
        "classification_threshold_clogged_conductivity": settings.clogged_conductivity_threshold,
        "classification_threshold_focusing": settings.focusing_threshold,
        "classification_threshold_dominant_edge_proxy": settings.dominant_edge_fraction_threshold,
        "classification_threshold_Q20_retreat": settings.front_retreat_prominence,
        "classification_threshold_A50_nonmonotonicity": settings.front_nonmonotonicity_threshold,
        "classification_threshold_A50_persistence": settings.front_persistence_threshold,
        "classification_threshold_braided_N50": settings.braided_n50_threshold,
        "classification_threshold_braided_slice_fraction": settings.braided_slice_fraction_threshold,
        "classification_threshold_front_profile_score": settings.front_profile_score_threshold,
        "classification_threshold_front_profile_amplitude": settings.front_profile_min_amplitude,
    }

def x_values_and_label(
    processed: Mapping[str, np.ndarray],
    mode: str,
) -> tuple[np.ndarray, str]:
    time = np.asarray(processed["time"], dtype=float)
    if mode == "normalized-time":
        span = time[-1] - time[0]
        if span <= 0:
            return np.zeros_like(time), r"normalized time"
        return (time - time[0]) / span, r"$(t-t_0)/(t_f-t_0)$"
    if mode == "throughput":
        throughput = np.asarray(processed["cumulative_injected_volume"], dtype=float)
        if np.all(np.isfinite(throughput)):
            return throughput, r"cumulative injected volume"
        warnings.warn("Q_in unavailable; falling back to time on the x-axis")
    return time, r"time"


def spatial_x_values_and_label(
    spatial: Mapping[str, np.ndarray],
    mode: str,
) -> tuple[np.ndarray, str]:
    time = np.asarray(spatial["time"], dtype=float)
    if mode == "normalized-time":
        span = time[-1] - time[0]
        return (
            np.zeros_like(time) if span <= 0 else (time - time[0]) / span,
            r"$(t-t_0)/(t_f-t_0)$",
        )
    if mode == "throughput":
        q = np.asarray(spatial["Q_in"], dtype=float)
        if np.all(np.isfinite(q)) and np.all(q >= 0):
            return cumulative_trapezoid(q, time), r"cumulative injected volume"
        warnings.warn("Q_in unavailable in spatial data; falling back to time")
    return time, r"time"


def save_csv(path: Path, table: Mapping[str, np.ndarray]) -> None:
    keys = list(table)
    lengths = {np.asarray(table[key]).size for key in keys}
    if len(lengths) != 1:
        raise ValueError(f"Cannot save {path}: columns have different lengths")
    matrix = np.column_stack([np.asarray(table[key]) for key in keys])
    np.savetxt(path, matrix, delimiter=",", header=",".join(keys), comments="")


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_figure(fig: plt.Figure, base_path: Path, settings: Settings) -> None:
    for extension in settings.formats:
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = settings.dpi
        fig.savefig(base_path.with_suffix(f".{extension}"), **kwargs)
    plt.close(fig)


def plot_conductivity(
    result: RunResult,
    peaks: np.ndarray,
    troughs: np.ndarray,
    settings: Settings,
) -> None:
    x, xlabel = x_values_and_label(result.processed, settings.x_axis)
    raw = result.processed["relative_conductivity"]
    smooth = result.processed["relative_conductivity_smoothed"]
    derivative = result.processed["d_log_conductivity_dt"]

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True, layout="constrained")
    axes[0].plot(x, raw, linewidth=1.0, alpha=0.55, label="raw")
    axes[0].plot(x, smooth, linewidth=2.0, label="smoothed")
    if peaks.size:
        axes[0].scatter(x[peaks], smooth[peaks], marker="^", zorder=3, label="peaks")
    if troughs.size:
        axes[0].scatter(x[troughs], smooth[troughs], marker="v", zorder=3, label="troughs")
    axes[0].axhline(1.0, linewidth=0.8, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$K/K_0$")
    axes[0].set_title(
        f"{result.name}: {result.metrics['hydraulic_response'].replace('_', ' ')}"
    )
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].plot(x, derivative, linewidth=1.5)
    axes[1].axhline(0.0, linewidth=0.8, linestyle="--")
    axes[1].set_ylabel(r"$d\ln K/dt$")
    axes[1].set_xlabel(xlabel)
    axes[1].grid(True, alpha=0.25)

    text = (
        rf"$K_f/K_0={result.metrics['conductivity_final_over_initial']:.3g}$" "\n"
        rf"$O_K={result.metrics['conductivity_oscillation_index']:.3f}$" "\n"
        rf"$N_{{ext}}={result.metrics['conductivity_significant_extrema']}$"
    )
    axes[0].text(
        0.02,
        0.03,
        text,
        transform=axes[0].transAxes,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    save_figure(fig, result.output_dir / "conductivity_diagnostics", settings)


def plot_spatial(result: RunResult, settings: Settings) -> None:
    if result.processed_spatial is None:
        return
    spatial = result.processed_spatial
    x, xlabel = spatial_x_values_and_label(spatial, settings.x_axis)

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 10.5), sharex=True, layout="constrained")
    axes[0].plot(x, spatial["x_A50_over_L"], linewidth=2.0, label=r"$x_{A50}/L$")
    axes[0].plot(
        x,
        spatial["x_Q20_over_L"],
        linewidth=1.0,
        alpha=0.45,
        label=r"raw $x_{Q20}/L$",
    )
    axes[0].plot(
        x,
        spatial["x_Q20_over_L_smoothed"],
        linewidth=2.0,
        label=r"smoothed $x_{Q20}/L$",
    )
    events = np.flatnonzero(spatial["x_Q20_reorganization_event"] > 0.5)
    if events.size:
        axes[0].scatter(
            x[events],
            spatial["x_Q20_over_L_smoothed"][events],
            marker="v",
            zorder=4,
            label="completed retreat--re-advance",
        )
    axes[0].axhline(settings.breakthrough_threshold, linestyle="--", linewidth=0.8)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("axial penetration")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        x,
        spatial["x_Q20_drawdown_from_running_max"],
        linewidth=1.8,
        label="drawdown from previous reach",
    )
    axes[1].plot(
        x,
        spatial["x_Q20_cumulative_retreat"],
        linewidth=1.8,
        label="cumulative retreat",
    )
    axes[1].axhline(
        settings.front_retreat_prominence,
        linestyle="--",
        linewidth=0.8,
        label="retreat threshold",
    )
    axes[1].set_ylabel(r"distance$/L$")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, spatial["max_abs_q_over_Qin"], linewidth=2.0)
    axes[2].axhline(
        settings.dominant_edge_fraction_threshold,
        linestyle="--",
        linewidth=0.8,
        label="dominant-edge proxy threshold",
    )
    axes[2].set_ylabel(r"$\max |q_e|/Q_{in}$")
    axes[2].legend(frameon=False)
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(x, spatial["n_A50_grains"], linewidth=1.8, label="A50 grains")
    axes[3].plot(x, spatial["n_Q20_edges"], linewidth=1.8, label="Q20 edges")
    axes[3].set_ylabel("count")
    axes[3].set_xlabel(xlabel)
    axes[3].legend(frameon=False)
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(
        f"{result.name}: {result.metrics.get('manuscript_regime', 'unclassified').replace('_', ' ')}"
    )
    save_figure(fig, result.output_dir / "spatial_metrics", settings)


def plot_regime_diagnostics(result: RunResult, settings: Settings) -> None:
    """Plot the four diagnostics used by the operational classification."""
    x_main, xlabel_main = x_values_and_label(result.processed, settings.x_axis)
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 11.0), layout="constrained")

    axes[0].plot(
        x_main,
        result.processed["relative_conductivity"],
        linewidth=1.0,
        alpha=0.45,
        label="raw",
    )
    axes[0].plot(
        x_main,
        result.processed["relative_conductivity_smoothed"],
        linewidth=2.0,
        label="smoothed",
    )
    axes[0].axhline(1.0, linestyle="--", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$K/K_0$")
    axes[0].legend(frameon=False)
    axes[0].grid(True, which="both", alpha=0.25)

    if result.processed_spatial is not None:
        spatial = result.processed_spatial
        x_spatial, xlabel_spatial = spatial_x_values_and_label(spatial, settings.x_axis)
        axes[1].plot(x_spatial, spatial["x_A50_over_L"], linewidth=2.0, label=r"$x_{A50}/L$")
        axes[1].plot(
            x_spatial,
            spatial["x_Q20_over_L_smoothed"],
            linewidth=2.0,
            label=r"$x_{Q20}/L$",
        )
        events = np.flatnonzero(spatial["x_Q20_reorganization_event"] > 0.5)
        if events.size:
            axes[1].scatter(
                x_spatial[events],
                spatial["x_Q20_over_L_smoothed"][events],
                marker="v",
                zorder=4,
                label="Q20 reorganization event",
            )
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].set_xlabel(xlabel_spatial)
        axes[1].set_ylabel("axial penetration")
        axes[1].legend(frameon=False, ncol=2)
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(
            x_spatial,
            spatial["x_Q20_drawdown_from_running_max"],
            linewidth=2.0,
            label="Q20 drawdown",
        )
        axes[2].plot(
            x_spatial,
            spatial["x_Q20_cumulative_retreat"],
            linewidth=1.8,
            label="cumulative retreat",
        )
        axes[2].axhline(settings.front_retreat_prominence, linestyle="--", linewidth=0.8)
        axes[2].set_xlabel(xlabel_spatial)
        axes[2].set_ylabel(r"distance$/L$")
        axes[2].legend(frameon=False)
        axes[2].grid(True, alpha=0.25)
    else:
        for index in (1, 2):
            axes[index].text(0.5, 0.5, "spatial metrics unavailable", ha="center", va="center")
            axes[index].set_axis_off()

    if result.focusing_summary is not None:
        summary = result.focusing_summary
        coordinate = summary["profile_coordinate"]
        coordinate_label = "profile time" if result.profile_times is not None else "profile snapshot index"
        axes[3].plot(
            coordinate,
            summary["flow_focusing_mean"],
            linewidth=2.0,
            label="central mean F",
        )
        axes[3].plot(
            coordinate,
            summary["flow_focusing_p90"],
            linewidth=1.5,
            label="central 90th percentile F",
        )
        axes[3].axhline(settings.focusing_threshold, linestyle="--", linewidth=0.8)
        axes[3].set_ylim(-0.02, 1.05)
        axes[3].set_xlabel(coordinate_label)
        axes[3].set_ylabel("flow focusing")
        axes[3].legend(frameon=False)
        axes[3].grid(True, alpha=0.25)
    elif result.processed_spatial is not None:
        spatial = result.processed_spatial
        x_spatial, xlabel_spatial = spatial_x_values_and_label(spatial, settings.x_axis)
        axes[3].plot(x_spatial, spatial["max_abs_q_over_Qin"], linewidth=2.0)
        axes[3].axhline(
            settings.dominant_edge_fraction_threshold,
            linestyle="--",
            linewidth=0.8,
        )
        axes[3].set_xlabel(xlabel_spatial)
        axes[3].set_ylabel("dominant-edge proxy")
        axes[3].grid(True, alpha=0.25)
    else:
        axes[3].text(0.5, 0.5, "focusing data unavailable", ha="center", va="center")
        axes[3].set_axis_off()

    title = result.metrics.get("operational_regime", "unclassified").replace("_", " ")
    confidence = result.metrics.get("classification_confidence", "unknown")
    fig.suptitle(f"{result.name}: {title} ({confidence} confidence)")
    save_figure(fig, result.output_dir / "regime_diagnostics", settings)


def plot_solid_evolution(result: RunResult, settings: Settings) -> None:
    x, xlabel = x_values_and_label(result.processed, settings.x_axis)
    processed = result.processed

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True, layout="constrained")
    axes[0].plot(
        x,
        processed["primary_removed_fraction_since_first_record"],
        linewidth=2.0,
        label="primary removed",
    )
    axes[0].plot(
        x,
        processed["secondary_added_fraction_since_first_record"],
        linewidth=2.0,
        label="secondary added",
    )
    axes[0].set_ylabel("solid volume fraction")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, processed["porosity"], linewidth=2.0, label="porosity")
    axes[1].plot(
        x,
        processed["primary_solid_fraction"],
        linewidth=1.5,
        label="primary solid",
    )
    axes[1].plot(
        x,
        processed["secondary_solid_fraction"],
        linewidth=1.5,
        label="secondary solid",
    )
    axes[1].set_ylabel("volume fraction")
    axes[1].set_xlabel(xlabel)
    axes[1].legend(frameon=False, ncol=3)
    axes[1].grid(True, alpha=0.25)

    save_figure(fig, result.output_dir / "solid_and_porosity_evolution", settings)


def select_snapshot_indices(number: int, maximum: int = 5) -> np.ndarray:
    if number <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, number - 1, min(maximum, number), dtype=int))


def plot_profile_diagnostics(result: RunResult, settings: Settings) -> None:
    if result.profile_axis is None:
        return
    x = result.profile_axis

    if result.focusing_profiles is not None:
        data = result.focusing_profiles
        selected = select_snapshot_indices(data.shape[0])
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.5), layout="constrained")

        if result.profile_times is not None:
            profile_coordinate = result.profile_times
            y0 = float(profile_coordinate[0])
            y1 = float(profile_coordinate[-1]) if profile_coordinate.size > 1 else y0 + 1.0
            coordinate_label = "profile time"
        else:
            profile_coordinate = np.arange(data.shape[0], dtype=float)
            y0 = 0.0
            y1 = float(max(data.shape[0] - 1, 1))
            coordinate_label = "profile snapshot index"

        image = axes[0].imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=(0.0, 1.0, y0, y1),
            vmin=0.0,
            vmax=1.0,
        )
        axes[0].set_ylabel(coordinate_label)
        axes[0].set_xlabel(r"$x/L$")
        axes[0].set_title("flow-focusing profile")
        fig.colorbar(image, ax=axes[0], label=r"$F=1-2N_{50}/N$")

        for index in selected:
            if result.profile_times is not None:
                label = f"t={profile_coordinate[index]:.3g}"
            else:
                label = f"snapshot {index}"
            axes[1].plot(x, data[index], linewidth=1.6, label=label)
        axes[1].axhline(
            settings.focusing_threshold,
            linestyle="--",
            linewidth=0.8,
            label="classification threshold",
        )
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_xlabel(r"$x/L$")
        axes[1].set_ylabel("flow-focusing index")
        axes[1].legend(frameon=False, ncol=2, fontsize="small")
        axes[1].grid(True, alpha=0.25)

        if result.focusing_summary is not None:
            summary = result.focusing_summary
            axes[2].plot(
                profile_coordinate,
                summary["flow_focusing_mean"],
                linewidth=1.8,
                label="central mean",
            )
            axes[2].plot(
                profile_coordinate,
                summary["flow_focusing_p90"],
                linewidth=1.5,
                label="central 90th percentile",
            )
            axes[2].plot(
                profile_coordinate,
                summary["flow_focusing_fraction_x_above_local_threshold"],
                linewidth=1.5,
                label="fraction of x strongly focused",
            )
            axes[2].axhline(
                settings.focusing_threshold,
                linestyle="--",
                linewidth=0.8,
            )
            axes[2].set_ylim(-0.05, 1.05)
            axes[2].set_ylabel("profile summary")
            axes[2].set_xlabel(coordinate_label)
            axes[2].legend(frameon=False, ncol=2, fontsize="small")
            axes[2].grid(True, alpha=0.25)

        save_figure(fig, result.output_dir / "flow_focusing_profiles", settings)

    if result.porosity_profiles is not None:
        data = result.porosity_profiles
        selected = select_snapshot_indices(data.shape[0])
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), layout="constrained")
        image = axes[0].imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=(0.0, 1.0, 0.0, max(data.shape[0] - 1, 1)),
        )
        axes[0].set_ylabel("snapshot index")
        axes[0].set_xlabel(r"$x/L$")
        axes[0].set_title("porosity profile")
        fig.colorbar(image, ax=axes[0], label=r"$\phi$")

        for index in selected:
            axes[1].plot(x, data[index], linewidth=1.6, label=f"snapshot {index}")
        axes[1].set_xlabel(r"$x/L$")
        axes[1].set_ylabel(r"$\phi$")
        axes[1].legend(frameon=False, ncol=2)
        axes[1].grid(True, alpha=0.25)
        save_figure(fig, result.output_dir / "porosity_profiles", settings)


def write_metric_definitions(path: Path) -> None:
    text = """Key post-processing definitions
===============================

Relative conductivity
---------------------
Constant flow: K/K0 = dp0/dp.
If Q_in is available: K/K0 = (Q_in/dp)/(Q_in,0/dp0).

Conductivity oscillation index
------------------------------
O_K = 1 - |ln(K_f/K_0)| / sum_i |Delta ln(K/K0)|,
calculated on the smoothed log-conductivity trajectory. O_K=0 for a
perfectly monotonic curve. Significant extrema are detected independently
with a prominence threshold; only repeated significant reversals are called
oscillatory.

A50 penetration and front kinematics
------------------------------------
x_A50/L is the furthest normalized grain-centroid position among grains that
have lost at least 50% of their initial primary-mineral volume. The script
also reports A50 total variation, cumulative retreat, nonmonotonicity, and the
final/max front-position ratio. A front candidate advances deeply, remains
mostly monotonic, and does not retreat substantially before the final state.

Q20 penetration and pathway reorganization
-------------------------------------------
x_Q20/L is the furthest downstream endpoint of an edge carrying at least 20%
of total inlet flow. A completed reorganization event is a significant retreat
of the smoothed x_Q20/L front followed by re-advance. Cumulative retreat and
the largest drawdown from the previously reached position are also reported.
A terminal retreat is kept separate from a completed retreat--re-advance cycle.

Slice flow focusing and braiding
--------------------------------
F(x) = 1 - 2 N50(x)/N(x), where N50 is the minimum number of edges carrying
50% of flow through a slice and N is the total number of slice-crossing edges.
F approaches one for strongly focused flow. Braided-flow evidence requires
that late-time N50 is at least the configured braided threshold over a
specified fraction of the sample; this distinguishes several persistent
important branches from a single wormhole.

Porosity-profile front support
------------------------------
When profiles_phi.txt contains at least two snapshots, the final-minus-initial
axial porosity change is fitted by two plateaus separated by one axial split.
The reported score is the fraction of profile variance explained by that
single step. It is optional supporting evidence only: a small porosity signal
is treated as unavailable because replacement can proceed with little net
porosity change.

Eight-regime manuscript classification
---------------------------------------
Every run receives exactly one label:
  compact_clogging
  channelized_clogging
  braided_channeling_with_net_closure
  persistent_wormholing
  tip_branching_exploration
  pathway_switching_exploration
  distributed_oscillatory_competition
  front_propagating_replacement

The decision tree uses conductivity trajectory and net change, A50 penetration
and persistence, Q20 retreat--re-advance, late flow focusing, late N50 branch
multiplicity, and optional porosity-front support. The summary records the
rule, fit (exact/approximate/fallback), confidence, and all threshold values.

Phase diagrams and replaced volume
----------------------------------
When positive Da_eff and cd_in are available, the batch post-processor places
runs at x=1/Da_eff and y=1/cd_in. The three-regime projection assigns all
non-monotonic and hydraulically balanced trajectories to competition, while
monotonic net opening and closure are assigned to dissolution- and
precipitation-dominated responses, respectively. The detailed map uses the
eight manuscript labels. Repeated simulations at one parameter point are
combined by majority label; mixed outcomes are marked, and the replaced-volume
heatmap uses their mean value. By default, replaced volume is the final
secondary-mineral volume fraction, matching the fifth column of params.txt.

Caution
-------
The Q20 metric detects axial loss and reconstruction of downstream hydraulic
access. It cannot detect a purely lateral switch between two pathways that
reach the same x position. N50 counts important edges per slice but does not
track their identities. Regime boundaries should therefore be checked against
representative network images and tested for threshold sensitivity.
"""
    path.write_text(text, encoding="utf-8")

_PHASE_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PHASE_DAEFF_PATTERNS = (
    re.compile(
        rf"(?i)(?:da[_\-\s]*eff|daeff|effective[_\-\s]*da)"
        rf"\s*(?:=|:)?\s*({_PHASE_FLOAT_RE})"
    ),
)
_PHASE_CDIN_PATTERNS = (
    re.compile(
        rf"(?i)(?:c[_\-\s]*d[_\-\s]*in|cd[_\-\s]*in|cdin|d[_\-\s]*in)"
        rf"\s*(?:=|:)?\s*({_PHASE_FLOAT_RE})"
    ),
)

_PHASE_METADATA_CANDIDATE_NAMES = (
    "phase_metadata.csv",
    "phase_metadata.tsv",
    "phase_metadata.txt",
    "new_singurindy.txt",
)


def _normalise_phase_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _phase_tokens(line: str, delimiter: str) -> list[str]:
    if delimiter == "comma":
        return next(csv.reader([line]))
    if delimiter == "tab":
        return next(csv.reader([line], delimiter="\t"))
    return re.split(r"\s+", line.strip())


def load_phase_metadata_table(path: Path | None) -> list[dict[str, Any]]:
    """Read a CSV, TSV, or whitespace table containing run, Da_eff, and cd_in.

    The parser searches for the header instead of assuming that it is the first
    line. This accepts the compact ``new_singurindy.txt`` format, in which a
    descriptive line precedes ``folder Da_eff cd_in ...``.
    """
    if path is None:
        return []
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Phase-metadata table not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index: int | None = None
    delimiter = "whitespace"
    header: list[str] = []
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate_delimiter = (
            "comma" if "," in line else "tab" if "\t" in line else "whitespace"
        )
        tokens = _phase_tokens(line, candidate_delimiter)
        keys = {_normalise_phase_key(token) for token in tokens}
        if keys & _PHASE_DAEFF_ALIASES and keys & _PHASE_CDIN_ALIASES:
            header_index = index
            delimiter = candidate_delimiter
            header = tokens
            break

    if header_index is None:
        raise ValueError(
            f"Could not find a header containing Da_eff and cd_in in {path}"
        )

    normalised = [_normalise_phase_key(token) for token in header]

    def first_index(aliases: set[str]) -> int | None:
        for idx, key in enumerate(normalised):
            if key in aliases:
                return idx
        return None

    selector_idx = first_index(_PHASE_SELECTOR_ALIASES)
    daeff_idx = first_index(_PHASE_DAEFF_ALIASES)
    cdin_idx = first_index(_PHASE_CDIN_ALIASES)
    if daeff_idx is None or cdin_idx is None:
        raise ValueError(f"Missing Da_eff or cd_in column in {path}")
    if selector_idx is None:
        raise ValueError(
            f"The phase-metadata table {path} also needs a run/folder/name/path column"
        )

    rows: list[dict[str, Any]] = []
    required_index = max(selector_idx, daeff_idx, cdin_idx)
    for line_number, raw_line in enumerate(
        lines[header_index + 1 :], header_index + 2
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = _phase_tokens(line, delimiter)
        if len(tokens) <= required_index:
            warnings.warn(
                f"Skipping short phase-metadata row {line_number} in {path}"
            )
            continue
        selector = tokens[selector_idx].strip()
        try:
            daeff = float(tokens[daeff_idx])
            cdin = float(tokens[cdin_idx])
        except ValueError:
            warnings.warn(
                f"Skipping non-numeric phase-metadata row {line_number} in {path}"
            )
            continue
        rows.append(
            {
                "selector": selector,
                "Da_eff": daeff,
                "cd_in": cdin,
                "source": f"metadata:{path.name}:{line_number}",
            }
        )
    return rows


def discover_phase_metadata_path(input_paths: Sequence[Path]) -> Path | None:
    """Find a likely phase-metadata table near the supplied result paths.

    The previous version silently fell back to values parsed from a common
    ancestor directory.  For directory layouts such as
    ``G5.00Daeff1.00/5/template`` this assigned every run the same phase
    coordinates.  We now preferentially discover an explicit metadata table
    and only use run-local automatic extraction as a fallback.
    """
    search_directories: list[Path] = []

    def add_directory(directory: Path) -> None:
        try:
            resolved = directory.expanduser().resolve()
        except OSError:
            return
        if resolved not in search_directories:
            search_directories.append(resolved)

    add_directory(Path.cwd())
    for raw_path in input_paths:
        path = raw_path.expanduser()
        if path.is_file():
            path = path.parent
        add_directory(path)
        current = path
        for _ in range(3):
            current = current.parent
            add_directory(current)

    candidates: list[tuple[int, int, Path]] = []
    for directory_rank, directory in enumerate(search_directories):
        for name_rank, filename in enumerate(_PHASE_METADATA_CANDIDATE_NAMES):
            candidate = directory / filename
            if not candidate.is_file():
                continue
            try:
                row_count = len(load_phase_metadata_table(candidate))
            except (OSError, ValueError):
                continue
            if row_count:
                # Prefer a table close to the supplied path, then a standard
                # filename, then the table with more usable rows.
                score = 10_000 - 100 * directory_rank - 10 * name_rank + row_count
                candidates.append((score, row_count, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = candidates[0][2]
    if len(candidates) > 1:
        alternatives = [str(item[2]) for item in candidates[1:4]]
        warnings.warn(
            "Several possible phase-metadata tables were found; using "
            f"{selected}. Other candidates: {', '.join(alternatives)}"
        )
    return selected


def _metadata_match_score(selector: str, run_dir: Path, run_name: str) -> int:
    selector_raw = selector.strip().replace("\\", "/").strip("/")
    selector_norm = selector_raw.lower()
    if not selector_norm:
        return -1

    path_norm = str(run_dir.resolve()).replace("\\", "/").lower().rstrip("/")
    candidates = {run_name.lower(), run_dir.name.lower()}
    candidates.update(part.lower() for part in run_dir.parts)

    if selector_norm == run_name.lower():
        return 120
    if selector_norm in candidates:
        return 100
    if "/" in selector_norm and path_norm.endswith("/" + selector_norm):
        return 110 + selector_norm.count("/")
    if path_norm.endswith("/" + selector_norm):
        return 90
    return -1


def _extract_phase_value(
    text: str,
    patterns: Sequence[re.Pattern[str]],
) -> float:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if math.isfinite(value):
                return value
    return math.nan


_CONFIG_PHASE_ASSIGNMENT_RE = re.compile(
    rf"^\s*(?P<key>Da[_\-\s]*eff|Daeff|c[_\-\s]*d[_\-\s]*in|cdin)"
    rf"\s*=\s*(?P<value>{_PHASE_FLOAT_RE})\s*(?:#.*)?$",
    flags=re.IGNORECASE,
)


def _canonical_config_phase_key(raw_key: str) -> str | None:
    """Map supported config variable spellings to the canonical output names."""
    key = _normalise_phase_key(raw_key)
    if key == "daeff":
        return "Da_eff"
    if key == "cdin":
        return "cd_in"
    return None


def _extract_last_phase_pair(text: str) -> tuple[float, float]:
    """Extract the last ``Daeff...cdin...`` pair embedded in arbitrary text.

    This is used only to *audit* the standalone values read from ``config.txt``;
    embedded values never override ``Da_eff = ...`` or ``cd_in = ...``.
    """
    pattern = re.compile(
        rf"Da[_\-\s]*eff\s*(?:=|:)?\s*(?P<da>{_PHASE_FLOAT_RE})"
        rf".*?c[_\-\s]*d[_\-\s]*in\s*(?:=|:)?\s*(?P<cd>{_PHASE_FLOAT_RE})",
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        # Folder names commonly use the compact spelling ``Daeff0.67cdin1.5``.
        compact = re.compile(
            rf"Daeff(?P<da>{_PHASE_FLOAT_RE}).*?cdin(?P<cd>{_PHASE_FLOAT_RE})",
            flags=re.IGNORECASE,
        )
        matches = list(compact.finditer(text))
    if not matches:
        return math.nan, math.nan
    match = matches[-1]
    try:
        return float(match.group("da")), float(match.group("cd"))
    except (TypeError, ValueError):
        return math.nan, math.nan


def _phase_values_close(a: float, b: float) -> bool:
    return bool(
        math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=1.0e-9, abs_tol=1.0e-12)
    )


def read_phase_config_record(
    run_dir: Path,
    config_filename: str = "config.txt",
) -> dict[str, Any]:
    """Read and audit phase parameters from the config beside ``params.txt``.

    The authoritative values are the standalone assignments ``Da_eff = ...``
    and ``cd_in = ...``.  Values embedded in ``dirname``, ``load_name``, or the
    filesystem path are retained only as consistency checks.  In particular,
    they can never silently replace the standalone assignments.
    """
    config_path = run_dir / config_filename
    empty: dict[str, Any] = {
        "config_Da_eff": math.nan,
        "config_cd_in": math.nan,
        "phase_config_path": str(config_path),
        "phase_config_exists": False,
        "phase_config_dirname": "",
        "phase_config_load_name": "",
        "phase_config_dirname_Da_eff": math.nan,
        "phase_config_dirname_cd_in": math.nan,
        "phase_source_path_Da_eff": math.nan,
        "phase_source_path_cd_in": math.nan,
        "phase_config_consistency": "missing",
        "phase_config_mismatch": False,
        "phase_parameter_source": f"missing:{config_path}",
    }
    if not config_path.is_file():
        return empty

    try:
        lines = config_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as exc:
        warnings.warn(f"Could not read phase parameters from {config_path}: {exc}")
        empty["phase_config_consistency"] = "unreadable"
        empty["phase_parameter_source"] = f"unreadable:{config_path}"
        return empty

    values: dict[str, float] = {}
    line_numbers: dict[str, int] = {}
    raw_fields: dict[str, str] = {}
    raw_assignment = re.compile(
        r"^\s*(?P<key>dirname|load_name)\s*=\s*(?P<value>.*?)\s*$",
        flags=re.IGNORECASE,
    )

    for line_number, raw_line in enumerate(lines, start=1):
        match = _CONFIG_PHASE_ASSIGNMENT_RE.match(raw_line)
        if match is not None:
            canonical_key = _canonical_config_phase_key(match.group("key"))
            if canonical_key is not None:
                try:
                    value = float(match.group("value"))
                except ValueError:
                    value = math.nan
                if math.isfinite(value):
                    if canonical_key in values and not math.isclose(
                        values[canonical_key], value,
                        rel_tol=1.0e-12, abs_tol=0.0,
                    ):
                        warnings.warn(
                            f"Multiple assignments to {canonical_key} in "
                            f"{config_path}; using the last value "
                            f"({value}, line {line_number})"
                        )
                    values[canonical_key] = value
                    line_numbers[canonical_key] = line_number

        raw_match = raw_assignment.match(raw_line)
        if raw_match is not None:
            raw_value = raw_match.group("value").strip().strip("'\"")
            raw_fields[raw_match.group("key").lower()] = raw_value

    daeff = values.get("Da_eff", math.nan)
    cdin = values.get("cd_in", math.nan)
    dirname_value = raw_fields.get("dirname", "")
    load_name_value = raw_fields.get("load_name", "")
    dirname_da, dirname_cd = _extract_last_phase_pair(dirname_value)
    path_da, path_cd = _extract_last_phase_pair(str(run_dir.resolve()))

    issues: list[str] = []
    if not math.isfinite(daeff):
        issues.append("Da_eff assignment missing")
    if not math.isfinite(cdin):
        issues.append("cd_in assignment missing")
    if math.isfinite(dirname_da) and math.isfinite(daeff) and not _phase_values_close(dirname_da, daeff):
        issues.append(
            f"dirname Daeff={dirname_da:g} differs from assignment {daeff:g}"
        )
    if math.isfinite(dirname_cd) and math.isfinite(cdin) and not _phase_values_close(dirname_cd, cdin):
        issues.append(
            f"dirname cdin={dirname_cd:g} differs from assignment {cdin:g}"
        )
    if math.isfinite(path_da) and math.isfinite(daeff) and not _phase_values_close(path_da, daeff):
        issues.append(
            f"folder Daeff={path_da:g} differs from assignment {daeff:g}"
        )
    if math.isfinite(path_cd) and math.isfinite(cdin) and not _phase_values_close(path_cd, cdin):
        issues.append(
            f"folder cdin={path_cd:g} differs from assignment {cdin:g}"
        )

    found = []
    if math.isfinite(daeff):
        found.append(f"Da_eff@L{line_numbers['Da_eff']}")
    if math.isfinite(cdin):
        found.append(f"cd_in@L{line_numbers['cd_in']}")
    detail = ",".join(found) if found else "no standalone assignments"
    consistency = "ok" if not issues else "; ".join(issues)

    return {
        "config_Da_eff": daeff,
        "config_cd_in": cdin,
        "phase_config_path": str(config_path.resolve()),
        "phase_config_exists": True,
        "phase_config_dirname": dirname_value,
        "phase_config_load_name": load_name_value,
        "phase_config_dirname_Da_eff": dirname_da,
        "phase_config_dirname_cd_in": dirname_cd,
        "phase_source_path_Da_eff": path_da,
        "phase_source_path_cd_in": path_cd,
        "phase_config_consistency": consistency,
        "phase_config_mismatch": bool(issues),
        "phase_parameter_source": f"config:{config_path.resolve()} ({detail})",
    }


def read_phase_parameters_from_config(
    run_dir: Path,
    config_filename: str = "config.txt",
) -> tuple[float, float, str]:
    """Backward-compatible wrapper around :func:`read_phase_config_record`."""
    record = read_phase_config_record(run_dir, config_filename)
    return (
        safe_float(record.get("config_Da_eff")),
        safe_float(record.get("config_cd_in")),
        str(record.get("phase_parameter_source", "unavailable")),
    )


def auto_detect_phase_parameters(run_dir: Path) -> tuple[float, float, str]:
    """Read phase coordinates from the config beside this run's params file."""
    return read_phase_parameters_from_config(run_dir)


def _best_metadata_values(
    run_dir: Path,
    run_name: str,
    metadata_rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[float, float, str]:
    """Return explicit metadata values for fallback use only."""
    if not metadata_rows:
        return math.nan, math.nan, ""
    scored: list[tuple[int, Mapping[str, Any]]] = []
    for row in metadata_rows:
        score = _metadata_match_score(
            str(row.get("selector", "")), run_dir, run_name
        )
        if score >= 0:
            scored.append((score, row))
    if not scored:
        return math.nan, math.nan, ""
    best_score = max(score for score, _ in scored)
    best_rows = [row for score, row in scored if score == best_score]
    row = best_rows[0]
    if len(best_rows) > 1:
        values = {
            (safe_float(item.get("Da_eff")), safe_float(item.get("cd_in")))
            for item in best_rows
        }
        if len(values) > 1:
            warnings.warn(
                f"Conflicting phase-metadata rows match {run_dir}; using the "
                "first best match as a fallback"
            )
    return (
        safe_float(row.get("Da_eff")),
        safe_float(row.get("cd_in")),
        str(row.get("source", "metadata")),
    )


def resolve_phase_parameters(
    run_dir: Path,
    run_name: str,
    metadata_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Resolve phase coordinates with the run-local config as authority.

    ``config.txt`` always wins.  An explicitly supplied metadata table is used
    only to fill a missing standalone assignment.  This avoids the previous
    failure mode in which an old ``--phase-metadata`` option silently replaced
    values that were present in the individual run configurations.
    """
    config = read_phase_config_record(run_dir)
    config_daeff = safe_float(config.get("config_Da_eff"))
    config_cdin = safe_float(config.get("config_cd_in"))
    metadata_daeff, metadata_cdin, metadata_source = _best_metadata_values(
        run_dir, run_name, metadata_rows
    )

    daeff = config_daeff if math.isfinite(config_daeff) else metadata_daeff
    cdin = config_cdin if math.isfinite(config_cdin) else metadata_cdin

    sources: list[str] = []
    if math.isfinite(config_daeff) or math.isfinite(config_cdin):
        sources.append(str(config.get("phase_parameter_source", "config")))
    if (
        (not math.isfinite(config_daeff) and math.isfinite(metadata_daeff))
        or (not math.isfinite(config_cdin) and math.isfinite(metadata_cdin))
    ):
        sources.append(f"metadata-fallback:{metadata_source}")

    metadata_conflicts: list[str] = []
    if math.isfinite(config_daeff) and math.isfinite(metadata_daeff) and not _phase_values_close(config_daeff, metadata_daeff):
        metadata_conflicts.append(
            f"metadata Daeff={metadata_daeff:g} ignored; config has {config_daeff:g}"
        )
    if math.isfinite(config_cdin) and math.isfinite(metadata_cdin) and not _phase_values_close(config_cdin, metadata_cdin):
        metadata_conflicts.append(
            f"metadata cdin={metadata_cdin:g} ignored; config has {config_cdin:g}"
        )

    valid = bool(
        math.isfinite(daeff) and daeff > 0.0
        and math.isfinite(cdin) and cdin > 0.0
    )
    inverse_daeff = 1.0 / daeff if valid else math.nan
    inverse_cdin = 1.0 / cdin if valid else math.nan

    resolved = dict(config)
    resolved.update(
        {
            "Da_eff": safe_float(daeff),
            "cd_in": safe_float(cdin),
            "inv_Da_eff": safe_float(inverse_daeff),
            "inv_cd_in": safe_float(inverse_cdin),
            "phase_parameters_available": valid,
            "phase_parameter_source": "+".join(dict.fromkeys(sources))
            if sources else str(config.get("phase_parameter_source", "unavailable")),
            "phase_metadata_Da_eff": metadata_daeff,
            "phase_metadata_cd_in": metadata_cdin,
            "phase_metadata_conflict": bool(metadata_conflicts),
            "phase_metadata_conflict_detail": "; ".join(metadata_conflicts),
        }
    )
    return resolved

def collapse_to_three_regimes(metrics: Mapping[str, Any]) -> str:
    """Project the detailed classification onto three broad regimes.

    The detailed regime is used first so that small conductivity wiggles do not
    turn an otherwise persistent wormhole into ``competition``.  Sustained
    oscillatory trajectories and all exploration classes remain in the
    competition sector.  Front-propagating replacement is classified from its
    net hydraulic response because it can open, close, or remain balanced.
    """
    full_regime = str(metrics.get("manuscript_regime", ""))
    trajectory = str(metrics.get("hydraulic_trajectory", "unknown"))
    terminal = str(metrics.get("hydraulic_terminal_state", "unknown"))

    if full_regime in {
        "tip_branching_exploration",
        "pathway_switching_exploration",
        "distributed_oscillatory_competition",
    }:
        return "competition"

    if trajectory == "oscillatory":
        return "competition"

    if full_regime == "persistent_wormholing":
        return "dissolution_dominated"

    if full_regime in {
        "compact_clogging",
        "channelized_clogging",
        "braided_channeling_with_net_closure",
    }:
        return "precipitation_dominated"

    if terminal == "opening":
        return "dissolution_dominated"
    if terminal == "closure":
        return "precipitation_dominated"
    return "competition"


def analyze_run(
    run_dir: Path,
    output_dir: Path,
    name: str,
    settings: Settings,
    phase_metadata_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[RunResult, np.ndarray, np.ndarray]:
    params = load_params(run_dir)
    spatial_raw = load_spatial(run_dir)
    processed, metrics, peaks, troughs = build_processed_series(
        params, spatial_raw, settings
    )

    spatial_summary, processed_spatial, _ = analyze_spatial_metrics(
        spatial_raw,
        settings,
    )
    metrics.update(spatial_summary)

    focusing: np.ndarray | None = None
    phi_profiles: np.ndarray | None = None
    profile_axis: np.ndarray | None = None
    profile_times: np.ndarray | None = None
    focusing_summary: dict[str, np.ndarray] | None = None
    if settings.include_profiles:
        (
            focusing,
            phi_profiles,
            profile_axis,
            profile_times,
            focusing_summary,
            profile_metrics,
        ) = process_profiles(run_dir, settings)
        metrics.update(profile_metrics)
    else:
        metrics.update(
            {
                "flow_focusing_profiles_available": False,
                "porosity_profiles_available": False,
            }
        )

    metrics.update(classify_operational_regime(metrics, settings))
    metrics.update(
        {
            "run": name,
            "source_directory": str(run_dir.resolve()),
        }
    )
    metrics.update(resolve_phase_parameters(run_dir, name, phase_metadata_rows))
    metrics["three_regime_projection"] = collapse_to_three_regimes(metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "processed_timeseries.csv", processed)
    if processed_spatial is not None:
        save_csv(output_dir / "processed_spatial_metrics.csv", processed_spatial)
    if focusing_summary is not None:
        save_csv(output_dir / "processed_focusing_summary.csv", focusing_summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metrics), handle, indent=2, sort_keys=True)
    classification_text = (
        f"Manuscript regime: {metrics.get('manuscript_regime', 'unclassified')}\n"
        f"Regime family: {metrics.get('regime_family', 'unknown')}\n"
        f"Definition fit: {metrics.get('regime_definition_fit', 'unknown')}\n"
        f"Confidence: {metrics.get('classification_confidence', 'unknown')}\n"
        f"Singurindy-compatible projection: "
        f"{metrics.get('singurindy_compatible_projection', 'unknown')}\n\n"
        f"Rationale:\n{metrics.get('classification_rationale', '')}\n"
    )
    (output_dir / "classification.txt").write_text(
        classification_text, encoding="utf-8"
    )
    write_metric_definitions(output_dir / "metric_definitions.txt")

    result = RunResult(
        name=name,
        source_dir=run_dir,
        output_dir=output_dir,
        params=params,
        spatial=spatial_raw,
        processed_spatial=processed_spatial,
        processed=processed,
        metrics=metrics,
        focusing_profiles=focusing,
        porosity_profiles=phi_profiles,
        profile_axis=profile_axis,
        profile_times=profile_times,
        focusing_summary=focusing_summary,
    )
    plot_conductivity(result, peaks, troughs, settings)
    plot_solid_evolution(result, settings)
    plot_spatial(result, settings)
    plot_regime_diagnostics(result, settings)
    if settings.include_profiles:
        plot_profile_diagnostics(result, settings)
    return result, peaks, troughs


def write_summary_csv(path: Path, results: Sequence[RunResult]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run",
        "manuscript_regime",
        "operational_regime",
        "operational_regime_with_hydraulic_fate",
        "regime_definition_fit",
        "regime_family",
        "singurindy_compatible_projection",
        "classification_confidence",
        "classification_rationale",
        "Da_eff",
        "cd_in",
        "inv_Da_eff",
        "inv_cd_in",
        "phase_parameters_available",
        "phase_parameter_source",
        "phase_config_path",
        "config_Da_eff",
        "config_cd_in",
        "phase_config_consistency",
        "phase_config_mismatch",
        "phase_metadata_conflict",
        "phase_metadata_conflict_detail",
        "three_regime_projection",
        "hydraulic_response",
        "conductivity_final_over_initial",
        "conductivity_oscillation_index",
        "conductivity_significant_extrema",
        "x_A50_over_L_final",
        "x_A50_over_L_max",
        "x_A50_nonmonotonicity_index",
        "x_A50_final_to_max_ratio",
        "deep_dissolution_penetration",
        "throughgoing_dissolution_penetration",
        "x_Q20_over_L_final_smoothed",
        "x_Q20_over_L_max_smoothed",
        "x_Q20_cumulative_retreat_over_L",
        "x_Q20_max_drawdown_from_previous_reach_over_L",
        "x_Q20_completed_retreat_recovery_events",
        "x_Q20_completed_reorganization_strength_over_L",
        "terminal_Q20_retreat",
        "strong_flow_focusing",
        "flow_focusing_score",
        "flow_focusing_score_source",
        "flow_focusing_late_mean",
        "flow_n50_late_mean",
        "flow_multichannel_fraction_x_late",
        "braided_flow_evidence",
        "front_like_candidate",
        "front_profile_supported",
        "porosity_front_step_score",
        "max_edge_flow_fraction_late_mean",
        "primary_removed_normalized_final",
        "primary_removed_volume_fraction_final",
        "secondary_added_fraction_final",
        "replaced_volume_fraction_final",
        "replaced_volume_added_fraction_final",
        "porosity_change",
    ]
    all_keys = set().union(*(result.metrics.keys() for result in results))
    for key in preferred + sorted(all_keys):
        if key in all_keys and key not in seen:
            keys.append(key)
            seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for result in results:
            row = {key: json_safe(result.metrics.get(key)) for key in keys}
            writer.writerow(row)


def comparison_x(result: RunResult, settings: Settings) -> tuple[np.ndarray, str]:
    return x_values_and_label(result.processed, settings.x_axis)


def plot_comparisons(
    results: Sequence[RunResult],
    output_dir: Path,
    settings: Settings,
) -> None:
    if len(results) < 2:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5), layout="constrained")
    xlabel = "time"
    for result in results:
        x, xlabel = comparison_x(result, settings)
        ax.plot(
            x,
            result.processed["relative_conductivity_smoothed"],
            linewidth=1.8,
            label=result.name,
        )
    ax.axhline(1.0, linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$K/K_0$")
    ax.legend(frameon=False, fontsize="small", ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    save_figure(fig, output_dir / "comparison_conductivity", settings)

    spatial_results = [
        result for result in results if result.processed_spatial is not None
    ]
    if spatial_results:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8.2, 9.0),
            sharex=False,
            layout="constrained",
        )
        for result in spatial_results:
            spatial = result.processed_spatial
            assert spatial is not None
            x, xlabel = spatial_x_values_and_label(spatial, settings.x_axis)
            axes[0].plot(x, spatial["x_A50_over_L"], linewidth=1.8, label=result.name)
            axes[1].plot(
                x,
                spatial["x_Q20_over_L_smoothed"],
                linewidth=1.8,
                label=result.name,
            )
            axes[2].plot(
                x,
                spatial["x_Q20_cumulative_retreat"],
                linewidth=1.8,
                label=result.name,
            )
        axes[0].set_ylabel(r"$x_{A50}/L$")
        axes[1].set_ylabel(r"$x_{Q20}/L$")
        axes[2].set_ylabel(r"cumulative retreat$/L$")
        axes[2].set_xlabel(xlabel)
        axes[2].axhline(
            settings.front_retreat_prominence,
            linestyle="--",
            linewidth=0.8,
        )
        for axis in axes[:2]:
            axis.set_ylim(-0.02, 1.05)
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False, fontsize="small", ncol=2)
        save_figure(
            fig,
            output_dir / "comparison_penetration_and_retreat",
            settings,
        )

    x_values: list[float] = []
    y_values: list[float] = []
    labels: list[str] = []
    for result in results:
        xa = safe_float(result.metrics.get("x_A50_over_L_max"))
        kf = safe_float(result.metrics.get("conductivity_final_over_initial"))
        if math.isfinite(xa) and math.isfinite(kf) and kf > 0:
            x_values.append(xa)
            y_values.append(kf)
            labels.append(result.name)
    if x_values:
        fig, ax = plt.subplots(figsize=(7.0, 5.5), layout="constrained")
        ax.scatter(x_values, y_values)
        for x_value, y_value, label in zip(x_values, y_values, labels):
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
        ax.axhline(1.0, linestyle="--", linewidth=0.8)
        ax.axvline(
            settings.deep_penetration_threshold,
            linestyle=":",
            linewidth=0.8,
        )
        ax.set_yscale("log")
        ax.set_xlim(-0.02, 1.05)
        ax.set_xlabel(r"maximum $x_{A50}/L$")
        ax.set_ylabel(r"final $K/K_0$")
        ax.grid(True, which="both", alpha=0.25)
        save_figure(
            fig,
            output_dir / "summary_penetration_vs_conductivity",
            settings,
        )

    diagnostic_rows: list[tuple[float, float, float, str]] = []
    for result in results:
        focusing = safe_float(result.metrics.get("flow_focusing_score"))
        reorganization = safe_float(
            result.metrics.get("x_Q20_completed_reorganization_strength_over_L")
        )
        penetration = safe_float(result.metrics.get("x_A50_over_L_max"))
        if math.isfinite(focusing) and math.isfinite(reorganization):
            diagnostic_rows.append(
                (
                    reorganization,
                    focusing,
                    penetration if math.isfinite(penetration) else 0.0,
                    result.name,
                )
            )
    if diagnostic_rows:
        fig, ax = plt.subplots(figsize=(7.2, 5.8), layout="constrained")
        maximum_reorganization = 0.0
        for reorganization, focusing, penetration, label in diagnostic_rows:
            maximum_reorganization = max(maximum_reorganization, reorganization)
            marker_size = 35.0 + 150.0 * float(np.clip(penetration, 0.0, 1.0))
            ax.scatter(reorganization, focusing, s=marker_size)
            ax.annotate(
                label,
                (reorganization, focusing),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
        ax.axvline(
            settings.front_retreat_prominence,
            linestyle="--",
            linewidth=0.8,
        )
        ax.axhline(
            settings.focusing_threshold,
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_xlabel(r"completed Q20 reorganization strength$/L$")
        ax.set_ylabel("flow-focusing score")
        ax.set_xlim(
            0.0,
            max(
                0.10,
                1.2 * settings.front_retreat_prominence,
                1.1 * maximum_reorganization,
            ),
        )
        ax.grid(True, alpha=0.25)
        save_figure(fig, output_dir / "summary_reorganization_vs_focusing", settings)

    oscillation_rows: list[tuple[float, float, str]] = []
    for result in results:
        oscillation = safe_float(
            result.metrics.get("conductivity_oscillation_index")
        )
        retreat = safe_float(
            result.metrics.get("x_Q20_cumulative_retreat_over_L")
        )
        if math.isfinite(oscillation) and math.isfinite(retreat):
            oscillation_rows.append((oscillation, retreat, result.name))
    if oscillation_rows:
        fig, ax = plt.subplots(figsize=(7.2, 5.8), layout="constrained")
        for oscillation, retreat, label in oscillation_rows:
            ax.scatter(oscillation, retreat)
            ax.annotate(
                label,
                (oscillation, retreat),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
        ax.axvline(settings.oscillation_threshold, linestyle="--", linewidth=0.8)
        ax.axhline(settings.front_retreat_prominence, linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"conductivity oscillation index $O_K$")
        ax.set_ylabel(r"cumulative Q20 retreat$/L$")
        ax.grid(True, alpha=0.25)
        save_figure(fig, output_dir / "summary_oscillation_vs_retreat", settings)


    # Manuscript-regime overview: same hydraulic/penetration plane as the
    # standard summary, but grouped by the exact eight-regime labels.
    marker_cycle = ("o", "s", "D", "^", "v", "P", "X", "*")
    grouped: dict[str, list[tuple[float, float, str]]] = {
        regime: [] for regime in MANUSCRIPT_REGIMES
    }
    for result in results:
        regime = str(result.metrics.get("manuscript_regime", ""))
        xa = safe_float(result.metrics.get("x_A50_over_L_max"))
        kf = safe_float(result.metrics.get("conductivity_final_over_initial"))
        if regime in grouped and math.isfinite(xa) and math.isfinite(kf) and kf > 0:
            grouped[regime].append((xa, kf, result.name))
    if any(grouped.values()):
        fig, ax = plt.subplots(figsize=(8.4, 6.2), layout="constrained")
        for marker, regime in zip(marker_cycle, MANUSCRIPT_REGIMES):
            rows = grouped[regime]
            if not rows:
                continue
            xs = [row[0] for row in rows]
            ys = [row[1] for row in rows]
            ax.scatter(
                xs,
                ys,
                marker=marker,
                s=70,
                label=regime.replace("_", " "),
            )
            for x_value, y_value, label in rows:
                ax.annotate(
                    label,
                    (x_value, y_value),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize="x-small",
                )
        ax.axhline(1.0, linestyle="--", linewidth=0.8)
        ax.axvline(settings.deep_penetration_threshold, linestyle=":", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlim(-0.02, 1.05)
        ax.set_xlabel(r"maximum $x_{A50}/L$")
        ax.set_ylabel(r"final $K/K_0$")
        ax.set_title("eight-regime classification")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize="small", ncol=2)
        save_figure(fig, output_dir / "summary_manuscript_regimes", settings)

def _replacement_value(metrics: Mapping[str, Any], metric_name: str) -> float:
    metric_key, _ = REPLACEMENT_METRIC_SPECS[metric_name]
    return safe_float(metrics.get(metric_key))


def collect_phase_records(
    results: Sequence[RunResult],
    replacement_metric: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        daeff = safe_float(result.metrics.get("Da_eff"))
        cdin = safe_float(result.metrics.get("cd_in"))
        x_value = safe_float(result.metrics.get("inv_Da_eff"))
        y_value = safe_float(result.metrics.get("inv_cd_in"))
        if not (
            math.isfinite(daeff)
            and daeff > 0.0
            and math.isfinite(cdin)
            and cdin > 0.0
            and math.isfinite(x_value)
            and math.isfinite(y_value)
        ):
            continue
        records.append(
            {
                "run": result.name,
                "source_directory": str(result.source_dir),
                "Da_eff": daeff,
                "cd_in": cdin,
                "inv_Da_eff": x_value,
                "inv_cd_in": y_value,
                "three_regime": str(
                    result.metrics.get("three_regime_projection", "competition")
                ),
                "full_regime": str(result.metrics.get("manuscript_regime", "")),
                "classification_confidence": str(
                    result.metrics.get("classification_confidence", "unknown")
                ),
                "replacement_metric": replacement_metric,
                "replaced_volume": _replacement_value(
                    result.metrics, replacement_metric
                ),
                "phase_parameter_source": str(
                    result.metrics.get("phase_parameter_source", "unavailable")
                ),
            }
        )
    return records


def _majority_label(
    values: Sequence[str],
    order: Sequence[str],
) -> tuple[str, float, bool, dict[str, int]]:
    counts = Counter(value for value in values if value)
    if not counts:
        return "", math.nan, False, {}
    maximum = max(counts.values())
    order_index = {value: index for index, value in enumerate(order)}
    winners = [value for value, count in counts.items() if count == maximum]
    winners.sort(key=lambda value: order_index.get(value, len(order_index)))
    selected = winners[0]
    agreement = maximum / sum(counts.values())
    return selected, float(agreement), len(counts) > 1, dict(counts)


def aggregate_phase_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            round(float(record["inv_Da_eff"]), 12),
            round(float(record["inv_cd_in"]), 12),
        )
        grouped[key].append(record)

    cells: list[dict[str, Any]] = []
    for (x_value, y_value), rows in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        coarse, coarse_agreement, coarse_mixed, coarse_counts = _majority_label(
            [str(row["three_regime"]) for row in rows], THREE_REGIMES
        )
        full, full_agreement, full_mixed, full_counts = _majority_label(
            [str(row["full_regime"]) for row in rows], MANUSCRIPT_REGIMES
        )
        replacement_values = np.asarray(
            [safe_float(row.get("replaced_volume")) for row in rows], dtype=float
        )
        replacement_values = replacement_values[np.isfinite(replacement_values)]
        replacement_mean = (
            float(np.mean(replacement_values))
            if replacement_values.size
            else math.nan
        )
        replacement_std = (
            float(np.std(replacement_values))
            if replacement_values.size
            else math.nan
        )
        cells.append(
            {
                "inv_Da_eff": x_value,
                "inv_cd_in": y_value,
                "Da_eff": 1.0 / x_value,
                "cd_in": 1.0 / y_value,
                "n_runs": len(rows),
                "runs": ";".join(str(row["run"]) for row in rows),
                "three_regime": coarse,
                "three_regime_agreement": coarse_agreement,
                "three_regime_mixed": coarse_mixed,
                "three_regime_counts": json.dumps(coarse_counts, sort_keys=True),
                "full_regime": full,
                "full_regime_agreement": full_agreement,
                "full_regime_mixed": full_mixed,
                "full_regime_counts": json.dumps(full_counts, sort_keys=True),
                "replaced_volume_mean": replacement_mean,
                "replaced_volume_std": replacement_std,
                "replaced_volume_min": (
                    float(np.min(replacement_values))
                    if replacement_values.size
                    else math.nan
                ),
                "replaced_volume_max": (
                    float(np.max(replacement_values))
                    if replacement_values.size
                    else math.nan
                ),
            }
        )
    return cells


def _write_dict_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in keys})


def _coordinate_edges(values: np.ndarray, scale: str) -> np.ndarray:
    values = np.asarray(sorted(set(float(value) for value in values)), dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    if values.size == 1:
        value = values[0]
        if scale == "log":
            factor = math.sqrt(2.0)
            return np.asarray([value / factor, value * factor], dtype=float)
        width = max(abs(value) * 0.2, 0.5)
        lower = (
            max(np.finfo(float).tiny, value - width)
            if value > 0.0
            else value - width
        )
        return np.asarray([lower, value + width], dtype=float)

    edges = np.empty(values.size + 1, dtype=float)
    if scale == "log":
        if np.any(values <= 0.0):
            raise ValueError("Logarithmic phase axes require positive coordinates")
        edges[1:-1] = np.sqrt(values[:-1] * values[1:])
        edges[0] = values[0] ** 2 / edges[1]
        edges[-1] = values[-1] ** 2 / edges[-2]
    else:
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        if values[0] > 0.0 and edges[0] <= 0.0:
            edges[0] = max(np.finfo(float).tiny, values[0] * 0.5)
    return edges


def _phase_grid(
    cells: Sequence[Mapping[str, Any]],
    value_key: str,
    categories: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(
        sorted({float(cell["inv_Da_eff"]) for cell in cells}), dtype=float
    )
    ys = np.asarray(sorted({float(cell["inv_cd_in"]) for cell in cells}), dtype=float)
    matrix = np.full((ys.size, xs.size), np.nan, dtype=float)
    x_index = {round(value, 12): index for index, value in enumerate(xs)}
    y_index = {round(value, 12): index for index, value in enumerate(ys)}
    category_index = {
        category: index for index, category in enumerate(categories or ())
    }
    for cell in cells:
        row = y_index[round(float(cell["inv_cd_in"]), 12)]
        column = x_index[round(float(cell["inv_Da_eff"]), 12)]
        value = cell.get(value_key)
        if categories is None:
            matrix[row, column] = safe_float(value)
        elif str(value) in category_index:
            matrix[row, column] = category_index[str(value)]
    return xs, ys, matrix


def _format_phase_tick(value: float) -> str:
    return f"{value:.3g}"


def _configure_phase_axes(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    settings: Settings,
) -> None:
    ax.set_xscale(settings.phase_axis_scale)
    ax.set_yscale(settings.phase_axis_scale)
    ax.set_xlabel(r"$1/\mathrm{Da}_{\mathrm{eff}}$")
    ax.set_ylabel(r"$1/c_D^{\mathrm{in}}$")

    if xs.size:
        x_edges = _coordinate_edges(xs, settings.phase_axis_scale)
        ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    if ys.size:
        y_edges = _coordinate_edges(ys, settings.phase_axis_scale)
        ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))

    if xs.size <= 12:
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [_format_phase_tick(value) for value in xs],
            rotation=35,
            ha="right",
        )
    if ys.size <= 12:
        ax.set_yticks(ys)
        ax.set_yticklabels([_format_phase_tick(value) for value in ys])
    ax.grid(True, which="both", alpha=0.20, linewidth=0.6)


def _annotate_phase_points(
    ax: plt.Axes,
    cells: Sequence[Mapping[str, Any]],
    key: str,
    codes: Mapping[str, str] | None,
    numeric: bool,
    enabled: bool,
) -> None:
    if not enabled or len(cells) > 100:
        return
    for cell in cells:
        value = cell.get(key)
        if numeric:
            number = safe_float(value)
            if not math.isfinite(number):
                continue
            text = f"{number:.3g}"
        else:
            label = str(value)
            text = codes.get(label, label) if codes is not None else label
        if int(cell.get("n_runs", 1)) > 1:
            text += f"\n(n={int(cell['n_runs'])})"
        if bool(cell.get(f"{key}_mixed", False)):
            text += "*"
        ax.annotate(
            text,
            (float(cell["inv_Da_eff"]), float(cell["inv_cd_in"])),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize="x-small",
        )


def _plot_three_regime_points(
    cells: Sequence[Mapping[str, Any]],
    ax: plt.Axes,
) -> list[Any]:
    """Plot the three classes using Singurindy--Berkowitz-style symbols."""
    styles = {
        "precipitation_dominated": {
            "marker": "v",
            "facecolors": "black",
            "edgecolors": "black",
        },
        "competition": {
            "marker": "o",
            "facecolors": "none",
            "edgecolors": "black",
        },
        "dissolution_dominated": {
            "marker": "o",
            "facecolors": "black",
            "edgecolors": "black",
        },
    }
    handles: list[Any] = []
    for category in THREE_REGIMES:
        rows = [cell for cell in cells if str(cell.get("three_regime")) == category]
        if not rows:
            continue
        style = styles[category]
        collection = ax.scatter(
            [float(cell["inv_Da_eff"]) for cell in rows],
            [float(cell["inv_cd_in"]) for cell in rows],
            s=115,
            linewidths=1.5,
            zorder=3,
            label=THREE_REGIME_LABELS[category],
            **style,
        )
        handles.append(collection)
    return handles


def _plot_full_regime_points(
    cells: Sequence[Mapping[str, Any]],
    ax: plt.Axes,
) -> list[Any]:
    markers = ("s", "P", "D", "o", "^", "v", "X", "h")
    palette = plt.get_cmap("tab10")
    handles: list[Any] = []
    for index, category in enumerate(MANUSCRIPT_REGIMES):
        rows = [cell for cell in cells if str(cell.get("full_regime")) == category]
        if not rows:
            continue
        collection = ax.scatter(
            [float(cell["inv_Da_eff"]) for cell in rows],
            [float(cell["inv_cd_in"]) for cell in rows],
            s=145,
            marker=markers[index],
            facecolors=[palette(index % palette.N)],
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            label=MANUSCRIPT_REGIME_LABELS[category],
        )
        handles.append(collection)
    return handles


def plot_categorical_phase_diagram(
    cells: Sequence[Mapping[str, Any]],
    output_dir: Path,
    settings: Settings,
    value_key: str,
    categories: Sequence[str],
    labels: Mapping[str, str],
    codes: Mapping[str, str],
    title: str,
    filename: str,
) -> None:
    if not cells:
        return
    xs = np.asarray(sorted({float(cell["inv_Da_eff"]) for cell in cells}))
    ys = np.asarray(sorted({float(cell["inv_cd_in"]) for cell in cells}))

    # Categorical phase diagrams are point maps rather than pcolormesh cells.
    # This avoids a single sampled point being rendered as a coloured rectangle
    # covering the whole plotting area and does not imply boundaries between
    # sparsely sampled parameter combinations.
    width = 8.8 if value_key == "three_regime" else 11.0
    fig, ax = plt.subplots(figsize=(width, 6.6))
    if value_key == "three_regime":
        handles = _plot_three_regime_points(cells, ax)
    else:
        handles = _plot_full_regime_points(cells, ax)

    mixed_key = f"{value_key}_mixed"
    mixed_rows = [cell for cell in cells if bool(cell.get(mixed_key, False))]
    if mixed_rows:
        ax.scatter(
            [float(cell["inv_Da_eff"]) for cell in mixed_rows],
            [float(cell["inv_cd_in"]) for cell in mixed_rows],
            marker="x",
            s=100,
            linewidths=1.4,
            color="black",
            zorder=5,
        )
        handles.append(
            Line2D(
                [0], [0], marker="x", linestyle="none", color="black",
                label="mixed outcomes at one parameter point"
            )
        )

    _configure_phase_axes(ax, xs, ys, settings)
    ax.set_title(title)
    _annotate_phase_points(
        ax,
        cells,
        value_key,
        codes,
        numeric=False,
        enabled=settings.phase_cell_labels,
    )

    if value_key == "three_regime":
        ax.legend(handles=handles, frameon=False, loc="best")
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.91)
    else:
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize="small",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.10, right=0.72, bottom=0.16, top=0.91)
    save_figure(fig, output_dir / filename, settings)


def plot_replaced_volume_heatmap(
    cells: Sequence[Mapping[str, Any]],
    output_dir: Path,
    settings: Settings,
) -> None:
    xs, ys, matrix = _phase_grid(cells, "replaced_volume_mean", None)
    if xs.size == 0 or ys.size == 0 or not np.any(np.isfinite(matrix)):
        return

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    finite_values = matrix[np.isfinite(matrix)]

    # Use real cells only when both axes contain at least two sampled values.
    # A scatter-square fallback is much more honest for a line sweep or a
    # singleton point and avoids filling the complete axes with one value.
    if xs.size >= 2 and ys.size >= 2:
        x_edges = _coordinate_edges(xs, settings.phase_axis_scale)
        y_edges = _coordinate_edges(ys, settings.phase_axis_scale)
        masked = np.ma.masked_invalid(matrix)
        artist = ax.pcolormesh(
            x_edges,
            y_edges,
            masked,
            cmap="viridis",
            shading="flat",
        )
    else:
        point_rows = [
            cell for cell in cells
            if math.isfinite(safe_float(cell.get("replaced_volume_mean")))
        ]
        artist = ax.scatter(
            [float(cell["inv_Da_eff"]) for cell in point_rows],
            [float(cell["inv_cd_in"]) for cell in point_rows],
            c=[safe_float(cell.get("replaced_volume_mean")) for cell in point_rows],
            cmap="viridis",
            marker="s",
            s=420,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

    _configure_phase_axes(ax, xs, ys, settings)
    _, metric_label = REPLACEMENT_METRIC_SPECS[settings.replacement_metric]
    ax.set_title("replaced-volume map")
    colorbar = fig.colorbar(artist, ax=ax, pad=0.03)
    colorbar.set_label(metric_label)
    _annotate_phase_points(
        ax,
        cells,
        "replaced_volume_mean",
        codes=None,
        numeric=True,
        enabled=settings.phase_cell_labels,
    )
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.16, top=0.91)
    save_figure(fig, output_dir / "phase_heatmap_replaced_volume", settings)


def plot_phase_diagrams(
    results: Sequence[RunResult],
    output_dir: Path,
    settings: Settings,
) -> None:
    if not settings.include_phase_diagrams:
        return

    # Always write a template so missing coordinates are obvious and easy to
    # fill.  Blank values are kept for runs that could not be resolved.
    template_rows = [
        {
            "run": result.name,
            "source_directory": str(result.source_dir),
            "Da_eff": safe_float(result.metrics.get("Da_eff")),
            "cd_in": safe_float(result.metrics.get("cd_in")),
            "parameter_source": str(
                result.metrics.get("phase_parameter_source", "unavailable")
            ),
        }
        for result in results
    ]
    _write_dict_rows(output_dir / "phase_metadata_template.csv", template_rows)

    unresolved = [
        result.name
        for result in results
        if not bool(result.metrics.get("phase_parameters_available", False))
    ]
    if unresolved:
        warnings.warn(
            "The following runs have no valid Da_eff/cd_in pair and are omitted "
            "from phase diagrams: " + ", ".join(unresolved)
        )

    records = collect_phase_records(results, settings.replacement_metric)
    _write_dict_rows(output_dir / "phase_diagram_points.csv", records)
    if not records:
        warnings.warn(
            "No runs have both positive Da_eff and cd_in; phase diagrams were "
            "skipped. Supply --phase-metadata or fill phase_metadata_template.csv."
        )
        return

    cells = aggregate_phase_records(records)
    _write_dict_rows(output_dir / "phase_diagram_grid.csv", cells)

    unique_coordinates = {
        (round(float(record["inv_Da_eff"]), 12), round(float(record["inv_cd_in"]), 12))
        for record in records
    }
    sources = {str(record.get("phase_parameter_source", "")) for record in records}
    print(
        f"[phase] {len(records)} resolved run(s), {len(unique_coordinates)} "
        f"unique parameter point(s); sources: {', '.join(sorted(sources))}"
    )

    # A single coordinate inferred automatically for several different runs is
    # almost always a shared-parent parsing error.  Do not create a misleading
    # one-cell phase map in that case.
    if (
        len(records) > 1
        and len(unique_coordinates) == 1
        and all(source.startswith("auto:") for source in sources)
    ):
        warnings.warn(
            "All runs were assigned the same phase coordinate by automatic "
            "extraction. Phase plots were skipped to avoid a misleading map. "
            "Pass --phase-metadata new_singurindy.txt (or edit "
            "phase_metadata_template.csv)."
        )
        return

    plot_categorical_phase_diagram(
        cells,
        output_dir,
        settings,
        value_key="three_regime",
        categories=THREE_REGIMES,
        labels=THREE_REGIME_LABELS,
        codes=THREE_REGIME_CODES,
        title="Singurindy--Berkowitz-style regime projection",
        filename="phase_diagram_three_regimes",
    )
    plot_categorical_phase_diagram(
        cells,
        output_dir,
        settings,
        value_key="full_regime",
        categories=MANUSCRIPT_REGIMES,
        labels=MANUSCRIPT_REGIME_LABELS,
        codes=MANUSCRIPT_REGIME_CODES,
        title="detailed mineral-replacement regime map",
        filename="phase_diagram_full_regimes",
    )
    plot_replaced_volume_heatmap(cells, output_dir, settings)


def discover_runs(paths: Sequence[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file() and path.name == "params.txt":
            found.append(path.parent)
            continue
        if not path.exists():
            warnings.warn(f"Skipping missing path: {path}")
            continue
        if not path.is_dir():
            warnings.warn(f"Skipping non-directory: {path}")
            continue
        if (path / "params.txt").exists():
            found.append(path)
            continue
        pattern = "**/params.txt" if recursive else "*/params.txt"
        found.extend(item.parent for item in sorted(path.glob(pattern)))

    unique: list[Path] = []
    seen: set[Path] = set()
    for run in found:
        resolved = run.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def unique_run_names(run_dirs: Sequence[Path]) -> list[str]:
    """Create stable labels without losing the parameter-folder context.

    A bare final directory name is retained when unique.  Duplicate names are
    expanded using the shortest unique suffix of the full path, rather than
    receiving order-dependent labels such as ``0_2`` or ``template_7``.
    """
    if not run_dirs:
        return []
    base_counts = Counter(run.name or "run" for run in run_dirs)
    names: list[str] = []
    used: set[str] = set()
    resolved_parts = [run.resolve().parts for run in run_dirs]

    for index, run in enumerate(run_dirs):
        base = run.name or "run"
        if base_counts[base] == 1:
            candidate = base
        else:
            parts = resolved_parts[index]
            candidate = base
            for depth in range(2, len(parts) + 1):
                suffix = "__".join(parts[-depth:])
                collisions = sum(
                    1
                    for other in resolved_parts
                    if len(other) >= depth and "__".join(other[-depth:]) == suffix
                )
                if collisions == 1:
                    candidate = suffix
                    break
        candidate = re.sub(r"[^A-Za-z0-9_.+-]+", "_", candidate).strip("_") or "run"
        original = candidate
        counter = 2
        while candidate in used:
            candidate = f"{original}_{counter}"
            counter += 1
        names.append(candidate)
        used.add(candidate)
    return names

def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process poreNetDP conductivity, penetration, Q20-front "
            "reorganization, and flow-focusing diagnostics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories, params.txt files, or parent directories containing runs.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively below supplied parent directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Root output directory. One subdirectory is created per run.",
    )
    parser.add_argument(
        "--conductivity-mode",
        choices=("auto", "constant-flow", "variable-flow"),
        default="auto",
    )
    parser.add_argument(
        "--x-axis",
        choices=("time", "normalized-time", "throughput"),
        default="time",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Savitzky--Golay window for ln(K/K0); use 0 for automatic.",
    )
    parser.add_argument("--smooth-polyorder", type=int, default=3)
    parser.add_argument(
        "--peak-prominence",
        type=float,
        default=0.03,
        help="Minimum peak/trough prominence in ln(K/K0).",
    )
    parser.add_argument(
        "--peak-distance",
        type=int,
        default=3,
        help="Minimum separation between significant conductivity extrema.",
    )
    parser.add_argument(
        "--net-log-threshold",
        type=float,
        default=0.10,
        help="|ln(Kf/K0)| below this value is hydraulically balanced.",
    )
    parser.add_argument(
        "--oscillation-threshold",
        type=float,
        default=0.15,
        help="Minimum conductivity oscillation index for an oscillatory label.",
    )
    parser.add_argument(
        "--breakthrough-threshold",
        type=float,
        default=0.95,
        help="Normalized axial position used for breakthrough timing.",
    )
    parser.add_argument(
        "--deep-penetration-threshold",
        type=float,
        default=0.50,
        help="Minimum maximum x_A50/L used to distinguish deep/channelized penetration.",
    )
    parser.add_argument(
        "--clogged-conductivity-threshold",
        type=float,
        default=0.10,
        help="Kf/K0 below this value is treated as terminal hydraulic failure.",
    )
    parser.add_argument(
        "--front-smooth-window",
        type=int,
        default=7,
        help="Savitzky--Golay window for x_Q20/L; use 0 for automatic.",
    )
    parser.add_argument(
        "--front-retreat-prominence",
        type=float,
        default=0.05,
        help="Minimum prominent Q20 retreat, expressed as a fraction of L.",
    )
    parser.add_argument(
        "--front-event-distance",
        type=int,
        default=3,
        help="Minimum separation between Q20 retreat troughs in saved samples.",
    )
    parser.add_argument(
        "--front-recovery-fraction",
        type=float,
        default=0.50,
        help="Required re-advance as a fraction of the Q20 retreat threshold.",
    )
    parser.add_argument(
        "--front-nonmonotonicity-threshold",
        type=float,
        default=0.20,
        help="Maximum A50 nonmonotonicity compatible with a propagating front.",
    )
    parser.add_argument(
        "--front-persistence-threshold",
        type=float,
        default=0.90,
        help="Minimum final/max A50 ratio compatible with a persistent front.",
    )
    parser.add_argument(
        "--front-profile-score-threshold",
        type=float,
        default=0.55,
        help="Minimum single-step variance-explained score for porosity-front support.",
    )
    parser.add_argument(
        "--front-profile-min-amplitude",
        type=float,
        default=0.005,
        help="Minimum 5--95 percentile axial porosity-change amplitude for front scoring.",
    )
    parser.add_argument(
        "--focusing-threshold",
        type=float,
        default=0.50,
        help="Late mean slice-focusing index classified as strong focusing.",
    )
    parser.add_argument(
        "--focusing-local-threshold",
        type=float,
        default=0.50,
        help="F(x) threshold used to count strongly focused axial slices.",
    )
    parser.add_argument(
        "--focusing-late-fraction",
        type=float,
        default=0.25,
        help="Final fraction of focusing snapshots used for late-time summaries.",
    )
    parser.add_argument(
        "--focusing-axial-margin",
        type=float,
        default=0.05,
        help="Fraction of sample length excluded at each end from focusing summaries.",
    )
    parser.add_argument(
        "--dominant-edge-fraction-threshold",
        type=float,
        default=0.20,
        help="Fallback strong-focusing threshold for max(|q_e|)/Q_in.",
    )
    parser.add_argument(
        "--braided-n50-threshold",
        type=float,
        default=2.0,
        help="Minimum N50 interpreted as more than one important flow branch.",
    )
    parser.add_argument(
        "--braided-slice-fraction-threshold",
        type=float,
        default=0.30,
        help="Minimum late axial fraction of slices satisfying the braided N50 criterion.",
    )
    parser.add_argument(
        "--phase-metadata",
        type=Path,
        default=None,
        help=(
            "Optional CSV/TSV/whitespace fallback with run or folder, Da_eff, "
            "and cd_in columns. Standalone assignments in each run-local "
            "config.txt are always authoritative; metadata fills only missing "
            "used only when this option is supplied explicitly."
        ),
    )
    parser.add_argument(
        "--phase-axis-scale",
        choices=("linear", "log"),
        default="linear",
        help="Scale used for both inverse-parameter axes in the phase diagrams.",
    )
    parser.add_argument(
        "--replacement-metric",
        choices=tuple(REPLACEMENT_METRIC_SPECS),
        default="secondary_final",
        help="Quantity displayed in the replaced-volume heatmap.",
    )
    parser.add_argument(
        "--no-phase-diagrams",
        action="store_true",
        help="Skip the two phase diagrams and the replaced-volume heatmap.",
    )
    parser.add_argument(
        "--no-phase-cell-labels",
        action="store_true",
        help="Do not annotate phase cells with regime codes or heatmap values.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-profiles",
        action="store_true",
        help="Skip profiles.txt and profiles_phi.txt processing.",
    )

    args = parser.parse_args(argv)
    fraction_fields = (
        "breakthrough_threshold",
        "deep_penetration_threshold",
        "clogged_conductivity_threshold",
        "front_retreat_prominence",
        "front_recovery_fraction",
        "front_nonmonotonicity_threshold",
        "front_persistence_threshold",
        "front_profile_score_threshold",
        "focusing_threshold",
        "focusing_local_threshold",
        "focusing_late_fraction",
        "focusing_axial_margin",
        "dominant_edge_fraction_threshold",
        "braided_slice_fraction_threshold",
    )
    for field in fraction_fields:
        value = float(getattr(args, field))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{field.replace('_', '-')} must lie in [0, 1]")
    if args.focusing_axial_margin >= 0.5:
        parser.error("--focusing-axial-margin must be smaller than 0.5")
    if args.smooth_polyorder < 0:
        parser.error("--smooth-polyorder must be non-negative")
    if args.front_event_distance < 1 or args.peak_distance < 1:
        parser.error("event distances must be at least one saved sample")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    validation_errors: list[str] = []
    if not 0.0 <= args.deep_penetration_threshold <= 1.0:
        validation_errors.append("--deep-penetration-threshold must lie in [0, 1]")
    if not 0.0 < args.breakthrough_threshold <= 1.0:
        validation_errors.append("--breakthrough-threshold must lie in (0, 1]")
    if args.deep_penetration_threshold > args.breakthrough_threshold:
        validation_errors.append(
            "--deep-penetration-threshold cannot exceed --breakthrough-threshold"
        )
    if args.front_retreat_prominence <= 0.0:
        validation_errors.append("--front-retreat-prominence must be positive")
    if args.front_recovery_fraction < 0.0:
        validation_errors.append("--front-recovery-fraction must be non-negative")
    if args.front_profile_min_amplitude < 0.0:
        validation_errors.append("--front-profile-min-amplitude must be non-negative")
    if args.braided_n50_threshold < 1.0:
        validation_errors.append("--braided-n50-threshold must be at least 1")
    if not 0.0 <= args.focusing_threshold <= 1.0:
        validation_errors.append("--focusing-threshold must lie in [0, 1]")
    if not 0.0 <= args.focusing_local_threshold <= 1.0:
        validation_errors.append("--focusing-local-threshold must lie in [0, 1]")
    if not 0.0 < args.focusing_late_fraction <= 1.0:
        validation_errors.append("--focusing-late-fraction must lie in (0, 1]")
    if not 0.0 <= args.focusing_axial_margin < 0.5:
        validation_errors.append("--focusing-axial-margin must lie in [0, 0.5)")
    if not 0.0 < args.dominant_edge_fraction_threshold <= 1.0:
        validation_errors.append(
            "--dominant-edge-fraction-threshold must lie in (0, 1]"
        )
    if validation_errors:
        print("Invalid post-processing settings:", file=sys.stderr)
        for message in validation_errors:
            print(f"  - {message}", file=sys.stderr)
        return 2

    run_dirs = discover_runs(args.runs, args.recursive)
    if not run_dirs:
        print(
            "No runs containing params.txt were found. Use --recursive for nested results.",
            file=sys.stderr,
        )
        return 2

    # Phase coordinates are read from the config.txt in each run directory.
    # A metadata table is considered only when the user supplies it explicitly.
    phase_metadata_path = args.phase_metadata
    try:
        phase_metadata_rows = load_phase_metadata_table(phase_metadata_path)
    except (OSError, ValueError) as exc:
        print(f"Could not read phase metadata override: {exc}", file=sys.stderr)
        return 2
    if phase_metadata_path is None:
        print("[phase] reading authoritative Da_eff and cd_in from each run-local config.txt")
    else:
        print(f"[phase] using {phase_metadata_path} only as fallback for missing config assignments")

    settings = Settings(
        conductivity_mode=args.conductivity_mode,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
        peak_prominence=args.peak_prominence,
        peak_distance=args.peak_distance,
        net_log_threshold=args.net_log_threshold,
        oscillation_threshold=args.oscillation_threshold,
        breakthrough_threshold=args.breakthrough_threshold,
        deep_penetration_threshold=args.deep_penetration_threshold,
        clogged_conductivity_threshold=args.clogged_conductivity_threshold,
        front_smooth_window=args.front_smooth_window,
        front_retreat_prominence=args.front_retreat_prominence,
        front_event_distance=args.front_event_distance,
        front_recovery_fraction=args.front_recovery_fraction,
        front_nonmonotonicity_threshold=args.front_nonmonotonicity_threshold,
        front_persistence_threshold=args.front_persistence_threshold,
        front_profile_score_threshold=args.front_profile_score_threshold,
        front_profile_min_amplitude=args.front_profile_min_amplitude,
        focusing_threshold=args.focusing_threshold,
        focusing_local_threshold=args.focusing_local_threshold,
        focusing_late_fraction=args.focusing_late_fraction,
        focusing_axial_margin=args.focusing_axial_margin,
        dominant_edge_fraction_threshold=args.dominant_edge_fraction_threshold,
        braided_n50_threshold=args.braided_n50_threshold,
        braided_slice_fraction_threshold=args.braided_slice_fraction_threshold,
        x_axis=args.x_axis,
        formats=tuple(args.formats),
        dpi=args.dpi,
        include_profiles=not args.no_profiles,
        include_phase_diagrams=not args.no_phase_diagrams,
        phase_axis_scale=args.phase_axis_scale,
        phase_cell_labels=not args.no_phase_cell_labels,
        replacement_metric=args.replacement_metric,
    )

    if args.output is not None:
        root_output = args.output.expanduser().resolve()
    elif len(run_dirs) == 1:
        root_output = run_dirs[0] / "postprocessing"
    else:
        root_output = Path.cwd() / "postprocessing"
    root_output.mkdir(parents=True, exist_ok=True)

    names = unique_run_names(run_dirs)
    results: list[RunResult] = []
    failures: list[tuple[Path, str]] = []

    for run_dir, name in zip(run_dirs, names):
        output_dir = (
            root_output
            if len(run_dirs) == 1 and args.output is None
            else root_output / name
        )
        try:
            result, _, _ = analyze_run(
                run_dir,
                output_dir,
                name,
                settings,
                phase_metadata_rows=phase_metadata_rows,
            )
        except Exception as exc:  # keep batch processing alive after one bad folder
            failures.append((run_dir, str(exc)))
            print(f"[failed] {run_dir}: {exc}", file=sys.stderr)
            continue
        results.append(result)
        event_count = int(
            result.metrics.get("x_Q20_completed_retreat_recovery_events", 0) or 0
        )
        focus = safe_float(result.metrics.get("flow_focusing_score"))
        daeff_log = safe_float(result.metrics.get("Da_eff"))
        cdin_log = safe_float(result.metrics.get("cd_in"))
        phase_source = str(result.metrics.get("phase_parameter_source", "unavailable"))
        if math.isfinite(daeff_log) and math.isfinite(cdin_log):
            consistency = str(
                result.metrics.get("phase_config_consistency", "unverified")
            )
            phase_note = (
                f", Da_eff={daeff_log:.4g}, cd_in={cdin_log:.4g} "
                f"[{phase_source}; check={consistency}]"
            )
        else:
            phase_note = ", phase coordinates unresolved"
        print(
            f"[ok] {name}: {result.metrics.get('manuscript_regime', 'unclassified')} "
            f"[{result.metrics.get('classification_confidence', 'unknown')} confidence], "
            f"Kf/K0={result.metrics['conductivity_final_over_initial']:.4g}, "
            f"O_K={result.metrics['conductivity_oscillation_index']:.3f}, "
            f"Q20 events={event_count}, focusing={focus:.3f}{phase_note}"
        )

    if not results:
        return 1

    write_summary_csv(root_output / "aggregate_summary.csv", results)
    phase_audit_rows = []
    for result in results:
        m = result.metrics
        phase_audit_rows.append({
            "run": result.name,
            "source_directory": str(result.source_dir.resolve()),
            "params_path": str((result.source_dir / "params.txt").resolve()),
            "config_path": m.get("phase_config_path", ""),
            "Da_eff_used": m.get("Da_eff"),
            "cd_in_used": m.get("cd_in"),
            "inv_Da_eff": m.get("inv_Da_eff"),
            "inv_cd_in": m.get("inv_cd_in"),
            "config_Da_eff": m.get("config_Da_eff"),
            "config_cd_in": m.get("config_cd_in"),
            "config_dirname": m.get("phase_config_dirname", ""),
            "dirname_Da_eff": m.get("phase_config_dirname_Da_eff"),
            "dirname_cd_in": m.get("phase_config_dirname_cd_in"),
            "folder_Da_eff": m.get("phase_source_path_Da_eff"),
            "folder_cd_in": m.get("phase_source_path_cd_in"),
            "config_check": m.get("phase_config_consistency", ""),
            "config_mismatch": m.get("phase_config_mismatch", False),
            "metadata_Da_eff": m.get("phase_metadata_Da_eff"),
            "metadata_cd_in": m.get("phase_metadata_cd_in"),
            "metadata_conflict": m.get("phase_metadata_conflict", False),
            "metadata_conflict_detail": m.get("phase_metadata_conflict_detail", ""),
            "parameter_source": m.get("phase_parameter_source", ""),
        })
    _write_dict_rows(root_output / "phase_config_audit.csv", phase_audit_rows)
    with (root_output / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [json_safe(result.metrics) for result in results],
            handle,
            indent=2,
            sort_keys=True,
        )
    write_metric_definitions(root_output / "metric_definitions.txt")
    plot_comparisons(results, root_output, settings)
    plot_phase_diagrams(results, root_output, settings)

    if failures:
        failure_path = root_output / "failed_runs.txt"
        failure_path.write_text(
            "\n".join(f"{path}\t{message}" for path, message in failures) + "\n",
            encoding="utf-8",
        )
        print(f"Completed with {len(failures)} failed run(s); see {failure_path}")

    print(f"Post-processing written to: {root_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
