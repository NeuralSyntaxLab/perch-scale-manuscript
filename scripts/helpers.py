from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from scripts import _paths as P
import matplotlib.pyplot as plt


def read_timeseries(
    file_path: str | Path,
    low_thrd: float = 2,
    high_thrd: float = 30,
    start_date: str | None = None,
    end_date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    *,
    keep_out_of_range: bool = False,  # True => mask to NaN; False => drop rows.
) -> pd.DataFrame:
    """
    Read a per-second weight CSV (or CSV.gz) with columns [Time, weight].
    - Automatically detects gzip vs. plain CSV via filename (.csv or .csv.gz).
    - Applies value thresholds and optional date/time filters.
    - If keep_out_of_range=True: keep all rows, set out-of-range weights to NaN.
      Else: drop out-of-range rows (legacy behavior).
    """

    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(p)

    # Guard: only allow .csv or .csv.gz
    if not (p.suffix == ".csv" or (p.suffix == ".gz" and p.name.endswith(".csv.gz"))):
        raise ValueError(f"Unsupported file type: {p.name}. Expected .csv or .csv.gz")

    # Pandas will infer compression from the suffix (.gz) automatically.
    # parse_dates for 'Time' to get datetime64 dtype directly.
    df = pd.read_csv(p, compression="infer", parse_dates=["Time"])

    # Normalize/verify column names
    cols = {c.lower(): c for c in df.columns}
    if "time" not in cols:
        raise ValueError("CSV must contain a 'Time' column.")
    if "weight" not in cols:
        # Fallback: use second column as 'weight' if needed
        if df.shape[1] < 2:
            raise ValueError("Could not infer 'weight' column.")
        # rename second column to 'weight' (preserving 'Time')
        second_col = [c for c in df.columns if c != cols["time"]][0]
        df = df.rename(columns={second_col: "weight"})
    else:
        # Standardize exact casing
        if cols["weight"] != "weight":
            df = df.rename(columns={cols["weight"]: "weight"})

    # Ensure numeric weight
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Threshold mask
    mask = df["weight"].between(low_thrd, high_thrd)

    if keep_out_of_range:
        df.loc[:, "weight"] = df["weight"].where(mask)
    else:
        df = df.loc[mask].copy()

    # Date range filters (inclusive)
    if start_date is not None:
        df = df.loc[df["Time"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df.loc[df["Time"] <= pd.to_datetime(end_date)]

    # Time-of-day filters (inclusive; compares only the clock time)
    if start_time is not None:
        st = pd.to_datetime(start_time).time()
        df = df.loc[df["Time"].dt.time >= st]
    if end_time is not None:
        et = pd.to_datetime(end_time).time()
        df = df.loc[df["Time"].dt.time <= et]

    return df.sort_values("Time").reset_index(drop=True)


def find_bird_file(bird_id: str | int, birds_dir: Path = P.BIRDS) -> Path:
    """
    Return the path to the bird CSV in data/birds/, accepting either
    'bird_<ID>_weight_report.csv.gz' or '.csv'.
    """
    birds_dir = birds_dir or P.BIRDS  # use default if not provided
    bird_id = str(bird_id)
    candidates = [
        birds_dir / f"bird_{bird_id}_weight_report.csv.gz",
        birds_dir / f"bird_{bird_id}_weight_report.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No CSV found for bird '{bird_id}'. Tried: {', '.join(str(c) for c in candidates)}"
    )


def find_stable_estimates(df, win_size=10, step=2, weight_fraction=0.09, reference_weight=None):
    """
    Compute 'stable' perch-scale estimates from a time series by scanning fixed-size windows
    and retaining those whose standard deviation (SD) is less than or equal to a threshold
    defined as (weight_fraction * reference_weight).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns:
            - 'Time': datetime-like or numeric, assumed sorted and regularly sampled elsewhere
            - 'weight': numeric grams, pre-filtered for artifacts elsewhere
    win_size : int, default 10
        Window size in *samples*. At ~1 Hz this corresponds to ~10 seconds.
    step : int, default 2
        Stride between successive window starts (in samples).
    weight_fraction : float, default 0.09
        Fraction of the reference weight used as the SD stability threshold.
        For the canonical method, 0.09 => SD ≤ 9% of manual weight.
    reference_weight : float or None, default None
        Manual bird weight in grams. If None, a fallback estimate is derived from
        the mode of df['weight'] (or from the mean if a mode is unavailable).
        A warning is issued when using this fallback.

    Returns
    -------
    stable_df : pandas.DataFrame
        DataFrame with the mean 'weight' of each stable window at its center 'Time':
            - 'Time'
            - 'weight'

    Notes
    -----
    - Stability criterion: SD (computed with ddof=1) ≤ weight_fraction * reference_weight.
    - For typical use (10-sample windows over ~600k samples), using ddof=1 (sample SD)
      is recommended; the difference from ddof=0 is small but ddof=1 is the standard
      unbiased estimator for windowed samples.
    """
    if win_size <= 0:
        raise ValueError("win_size must be a positive integer.")
    if step <= 0:
        raise ValueError("step must be a positive integer.")
    if 'Time' not in df.columns or 'weight' not in df.columns:
        raise ValueError("df must contain 'Time' and 'weight' columns.")

    weights = df['weight'].to_numpy()
    times = df['Time'].to_numpy()

    n = len(weights)
    if n < win_size:
        # Not enough samples to form a single window
        return pd.DataFrame({'Time': pd.Series([], dtype=df['Time'].dtype),
                             'weight': pd.Series([], dtype=float)})

    # Derive reference weight if not supplied (robust fallback)
    ref_w = reference_weight
    if ref_w is None:
        mode_weight = pd.Series(weights).mode(dropna=True)
        if len(mode_weight) == 0:
            ref_w = float(np.nanmean(weights))  # last-resort fallback
            warnings.warn(
                "reference_weight not provided; falling back to mean weight of series. "
                "Provide the manual weight for canonical behavior."
            )
        else:
            ref_w = float(mode_weight.mean())
            warnings.warn(
                "reference_weight not provided; falling back to mode-based estimate. "
                "Provide the manual weight for canonical behavior."
            )
        if pd.isna(ref_w):
            # All-NaN case after fallback
            return pd.DataFrame({'Time': pd.Series([], dtype=df['Time'].dtype),
                                 'weight': pd.Series([], dtype=float)})
        if ref_w < 12:
            warnings.warn(
                f"Derived reference weight {ref_w:.2f} g seems unusually low. "
                "Check data quality or pass a manual reference weight."
            )

    std_threshold = weight_fraction * ref_w

    means = []
    center_times = []

    # Sliding window scan
    # Use sample SD (ddof=1) as recommended for finite windows
    for start in range(0, n - win_size + 1, step):
        win = weights[start:start + win_size]
        # Skip windows with NaNs
        if np.isnan(win).any():
            continue
        win_std = np.std(win, ddof=1)
        if win_std <= std_threshold:
            win_mean = float(np.mean(win))
            center_idx = start + win_size // 2
            means.append(win_mean)
            center_times.append(times[center_idx])

    if not means:
        return pd.DataFrame({'Time': pd.Series([], dtype=df['Time'].dtype),
                             'weight': pd.Series([], dtype=float)})

    stable_df = pd.DataFrame({'Time': np.array(center_times), 'weight': np.array(means)})
    return stable_df



# ---------- helpers for longitudinal & summary analysis ----------

def _ensure_fig_ax(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax

def _ref_and_thresholds(bird_id, weights_dict, df, low_thrd, high_thrd, tol):
    # weights_dict keyed by canonical IDs (e.g., '1','2','3', 'p1','t1')
    aw = weights_dict.get(str(bird_id), [])
    aw_vals = [v for v in aw if v is not None]
    ref = np.mean(aw_vals) if len(aw_vals) else (df['weight'].mean() if not df['weight'].empty else None)
    if ref is not None:
        low_thrd, high_thrd = ref * (1 - tol), ref * (1 + tol)
    return ref, low_thrd, high_thrd, aw

def _downsample(df, raw_downsample, raw_day, raw_night):
    if raw_day is not None and raw_night is not None and not df.empty:
        df = df.copy()
        df['hour'] = df['Time'].dt.hour
        df['is_night'] = ((df['hour'] >= 20) | (df['hour'] < 6))
        return (pd.concat([
            df.loc[~df['is_night']].iloc[::max(1, int(raw_day))],
            df.loc[df['is_night']].iloc[::max(1, int(raw_night))]
        ]).sort_index())
    return df.iloc[::max(1, int(raw_downsample))]

# ---------- Daily mode from all data (used in summary) ----------
def calc_mode_per_day(df, actual_weight=None, tolerance_fraction=0.3):
    # Returns dict: date -> mode(weight) within [ref*(1±tol)], else np.nan
    out = {}
    if df.empty:
        return out
    ref = actual_weight if actual_weight is not None else df['weight'].median()
    low, high = ref*(1 - tolerance_fraction), ref*(1 + tolerance_fraction)

    df_day = df[(df['weight'] >= low) & (df['weight'] <= high)].copy()
    if df_day.empty:
        return {d: np.nan for d in pd.to_datetime(df['Time']).dt.date.astype(str).unique()}

    df_day['date'] = pd.to_datetime(df_day['Time']).dt.date.astype(str)
    for d, g in df_day.groupby('date'):
        m = g['weight'].mode()
        out[d] = float(m.iloc[0]) if not m.empty else np.nan
    return out

def compute_summary_metrics(
    bird_id,
    weight_report_csv,
    weights_dict,
    low_thrd=1, high_thrd=30,
    win_size=10, step=10, weight_fraction=0.09,
    start_date=None, end_date=None, start_time=None, end_time=None,
    tolerance_fraction=0.3,
    weights_csv_path="daily_manual_weights.csv",
    keep_out_of_range=False,
) -> pd.DataFrame:
    """Compute per-day summary metrics of 'stable' and 'mode' perch-scale weight estimates.

    This function reads a per-second weight time series (via :func:`read_timeseries`),
    applies value and date/time filters, restricts values to a tolerance range around
    a reference/manual weight, finds temporally "stable" windows using
    :func:`find_stable_estimates`, and aggregates per-day statistics.

    The returned DataFrame contains one row per date from the (filtered) input
    series and includes median/mean/std/count of stable-window estimates,
    a daily stable-mode, a mode estimate computed from all (filtered) data for the
    day, and an optional true/manual weight lookup from a wide CSV provided by
    ``weights_csv_path``.

    Parameters
    ----------
    bird_id : str or int
        Bird identifier (used for reference weight lookup and to label results).
    weight_report_csv : str or pathlib.Path
        Path to the per-second weight CSV file for this bird. Accepted suffixes:
        ``.csv`` and ``.csv.gz``. The file must contain a ``Time`` column and a
        weight column (named ``weight`` or inferred as the second column).
    weights_dict : mapping
        Dictionary of manual/annotated weights keyed by bird id; used by
        ``_ref_and_thresholds`` to derive a reference weight when available.
    low_thrd, high_thrd : float, optional
        Absolute min/max thresholds applied by :func:`read_timeseries` before
        further processing (defaults: 1, 30 grams).
    win_size : int, optional
        Window size in samples for stability scanning (default 10 samples).
    step : int, optional
        Stride between successive window starts in samples (default 10).
    weight_fraction : float, optional
        Fraction of the reference weight used as the SD stability threshold
        (default 0.09 -> 9% of reference weight).
    start_date, end_date : str or None, optional
        Inclusive date filters (ISO-style strings) applied to the time series.
    start_time, end_time : str or None, optional
        Clock-time-of-day filters (inclusive) applied to the series.
    tolerance_fraction : float, optional
        Fractional tolerance used to compute an allowed range around the
        reference weight. The code calls ``_ref_and_thresholds`` which scales
        the reference by (1 - tol, 1 + tol) (default 0.3 -> ±30%).
    weights_csv_path : str, optional
        Path to a "wide" CSV of manual/true weights indexed by a ``bird_id``
        column. If present, the function will attempt to map per-day true
        weights into the summary (default: "daily_manual_weights.csv").
    keep_out_of_range : bool, optional
        If True, rows outside ``[low_thrd, high_thrd]`` are kept but their
        ``weight`` values are set to NaN; otherwise out-of-range rows are
        dropped (default False).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns (in this order):
        - ``date`` (str, YYYY-MM-DD)
        - ``stable_median`` (median of stable-window means for that day)
        - ``stable_mode`` (mode of stable-window means for that day)
        - ``stable_mean`` (mean of stable-window means for that day)
        - ``stable_std`` (std dev of stable-window means for that day)
        - ``stable_count`` (number of stable windows detected for that day)
        - ``mode_estimate`` (mode estimate computed from all data that day)
        - ``true_weight`` (optional manual weight looked up from ``weights_csv_path``)
        - ``bird_id`` (stringified bird identifier)

    Notes
    -----
    - If no data remain after filtering, an empty DataFrame with the same
      column layout is returned.
    - If no stable windows are found, an empty DataFrame with the same column
      layout is returned.
    - The function relies on :func:`find_stable_estimates` and
      :func:`calc_mode_per_day` for the windowing and mode computations.

    Examples
    --------
    >>> summary = compute_stable_weight_summary('5', 'data/birds/bird_5_weight_report.csv', weights_dict)

    """
    df0 = read_timeseries(
        file_path=weight_report_csv,
        low_thrd=low_thrd, high_thrd=high_thrd,
        start_date=start_date, end_date=end_date,
        start_time=start_time, end_time=end_time,
        keep_out_of_range=keep_out_of_range
    )
    if df0.empty:
        print(f"No data after filtering for {bird_id}.")
        return pd.DataFrame(columns=['date','stable_median','stable_mode','stable_mean','stable_std','stable_count','mode_estimate','true_weight','bird_id'])

    all_dates_orig = pd.to_datetime(df0['Time']).dt.date.astype(str).unique()

    ref, lo, hi, _ = _ref_and_thresholds(str(bird_id), weights_dict, df0, low_thrd, high_thrd, tolerance_fraction)
    df = df0[(df0['weight'] >= lo) & (df0['weight'] <= hi)].copy()

    rel = find_stable_estimates(
        df, win_size=win_size, step=step,
        weight_fraction=weight_fraction, reference_weight=ref
    )
    if rel is None or len(rel)==0:
        print("No stable windows found.")
        return pd.DataFrame(columns=['date','stable_median','stable_mode','stable_mean','stable_std','stable_count','mode_estimate','true_weight','bird_id'])

    rel['date'] = pd.to_datetime(rel['Time']).dt.date.astype(str)
    g = rel.groupby('date')['weight']
    summary = g.agg(
        stable_median='median',
        stable_mean='mean',
        stable_std='std',
        stable_count='size'
    ).reset_index()

    # Daily stable mode
    stable_mode = g.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index(name='stable_mode')
    summary = summary.merge(stable_mode, on='date', how='left')

    # Mode estimate from all data per day
    mode_map = calc_mode_per_day(df, actual_weight=ref, tolerance_fraction=tolerance_fraction)
    summary['mode_estimate'] = summary['date'].map(mode_map)

    # Optional: true weights (from a wide CSV indexed by bird_id)
    summary['true_weight'] = np.nan
    try:
        wdf = pd.read_csv(weights_csv_path)  # don't force index
        if 'bird_id' not in wdf.columns:
            raise ValueError("CSV must contain a 'bird_id' column")

        # Match the dtype of bird_id to the CSV
        if pd.api.types.is_numeric_dtype(wdf['bird_id']):
            bid = pd.to_numeric(bird_id, errors='coerce')
        else:
            wdf['bird_id'] = wdf['bird_id'].astype(str).str.strip()
            bid = str(bird_id).strip()

        wdf = wdf.set_index('bird_id')

        if bid in wdf.index:
            row = wdf.loc[bid]  # Series with columns like '2025-06-11', ...
            # Map summary dates to that row; non-existing dates become NaN
            summary['true_weight'] = pd.to_numeric(summary['date'].map(row), errors='coerce')
        else:
            print(f"bird_id {bird_id!r} not found in '{weights_csv_path}'")

    except Exception as e:
        print(f"Couldn't load true weights from '{weights_csv_path}': {e}")

    summary['bird_id'] = str(bird_id)

    # Add missing dates
    all_dates = pd.Series(all_dates_orig, name='date')
    summary = all_dates.to_frame().merge(summary, on='date', how='left').sort_values('date').reset_index(drop=True)
    summary['stable_count'] = summary['stable_count'].fillna(0).astype(int)

    return summary[['date','stable_median','stable_mode','stable_mean','stable_std','stable_count','mode_estimate','true_weight','bird_id']]
