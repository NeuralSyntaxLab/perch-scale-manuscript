from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from scripts import _paths as P


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
