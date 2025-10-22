from __future__ import annotations
from pathlib import Path
import pandas as pd
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