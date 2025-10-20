import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from analyze_reliable_weight import calc_reliable_measure

def read_timeseries(
    file_path,
    low_thrd=2,
    high_thrd=30,
    start_date=None,
    end_date=None,
    start_time=None,
    end_time=None,
    *,
    keep_out_of_range=False,   # True => mask to NaN (keep timestamps). False => drop rows.
):
    """
    Read a per-second weight CSV and (optionally) mask out-of-range weights while preserving timestamps.
    - If keep_out_of_range=True: out-of-range values are set to NaN, rows are kept.
    - If keep_out_of_range=False: out-of-range rows are removed (legacy behavior).
    Date/time filters are applied on 'Time' only (rows retained even if 'weight' is NaN).
    """
    df = pd.read_csv(file_path)

    # Ensure expected columns
    if 'Time' not in df.columns:
        raise ValueError("CSV must contain a 'Time' column (or rename before calling).")
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

    if 'weight' not in df.columns:
        # Fallback: use second column as 'weight' if needed
        if len(df.columns) < 2:
            raise ValueError("Could not infer 'weight' column.")
        df = df.rename(columns={df.columns[1]: 'weight'})

    # Coerce numeric and build mask
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    mask = df['weight'].between(low_thrd, high_thrd)

    if keep_out_of_range:
        # Preserve all rows; set out-of-range weights to NaN
        df.loc[:, 'weight'] = df['weight'].where(mask)
    else:
        # Old behavior: drop rows outside thresholds
        df = df.loc[mask].copy()

    # Date filters (inclusive)
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        df = df.loc[df['Time'] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        df = df.loc[df['Time'] <= end_ts]

    # Time-of-day filters (e.g., "12:00")
    if start_time is not None:
        st = pd.to_datetime(start_time).time()
        df = df.loc[df['Time'].dt.time >= st]
    if end_time is not None:
        et = pd.to_datetime(end_time).time()
        df = df.loc[df['Time'].dt.time <= et]

    # Stable ordering and minimal index surprises
    df = df.sort_values('Time').reset_index(drop=True)
    return df

def describe_timeseries(df):
    if df.empty:
        print("DataFrame is empty.")
        return
    df['date'] = df['Time'].dt.date.astype(str)
    print(f"Number of records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Weight range: {df['weight'].min()} to {df['weight'].max()}")
    print(f"Mean weight: {df['weight'].mean():.2f} g")
    return

def calc_mode_per_day(df, actual_weight, tolerance_fraction=0.3):
    df['date'] = df['Time'].dt.date.astype(str)
    mode_per_day = {}
    lower = (1 - tolerance_fraction) * actual_weight
    upper = (1 + tolerance_fraction) * actual_weight
    for date, grp in df.groupby('date'):
        modes = grp['weight'].mode()
        val = np.nan
        if not modes.empty and not np.isnan(actual_weight):
            candidates = modes[(modes >= lower) & (modes <= upper)]
            # candidates = modes
            if len(candidates) == 1:
                val = round(candidates.iloc[0], 2)
            elif len(candidates) > 1:
                # Find the mode closest to the actual value
                find_best_mode = np.argmin([np.abs(c - actual_weight) for c in candidates])
                val = round(candidates.iloc[find_best_mode], 2)
        mode_per_day[date] = val
    return mode_per_day



def reliable_means(df, win_size=50, step=5, std_percentile=None, weight_fraction=0.05, reference_weight=None, tolerance_fraction=0.3):
    if len(df) < win_size:
        return {}
    rel = calc_reliable_measure(df, win_size=win_size, step=step, std_percentile=std_percentile, weight_fraction=weight_fraction, reference_weight=reference_weight, tolerance_fraction=tolerance_fraction)
    rel['date'] = pd.to_datetime(rel['Time']).dt.date.astype(str)
    return rel.groupby('date')['Weight'].mean().to_dict()

def stable_modes_per_day(df, win_size=50, step=5, std_percentile=None, weight_fraction=0.05, reference_weight=None):
    if len(df) < win_size:
        return {}
    rel = calc_reliable_measure(df, win_size=win_size, step=step, std_percentile=std_percentile, weight_fraction=weight_fraction, reference_weight=reference_weight)
    rel['date'] = pd.to_datetime(rel['Time']).dt.date.astype(str)
    mode_per_day = {}
    for date, grp in rel.groupby('date'):
        modes = grp['Weight'].mode()
        val = np.nan
        if not modes.empty:
            val = modes.iloc[0]  # Take the first mode as the stable value
            # if len(modes) == 1:
            #     val = round(modes.iloc[0], 2)
            # elif len(modes) > 1:
            #     # Find the mode closest to the actual value
            #     find_best_mode = np.argmin([np.abs(c - reference_weight) for c in modes])
            #     val = round(modes.iloc[find_best_mode], 2)
        mode_per_day[date] = val
    return mode_per_day


def scatter_plot(
    df, 
    x_col, 
    y_col, 
    title, 
    group_col='bird', 
    top_n=None, 
    tolerance_fraction=0.3,
    exclude_out_of_tolerance=False,
    color_by_bird=True
):
    clean = df.dropna(subset=[x_col, y_col]).copy()

    # Exclude estimates not within tolerance if requested
    if exclude_out_of_tolerance:
        tol = np.abs(clean[y_col]) * tolerance_fraction
        abs_err = np.abs(clean[x_col] - clean[y_col])
        clean = clean[abs_err <= tol]

    # Select top N most accurate points *per group* if requested
    if top_n is not None and group_col and group_col in clean.columns:
        clean['abs_err'] = np.abs(clean[x_col] - clean[y_col])
        # Get top N per group
        top_rows = []
        for group, grp in clean.groupby(group_col):
            top_rows.append(grp.nsmallest(top_n, 'abs_err'))
        clean = pd.concat(top_rows)
    elif top_n is not None:
        # Fallback: just get top N globally if group_col not used
        clean['abs_err'] = np.abs(clean[x_col] - clean[y_col])
        clean = clean.nsmallest(top_n, 'abs_err')
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color logic
    if color_by_bird and group_col and group_col in clean.columns:
        for group, grp in clean.groupby(group_col):
            ax.scatter(grp[x_col], grp[y_col], label=str(group), alpha=0.7)
    else:
        ax.scatter(clean[x_col], clean[y_col], alpha=0.7, color='blue')

    if not clean.empty:
        # Correlation
        r, p = pearsonr(clean[x_col], clean[y_col])
        # Linear regression
        slope, intercept, r_value, p_reg, stderr = linregress(clean[x_col], clean[y_col])
        r_squared = r_value ** 2
        # MAE
        mae = np.abs(clean[x_col] - clean[y_col]).mean()

        # Scatter, identity line
        min_val = min(clean[[x_col, y_col]].min())
        max_val = max(clean[[x_col, y_col]].max())
        ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', label='x=y')

        # Annotate
        eqn = f"y = {slope:.2f}x + {intercept:.2f}"
        stats = f"r = {r:.3f}\np = {p:.3g}\nR² = {r_squared:.3f}\nMAE = {mae:.2f} g"
        ax.text(
            0.05, 0.95, eqn + "\n" + stats, transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
    else:
        r, p, r_squared, mae = np.nan, np.nan, np.nan, np.nan

    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(
        title + (f" (Top {top_n} Per {group_col})" if top_n and group_col else "")
    )
    if color_by_bird and group_col and group_col in clean.columns:
        ax.legend()
    else:
        # Add MAE to legend if not grouping by bird
        ax.legend([f"x=y\nMAE = {mae:.2f} g"])
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    if not np.isnan(r):
        print(
            f"Correlation between {x_col} and {y_col}" +
            (f" (top {top_n} per {group_col})" if top_n and group_col else "") +
            f": r = {r:.3f}, p = {p:.3g}, MAE = {mae:.2f} g"
        )
    else:
        print(f"No data for correlation {x_col} vs {y_col}" + (f" (top {top_n} per {group_col})" if top_n and group_col else ""))

def plot_weight_accuracy_scatter(
    df, 
    estimate_col='reliable_full', 
    actual_col='actual', 
    group_col='bird', 
    title='Automated vs. Actual Weight', 
    tolerance_fraction=0.3, 
    top_n=None,
    show_regression=True, 
    show_identity=True,
    figsize=(8, 8)
):
    """
    Scatter plot comparing estimated and actual weights, with error metrics and annotation.

    Parameters:
    - df: DataFrame containing columns for estimate, actual, and group (bird).
    - estimate_col: str, column name for system estimates.
    - actual_col: str, column name for ground-truth values.
    - group_col: str or None, for coloring by bird.
    - title: str, plot title.
    - tolerance_fraction: float, fraction of actual weight for relative "within tolerance".
    - top_n: int or None, if specified, show top N most accurate points.
    - show_regression: bool, draw regression line.
    - show_identity: bool, draw identity line.
    - figsize: tuple, figure size.
    """
    clean = df.dropna(subset=[estimate_col, actual_col]).copy()
    if clean.empty:
        print("No data to plot.")
        return

    # Assign axes: x = actual, y = estimate
    x = clean[actual_col].values
    y = clean[estimate_col].values
    abs_err = np.abs(x - y)
    clean['abs_err'] = abs_err

    # TOP N FILTERING
    if top_n is not None:
        if group_col and group_col in clean.columns:
            # Top N per group
            top_rows = []
            for group, grp in clean.groupby(group_col):
                top_rows.append(grp.nsmallest(top_n, 'abs_err'))
            clean = pd.concat(top_rows)
        else:
            # Top N overall
            clean = clean.nsmallest(top_n, 'abs_err')

        x = clean[actual_col].values
        y = clean[estimate_col].values
        abs_err = clean['abs_err'].values

    # Relative tolerance for each point (tolerance is based on actual)
    tol = np.abs(x) * tolerance_fraction
    within_tol = (abs_err <= tol)
    percent_within = within_tol.mean() * 100
    n_points = len(x)

    # Per-bird stats for within tolerance
    per_bird_summary = {}
    if group_col and group_col in clean.columns:
        for group, grp in clean.groupby(group_col):
            x_bird = grp[actual_col].values
            y_bird = grp[estimate_col].values
            tol_bird = np.abs(x_bird) * tolerance_fraction
            abs_err_bird = np.abs(x_bird - y_bird)
            within_bird = (abs_err_bird <= tol_bird).mean() * 100
            per_bird_summary[group] = within_bird

    # Error metrics
    mae = abs_err.mean()
    rmse = np.sqrt(((x - y) ** 2).mean())
    bias = (y - x).mean()

    # Association metrics
    r, p_corr = pearsonr(x, y)
    slope, intercept, r_value, p_slope, stderr = linregress(x, y)
    r_squared = r_value ** 2

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Color map for distinct bird colors
    if group_col and group_col in clean.columns:
        birds = list(clean[group_col].unique())
        colors = plt.colormaps['tab10'].resampled(len(birds))
        for i, (group, grp) in enumerate(clean.groupby(group_col)):
            mean_x = grp[actual_col].mean()
            mean_y = grp[estimate_col].mean()
            std_x = grp[actual_col].std()
            std_y = grp[estimate_col].std()
            color = colors(i)
            # Draw "plus": horizontal and vertical lines centered at mean
            ax.plot(
                [mean_x - std_x, mean_x + std_x], [mean_y, mean_y], 
                color=color, lw=2
            )
            ax.plot(
                [mean_x, mean_x], [mean_y - std_y, mean_y + std_y],
                color=color, lw=2, label=str(group)
            )
            # Optionally, plot mean as a point (optional)
            ax.scatter([mean_x], [mean_y], color=color, s=60, zorder=5)
    else:
        # No grouping, plot all as one
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        std_x = np.std(x)
        std_y = np.std(y)
        ax.plot([mean_x - std_x, mean_x + std_x], [mean_y, mean_y], color='black', lw=2)
        ax.plot([mean_x, mean_x], [mean_y - std_y, mean_y + std_y], color='black', lw=2)
        ax.scatter([mean_x], [mean_y], color='black', s=60, zorder=5)

    # Identity line (x = y)
    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    if show_identity:
        ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', label='y = x')

    # Regression line
    if show_regression:
        reg_x = np.linspace(min_val, max_val, 100)
        reg_y = slope * reg_x + intercept
        ax.plot(reg_x, reg_y, '-', color='red', label=f'Regression')

    # Per-bird within-tolerance (short summary)
    per_bird_text = ", ".join([f"{bird}: {pct:.1f}%" for bird, pct in per_bird_summary.items()])

    # Annotate metrics
    annotation = (
        f"Tolerance: ±{tolerance_fraction*100:.1f}% of actual\n"
        f"Within tolerance: {percent_within:.1f}%\n"
        # f"By bird: {per_bird_text}\n"
        f"$MAE$ = {mae:.2f} g\n"
        f"$RMSE$ = {rmse:.2f} g\n"
        f"Bias = {bias:+.2f} g\n"
        f"$r$ = {r:.3f}, $p$ = {p_corr:.3g}\n"
        f"$R^2$ = {r_squared:.3f}\n"
        f"y = {slope:.2f}x + {intercept:.2f}\n"
        f"$n$ = {n_points}"
        + (f"\nTop {top_n} per {group_col}" if top_n and group_col else (f"\nTop {top_n} overall" if top_n else ""))
    )
    ax.text(
        0.05, 0.95, annotation,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75)
    )

    ax.set_xlabel("Actual Weight (g)")
    ax.set_ylabel("Estimated Weight (g)")
    ax.set_title(title)
    # show legend in bottom right corner
    ax.legend(loc='lower right', title=group_col.capitalize())
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print summary r and p, and MAE
    print(f"Pearson r = {r:.3f}, p = {p_corr:.3g}, MAE = {mae:.2f} g")
    print(f"Linear regression: y = {slope:.2f}x + {intercept:.2f}, R^2 = {r_squared:.3f}, n = {n_points}")
    

def main(
        window_size=10, step=10, time_split=False, std_percentile=None,
        weight_fraction=0.09, start_date=None, end_date=None, start_time=None,
        end_time=None, top_n=None, tolerance_fraction=0.3, low_threshold=1, high_threshold=30,
        exclude_out_of_tolerance=False, color_by_bird=True
    ):
    actual_df = pd.read_csv('weights.csv', index_col=0)
    date_cols = [c for c in actual_df.columns if c not in ['mean', 'std']]


    # # Apply cutoffs based on +/-X% of actual weight
    # # Get actual weights for this bird
    # actual_weights = weights_dict.get(bird_name, {})
    # if actual_weights:
    #     values = [v for v in actual_weights if v is not None]
    #     reference_weight = np.mean(values) if values else None
    # else:
    #     reference_weight = None

    # # Determine +-30% (or other, based on user preference) cutoff thresholds based on reference weight
    # if reference_weight is not None:
    #     low_thrd = reference_weight * (1 - tolerance_fraction)
    #     high_thrd = reference_weight * (1 + tolerance_fraction)
    # else:
    #     low_thrd = low_thrd
    #     high_thrd = high_thrd


    records = []
    ts_files = glob('a_*_weight_report.csv')
    for file in ts_files:
        bird = Path(file).stem.split('_')[1]
        print(f"Processing file: {file} for bird: {bird}")
        if bird not in actual_df.index:
            continue
        
        actual_map = {d: actual_df.at[bird, d] for d in date_cols}
        actual_weight = actual_df.at[bird, 'mean']

        df_full = read_timeseries(file, start_date=start_date, end_date=end_date, low_thrd=low_threshold, high_thrd=high_threshold)
        mode_map_full = calc_mode_per_day(df_full, actual_weight=actual_weight, tolerance_fraction=tolerance_fraction)
        rel_full = reliable_means(df_full, win_size=window_size, step=step, std_percentile=std_percentile, weight_fraction=weight_fraction, reference_weight=actual_weight)

        if time_split:
            df_time_split = read_timeseries(file, start_date=start_date, end_date=end_date, start_time=start_time, end_time=end_time)
            mode_map_time_split = calc_mode_per_day(df_time_split, actual_weight=actual_weight, tolerance_fraction=tolerance_fraction)
            rel_time_split = reliable_means(df_time_split, win_size=window_size, step=step, std_percentile=std_percentile, weight_fraction=weight_fraction, reference_weight=actual_weight)
        
       
        for date in date_cols:
            records.append({
                'bird': bird,
                'date': date,
                'actual': actual_map.get(date, np.nan),
                'mode_full': mode_map_full.get(date, np.nan),
                'mode_time_split': mode_map_time_split.get(date, np.nan) if time_split else np.nan,
                'reliable_full': rel_full.get(date, np.nan),
                'reliable_time_split': rel_time_split.get(date, np.nan) if time_split else np.nan
            })

    results = pd.DataFrame(records)
    print(results.shape)
    # plot_weight_accuracy_scatter(
    #     results,
    #     estimate_col='mode_full', 
    #     actual_col='actual',
    #     group_col='bird',
    #     title='Mode estimate vs. Actual Weight (All data)',
    #     tolerance_fraction=tolerance_fraction,  
    #     show_regression=False,
    # )

    # plot_weight_accuracy_scatter(
    #     results,
    #     estimate_col='reliable_full', 
    #     actual_col='actual',
    #     group_col='bird',
    #     title='Reliable estimate vs. Actual Weight (All data)',
    #     tolerance_fraction=tolerance_fraction,          
    #     show_regression=False,
    # )

    scatter_plot(
        results, 
        'mode_full', 
        'actual', 
        'Mode (All Data) vs Actual Weight',
        tolerance_fraction=tolerance_fraction,
        exclude_out_of_tolerance=exclude_out_of_tolerance,
        color_by_bird=color_by_bird
    )
    scatter_plot(
        results, 
        'reliable_full', 
        'actual', 
        'Reliable (All Data) vs Actual',
        tolerance_fraction=tolerance_fraction,
        exclude_out_of_tolerance=exclude_out_of_tolerance,
        color_by_bird=color_by_bird
    )
    # if time_split:
    #     scatter_plot(results, 'mode_time_split', 'actual', 'Mode (Time Split) vs Actual')
    #     scatter_plot(results, 'reliable_time_split', 'actual', 'Reliable (Time Split) vs Actual')

    # N = top_n
    # # Top N most accurate
    # if N is not None:
    #     print(f"\n-- Top {N} Most Accurate Estimates --")
    #     scatter_plot(results, 'mode_full', 'actual', 'Mode (All Data) vs Actual', top_n=N)
    #     scatter_plot(results, 'reliable_full', 'actual', 'Reliable (All Data) vs Actual', top_n=N)
    #     if time_split:
    #         scatter_plot(results, 'mode_time_split', 'actual', 'Mode (Time Split) vs Actual', top_n=N)
    #         scatter_plot(results, 'reliable_time_split', 'actual', 'Reliable (Time Split) vs Actual', top_n=N)


if __name__ == '__main__':
    import json 
    weights_json = "weights.json"  # Path to actual weights JSON

    # Read weights.json and get bird names
    with open(weights_json, 'r') as f:
        weights_dict = json.load(f)
    weights_dict = {k: v for k, v in weights_dict.items() if k != 'start_date'}
    bird_names = list(weights_dict.keys())
    print(bird_names)
    
    main(window_size=10, step=10, std_percentile=None, 
         weight_fraction=0.09, start_date='2025-06-11', 
         time_split=False, start_time=None, end_time=None,
         low_threshold=1, high_threshold=50, tolerance_fraction=0.3,
         exclude_out_of_tolerance=True, color_by_bird=True)
