import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# --- External libs you provide ---
from pipeline import read_timeseries, calc_mode_per_day
from analyze_reliable_weight import calc_reliable_measure


# ---------------------- Helpers ----------------------
def _ensure_fig_ax(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax

def _ref_and_thresholds(bird_name, weights_dict, df, low_thrd, high_thrd, tol):
    aw = weights_dict.get(bird_name, [])
    aw_vals = [v for v in aw if v is not None]
    ref = np.mean(aw_vals) if len(aw_vals) else (df['weight'].mean() if not df['weight'].empty else None)
    if ref is not None:
        low_thrd, high_thrd = ref * (1 - tol), ref * (1 + tol)
    return ref, low_thrd, high_thrd, aw

def _downsample(df, raw_downsample, raw_day, raw_night):
    if raw_day is not None and raw_night is not None:
        df['hour'] = df['Time'].dt.hour
        df['is_night'] = ((df['hour'] >= 20) | (df['hour'] < 6))
        return (pd.concat([
            df.loc[~df['is_night']].iloc[::max(1, int(raw_day))],
            df.loc[df['is_night']].iloc[::max(1, int(raw_night))]
        ]).sort_index())
    return df.iloc[::max(1, int(raw_downsample))]


# ---------------------- Longitudinal ----------------------
def plot_longitudinal(
    bird_name,
    weight_report_csv,
    weights_dict,
    low_thrd=1,
    high_thrd=30,
    win_size=10,
    step=10,
    std_percentile=None,
    weight_fraction=0.09,
    start_date=None,
    end_date=None,
    start_time=None,
    end_time=None,
    figsize=(12, 4),
    raw_downsample=5,
    rel_downsample=5,
    apply_ylim=False,
    add_actual_weights=True,
    ax=None,
    raw_downsample_day=None,
    raw_downsample_night=None,
    tolerance_fraction=0.3,
    keep_out_of_range=True,   # True => mask to NaN (keep timestamps). False => drop rows.
):
    df0 = read_timeseries(
        file_path=weight_report_csv,
        low_thrd=low_thrd,
        high_thrd=high_thrd,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        keep_out_of_range=keep_out_of_range
    )
    if df0.empty:
        print(f"No data after filtering for {bird_name}."); return None, None

    ref, lo, hi, actual_weights = _ref_and_thresholds(bird_name, weights_dict, df0, low_thrd, high_thrd, tolerance_fraction)
    mask = (df0['weight'] >= lo) & (df0['weight'] <= hi)
    if not mask.any():
        print(f"No data within thresholds for {bird_name} (ref: {ref}, lo: {lo}, hi: {hi})."); return None, None
    df = df0.assign(weight=df0['weight'].where(df0['weight'].between(lo, hi)))
    df['date'] = df['Time'].dt.date.astype(str)

    rel = calc_reliable_measure(
        df, win_size=win_size, step=step,
        std_percentile=std_percentile, weight_fraction=weight_fraction,
        reference_weight=ref
    )
    rel = rel if (rel is not None and len(rel)) else pd.DataFrame(columns=['Time','Weight'])
    rel['date'] = pd.to_datetime(rel['Time']).dt.date.astype(str) if not rel.empty else rel.get('date', pd.Series(dtype=str))

    fig, ax = _ensure_fig_ax(ax, figsize)
    df_plot = _downsample(df0, raw_downsample, raw_downsample_day, raw_downsample_night)
    if len(df_plot):
        ax.scatter(df_plot['Time'], df_plot['weight'], s=7, alpha=0.3, color='blue', label='Raw')

    if len(rel):
        rel_plot = rel.iloc[::max(1, int(rel_downsample))]
        ax.scatter(pd.to_datetime(rel_plot['Time']), rel_plot['Weight'], s=5, color='orange', label='Stable', zorder=3)

    for day in sorted(df['date'].unique()):
        ax.axvline(pd.to_datetime(day), color='k', linestyle='--', alpha=0.2, zorder=0)

    if add_actual_weights and isinstance(actual_weights, (list, tuple)) and len(actual_weights):
        days = sorted(df['date'].unique())
        for i, day in enumerate(days):
            if i >= len(actual_weights) or actual_weights[i] is None: continue
            day_center = pd.to_datetime(day) + pd.Timedelta(hours=12)
            val = actual_weights[i]
            ax.plot(day_center, val, marker='_', color='red', markersize=30, markeredgewidth=4, label='Actual' if i==0 else "", zorder=4)
            ax.plot(day_center, val*(1+tolerance_fraction), marker='_', color='black', markersize=15, markeredgewidth=2, label=f'±{int(100*tolerance_fraction)}%' if i==0 else "", zorder=5)
            ax.plot(day_center, val*(1-tolerance_fraction), marker='_', color='black', markersize=15, markeredgewidth=2, zorder=5)

    days = sorted(df['date'].unique())
    ax.set_xticks([pd.to_datetime(d) for d in days])
    ax.set_xticklabels(np.arange(len(days)), ha='right')
    if not df['Time'].empty:
        ax.set_xlim(df['Time'].min() - pd.Timedelta(hours=4), df['Time'].max() + pd.Timedelta(hours=4))
    if apply_ylim and not rel['Weight'].empty:
        ax.set_ylim(rel['Weight'].min() * 0.55, rel['Weight'].max() * 1.45)
    else:
        # Explicitly re-enable autoscaling and recompute from current artists
        ax.set_ylim(0, df0['weight'].max())

    ax.set_xlabel('Time (Days)'); ax.set_ylabel('Weight (g)'); ax.set_title(f'Bird {bird_name}: longitudinal')
    ax.legend(loc='best')
    plt.tight_layout()
    return fig, ax


# ---------------------- Summary (compute) ----------------------
def compute_reliable_weight_summary(
    bird_name,
    weight_report_csv,
    weights_dict,
    low_thrd=1,
    high_thrd=30,
    win_size=10,
    step=10,
    std_percentile=None,
    weight_fraction=0.09,
    start_date=None,
    end_date=None,
    start_time=None,
    end_time=None,
    tolerance_fraction=0.3,
    weights_csv_path="weights.csv",
    keep_out_of_range=False,
) -> pd.DataFrame:
    df0 = read_timeseries(
        file_path=weight_report_csv,
        low_thrd=low_thrd, high_thrd=high_thrd,
        start_date=start_date, end_date=end_date,
        start_time=start_time, end_time=end_time,
        keep_out_of_range=keep_out_of_range
    )
    if df0.empty:
        print(f"No data after filtering for {bird_name}.")
        return pd.DataFrame(columns=['date','stable_median','stable_mode','stable_mean','stable_std','stable_count','mode_estimate','true_weight','bird_name'])

    all_dates_orig = pd.to_datetime(df0['Time']).dt.date.astype(str).unique()

    ref, lo, hi, _ = _ref_and_thresholds(bird_name, weights_dict, df0, low_thrd, high_thrd, tolerance_fraction)
    df = df0[(df0['weight'] >= lo) & (df0['weight'] <= hi)].copy()

    rel = calc_reliable_measure(
        df, win_size=win_size, step=step,
        std_percentile=std_percentile, weight_fraction=weight_fraction,
        reference_weight=ref
    )
    if rel is None or len(rel)==0:
        print("No reliable windows found.")
        return pd.DataFrame(columns=['date','stable_median','stable_mode','stable_mean','stable_std','stable_count','mode_estimate','true_weight','bird_name'])

    rel['date'] = pd.to_datetime(rel['Time']).dt.date.astype(str)
    g = rel.groupby('date')['Weight']
    summary = g.agg(
        stable_median='median',
        stable_mean='mean',
        stable_std='std',
        stable_count='size'
    ).reset_index()
    # Fast daily mode for 'stable_mode'
    stable_mode = g.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index(name='stable_mode')
    summary = summary.merge(stable_mode, on='date', how='left')

    # Mode estimate from all data per day
    mode_map = calc_mode_per_day(df, actual_weight=ref, tolerance_fraction=tolerance_fraction)
    summary['mode_estimate'] = summary['date'].map(mode_map)

    # True weights (optional)
    summary['true_weight'] = np.nan
    try:
        wdf = pd.read_csv(weights_csv_path, index_col=0)
        if bird_name in wdf.index:
            summary['true_weight'] = pd.to_numeric(summary['date'].map(lambda d: wdf.loc[bird_name, d] if d in wdf.columns else np.nan), errors='coerce')
    except Exception as e:
        print(f"Note: could not load '{weights_csv_path}' ({e}).")

    summary['bird_name'] = bird_name
    
    # Add missing dates with NaNs but 0 in stable_count
    all_dates = pd.Series(all_dates_orig, name='date')
    summary = all_dates.to_frame().merge(summary, on='date', how='left').sort_values('date').reset_index(drop=True)
    summary['stable_count'] = summary['stable_count'].fillna(0).astype(int)

    return summary[['date','stable_median','stable_mode','stable_mean','stable_std','stable_count','mode_estimate','true_weight','bird_name']]


# ---------------------- Summary (plot) ----------------------
def plot_reliable_weight_summary(
    summary: pd.DataFrame,
    bird_name: str,
    main_estimator='mean',
    plot_mode=True,
    plot_mean=True,
    plot_median=True,
    plot_actual_weights=False,
    ax=None,
    figsize=(12, 4),
    drop_last=False,   # preserves earlier behavior
):
    if summary is None or summary.empty:
        print("Empty summary; nothing to plot."); return None, None

    fig, ax1 = _ensure_fig_ax(ax, figsize)
    s = summary.iloc[:-1] if drop_last and len(summary) > 1 else summary

    est_key = {'mean':'stable_mean','median':'stable_median','mode':'stable_mode'}.get(main_estimator, 'stable_mean')
    x = np.arange(len(s))
    bw = 0.2

    ax1.bar(x, s[est_key], yerr=s['stable_std'], width=bw, alpha=1, label=f'{main_estimator.capitalize()} (bar)', color='orange', capsize=4)

    off = 0
    colors = {'stable_mean':'tab:green','stable_median':'tab:blue','stable_mode':'tab:purple'}
    if plot_mean and est_key!='stable_mean':
        off += 1; ax1.bar(x+off*bw, s['stable_mean'], yerr=s['stable_std'], width=bw, alpha=0.6, label='Mean', color=colors['stable_mean'], capsize=4)
    if plot_median and est_key!='stable_median':
        off += 1; ax1.bar(x+off*bw, s['stable_median'], yerr=s['stable_std'], width=bw, alpha=0.6, label='Median', color=colors['stable_median'], capsize=4)
    if plot_mode and est_key!='stable_mode':
        off += 1; ax1.bar(x+off*bw, s['stable_mode'], yerr=s['stable_std'], width=bw, alpha=0.6, label='Mode', color=colors['stable_mode'], capsize=4)
        if 'mode_estimate' in s: ax1.plot(x, s['mode_estimate'], 'ko', ms=5, label='Mode estimate')

    ax1.set_xticks(x + (bw*off/2 if off else 0)); ax1.set_xticklabels(s['date'], rotation=45)
    ax1.set_xlabel('Date'); ax1.set_ylabel('Weight (g)'); ax1.set_title(f'Bird {bird_name}: reliable weights')

    ax2 = ax1.twinx(); ax2.plot(x, s['stable_count'], 's--', color='tab:red', label='Count'); ax2.set_ylabel('Count')
    if plot_actual_weights and s['true_weight'].notna().any():
        ax1.plot(x, s['true_weight'], 'r*', ms=8, label='True weight')

    lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    if lines or lines2: ax1.legend(lines+lines2, labels+labels2, loc='upper left')
    plt.tight_layout()
    return fig, ax1


# ---------------------- Orchestrator ----------------------
def plot_bird_weight(
    bird_name,
    weight_report_csv,
    weights_dict,
    which='both',                # 'longitudinal' | 'summary' | 'both'
    figsize=None,                # None -> auto: (12,8) for both; (12,4) for single
    # thresholds/reliability/time filters
    low_thrd=1, high_thrd=30, tolerance_fraction=0.3,
    win_size=10, step=10, std_percentile=None, weight_fraction=0.09,
    start_date=None, end_date=None, start_time=None, end_time=None,
    # longitudinal tuning
    raw_downsample=5, raw_downsample_day=None, raw_downsample_night=None,
    rel_downsample=5, apply_ylim=False, add_actual_weights=True,
    # summary tuning
    main_estimator='mean', plot_mode=False, plot_mean=False, plot_median=False,
    plot_actual_weights_summary=True, weights_csv_path="weights.csv",
    save_path=None, keep_out_of_range=False,
):
    which = which.lower()
    figsize = (12, 8) if (figsize is None and which=='both') else (figsize or (12, 4))

    if which == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        plot_longitudinal(
            bird_name, weight_report_csv, weights_dict,
            low_thrd, high_thrd, win_size, step, std_percentile, weight_fraction,
            start_date, end_date, start_time, end_time,
            figsize, raw_downsample, rel_downsample, apply_ylim, add_actual_weights, ax1,
            raw_downsample_day, raw_downsample_night, tolerance_fraction, keep_out_of_range=keep_out_of_range
        )

        summary = compute_reliable_weight_summary(
            bird_name, weight_report_csv, weights_dict,
            low_thrd, high_thrd, win_size, step, std_percentile, weight_fraction,
            start_date, end_date, start_time, end_time, tolerance_fraction, weights_csv_path, 
            keep_out_of_range=keep_out_of_range
        )
        plot_reliable_weight_summary(
            summary, bird_name, main_estimator, plot_mode, plot_mean, plot_median,
            plot_actual_weights_summary, ax2, figsize=(12,4)
        )

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(save_path, bbox_inches='tight', dpi=300)
        return fig, (ax1, ax2), summary

    if which == 'longitudinal':
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_longitudinal(
            bird_name, weight_report_csv, weights_dict,
            low_thrd, high_thrd, win_size, step, std_percentile, weight_fraction,
            start_date, end_date, start_time, end_time,
            figsize, raw_downsample, rel_downsample, apply_ylim, add_actual_weights, ax,
            raw_downsample_day, raw_downsample_night, tolerance_fraction
        )
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(save_path, bbox_inches='tight', dpi=300)
        return fig, ax, None

    if which == 'summary':
        summary = compute_reliable_weight_summary(
            bird_name, weight_report_csv, weights_dict,
            low_thrd, high_thrd, win_size, step, std_percentile, weight_fraction,
            start_date, end_date, start_time, end_time, tolerance_fraction, weights_csv_path, 
            keep_out_of_range=keep_out_of_range
        )
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_reliable_weight_summary(
            summary, bird_name, main_estimator, plot_mode, plot_mean, plot_median,
            plot_actual_weights_summary, ax, figsize
        )
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(save_path, bbox_inches='tight', dpi=300)
        return fig, ax, summary

    raise ValueError("which must be 'longitudinal', 'summary', or 'both'.")


# ---------------------- CLI / Usage ----------------------
if __name__ == "__main__":
    # User config
    low_thrd, high_thrd = 1, 35
    win_size, step = 10, 10
    std_percentile, weight_fraction = None, 0.09
    start_date, end_date, start_time, end_time = '2025-06-11', None, None, None
    raw_downsample, raw_downsample_day, raw_downsample_night = 1, 2, 40
    rel_downsample, tolerance_fraction = 5, 0.3
    weights_json, weights_csv_path = "weights.json", "weights.csv"

    with open(weights_json, 'r') as f:
        weights_dict = json.load(f)
    weights_dict = {k: v for k, v in weights_dict.items() if k != 'start_date'}
    bird_names = list(weights_dict.keys())
    print("Birds:", bird_names)

    bird_name = 'llb84b35'
    weight_report_csv = Path(f"a_{bird_name}_weight_report.csv").absolute()

    # --- Option 1: longitudinal only ---
    plot_bird_weight(
        bird_name, weight_report_csv, weights_dict,
        which='both', figsize=None,  # -> (12,4)
        low_thrd=low_thrd, high_thrd=high_thrd, tolerance_fraction=tolerance_fraction,
        win_size=win_size, step=step, std_percentile=std_percentile, weight_fraction=weight_fraction,
        start_date=start_date, end_date=end_date, start_time=start_time, end_time=end_time,
        raw_downsample=raw_downsample, raw_downsample_day=raw_downsample_day, raw_downsample_night=raw_downsample_night,
        rel_downsample=rel_downsample, apply_ylim=False, add_actual_weights=True, keep_out_of_range=True,
        # save_path=f"figures/summary_{bird_name}.svg"
    )
    plt.show()

    # --- Option 2: joint plot (both) ---
    # plot_bird_weight(
    #     bird_name, weight_report_csv, weights_dict,
    #     which='both', figsize=None,  # -> (12,8)
    #     low_thrd=low_thrd, high_thrd=high_thrd, tolerance_fraction=tolerance_fraction,
    #     win_size=win_size, step=step, std_percentile=std_percentile, weight_fraction=weight_fraction,
    #     start_date=start_date, end_date=end_date, start_time=start_time, end_time=end_time,
    #     raw_downsample=raw_downsample, raw_downsample_day=raw_downsample_day, raw_downsample_night=raw_downsample_night,
    #     rel_downsample=rel_downsample, apply_ylim=False, add_actual_weights=True,
    #     main_estimator='mean', plot_mode=False, plot_mean=False, plot_median=False,
    #     plot_actual_weights_summary=True, weights_csv_path=weights_csv_path,
    #     save_path=f"figures/longitudinal_summary_{bird_name}.svg",
    # )
    # plt.show()
    # --- Option 3: loop all birds and build a full summary table ---
    # summaries = []
    # for b in bird_names:
    #     csv_path = Path(f"a_{b}_weight_report.csv").absolute()
    #     s = compute_reliable_weight_summary(
    #         b, csv_path, weights_dict,
    #         low_thrd=low_thrd, high_thrd=high_thrd, win_size=win_size, step=step,
    #         std_percentile=std_percentile, weight_fraction=weight_fraction,
    #         start_date=start_date, end_date=end_date, start_time=start_time, end_time=end_time,
    #         tolerance_fraction=tolerance_fraction, weights_csv_path=weights_csv_path
    #     )
    #     if s is not None and not s.empty: summaries.append(s)
    # if summaries:
    #     all_summaries = pd.concat(summaries, ignore_index=True)
    #     print(all_summaries)
    #     all_summaries.to_csv('all_birds_summary.csv', index=False)
