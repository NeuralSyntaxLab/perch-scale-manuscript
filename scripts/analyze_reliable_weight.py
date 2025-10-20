import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import datetime
from scipy.stats import mode as scipy_mode

def read_and_filter(
    bird_name,
    data_dir='.',
    low_thrd=2,
    high_thrd=30,
    time_col='Time',
    start_date=None,
    end_date=None
):
    """
    Reads a bird's weight report CSV, filters, sorts, and returns a cleaned dataframe.

    Parameters:
        bird_name: str, the name of the bird
        data_dir: str or Path, directory where CSV is located
        low_thrd: float, lower weight threshold (inclusive)
        high_thrd: float, upper weight threshold (inclusive)
        time_col: str, name of the time column

    Returns:
        pd.DataFrame with filtered and sorted data, columns ['Time', 'weight']
    """
    filename = Path(data_dir) / f"a_{bird_name}_weight_report.csv"
    if not filename.exists():
        raise FileNotFoundError(f"{filename} not found.")

    df = pd.read_csv(filename)
    # Standardize column names
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
    if 'weight' not in df.columns:
        # Rename column 1 to 'weight' (assuming that's where the weights are)
        df.rename(columns={df.columns[1]: 'weight'}, inplace=True)

    # Filter by thresholds
    df = df[(df['weight'] >= low_thrd) & (df['weight'] <= high_thrd)].copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    #Filter by date range if provided
    if start_date is not None:
        df = df[df[time_col] >= pd.to_datetime(start_date)]
    if end_date is not None:    
        df = df[df[time_col] <= pd.to_datetime(end_date)]

    return df[['Time', 'weight']]

# Example usage:
# df_filtered = read_and_filter('bird1')

def split_day_night(df, day_start='07:00', day_end='19:00'):
    """
    Splits a day's dataframe into daytime and nighttime segments.
    Assumes 'Time' is datetime.
    Returns: (daytime_df, nighttime_df)
    """
    # Convert strings to time objects
    t_start = datetime.time.fromisoformat(day_start)
    t_end = datetime.time.fromisoformat(day_end)
    
    times = df['Time'].dt.time
    is_day = times >= t_start
    is_day &= times < t_end
    daytime_df = df[is_day].copy()
    nighttime_df = df[~is_day].copy()
    return daytime_df, nighttime_df

    import datetime


def calc_reliable_measure(df, win_size=10, step=2, std_percentile=None, weight_fraction=0.09, tolerance_fraction=0.3, reference_weight=None):
    """
    Calculate reliable weight measurements using a moving window.
    Uses percentile threshold if std_percentile is provided, else percent of mode weight.

    Parameters:
        df: DataFrame with 'Time' and 'weight' columns.
        win_size: window size in samples (default 100)
        step: number of samples to skip between each window (default 1; e.g., step=10 checks every 10th sample)
        std_percentile: None for mode-based threshold, or int (e.g., 5 for 5th percentile) for the percentile of std threshold
        weight_fraction: float, percentage of mode weight to use as threshold if std_percentile is None (default 0.05 for 5%)
    Returns:
        DataFrame with 'Time' and 'Weight' of reliable measurements.
        Float value of the std threshold used to filter measurements.
    """
    weights = df['weight'].values
    times = df['Time'].values

    means = []
    stds = []
    center_times = []
    for i in range(0, len(weights) - win_size + 1, step):
        win = weights[i:i+win_size]
        win_std = np.std(win)
        win_mean = np.mean(win)
        win_time = times[i + win_size//2]
        means.append(win_mean)
        stds.append(win_std)
        center_times.append(win_time)

    means = np.array(means)
    stds = np.array(stds)
    center_times = np.array(center_times)

    if reference_weight is None:
        # Calculate mode of weights for
        mode_weight = pd.Series(weights).mode()
        if len(mode_weight) == 0:
            reference_weight = np.mean(means)  # fallback, very rare
        else:
            reference_weight = float(mode_weight.mean()) # use mean of modes if multiple modes exist
        if reference_weight < 12:
            print(f"Warning: Default reference weight {reference_weight:.2f} g is unusually low, check data quality or insert manual reference value!")

    # Decide threshold
    if std_percentile is not None:
        std_threshold = np.percentile(stds, std_percentile)
        threshold_info = f"{std_percentile}th percentile of window stds: {std_threshold:.4f} g"
    else:
        # Use reference weight as threshold
        std_threshold = weight_fraction * reference_weight
        # print(f"{weight_fraction*100:.1f}% of reference weight ({reference:.2f}g): {std_threshold:.4f} g")

    reliable_idx = stds < std_threshold
    reliable_means = means[reliable_idx]
    reliable_times = center_times[reliable_idx]

    # lower = (1 - tolerance_fraction) * reference_weight
    # upper = (1 + tolerance_fraction) * reference_weight
    reliable_df = pd.DataFrame({'Time': reliable_times, 'Weight': reliable_means})
    # reliable_df = reliable_df[(reliable_df['Weight'] >= lower) & (reliable_df['Weight'] <= upper)]  # filter out unrealistic weights +/- 30% of reference weight
    
    return reliable_df


def split_to_days(df):
    """
    Split a weight report DataFrame into a dict of daily DataFrames.
    Assumes 'Time' is a datetime64 dtype.
    """
    df = df.copy()
    df['date'] = df['Time'].dt.date
    days = {str(day): group.drop(columns=['date']).reset_index(drop=True)
            for day, group in df.groupby('date')}
    return days


def main(
    bird_name,
    start_date=None,
    end_date=None,
    data_dir='.',
    win_size=100,
    step=1,
    std_percentile=None,
    weight_fraction=0.05,
    daytime_only=False,
    day_start='07:00',
    day_end='19:00'
):
    """
    Process a bird's weight report and calculate reliable measurements.
    If daytime_only=True, use only daytime data (across all days).
    Returns reliable DataFrame, threshold, and a threshold info string.
    """
    df = read_and_filter(bird_name, data_dir=data_dir, start_date=start_date, end_date=end_date)

    if daytime_only:
        df, _ = split_day_night(df, day_start=day_start, day_end=day_end)
        print(f"Daytime only: using {len(df)} records.")
    else:
        print(f"Using all data: {len(df)} records.")

    if len(df) < win_size:
        print("Not enough data for a single window.")
        return pd.DataFrame(columns=['Time', 'Weight']), None, None

    rel_df = calc_reliable_measure(
        df,
        win_size=win_size,
        step=step,
        std_percentile=std_percentile,
        weight_fraction=weight_fraction
    )

    print(f"Found {len(rel_df)} reliable measurements.")

    return rel_df

def plot_weight_ts(
    df,
    reliable_df=None,
    figsize=(15, 6),
    all_color='gray',
    all_alpha=0.4,
    reliable_color='crimson',
    reliable_marker='o',
    reliable_ms=6,
    lw=1,
    title=None
):
    """
    Plot weight timeseries with optional overlay of reliable measurements.

    Parameters:
        df: DataFrame with 'Time' and 'weight'
        reliable_df: DataFrame with 'Time' and 'Weight' (optional)
        figsize: tuple, figure size
        all_color: color for all data
        reliable_color: color for reliable overlay
        reliable_marker: marker for reliable overlay
        reliable_ms: marker size for reliable overlay
        lw: line width for all data
        title: optional plot title
    """
    # Ensure proper dtypes
    df = df[(df['weight'] >= 2) & (df['weight'] <= 30)].copy()
    df = df.sort_values('Time')
    if reliable_df is not None:
        reliable_df = reliable_df[(reliable_df['Weight'] >= 2) & (reliable_df['Weight'] <= 30)].copy()
        reliable_df = reliable_df.sort_values('Time')

    # Detect span in days
    times = pd.to_datetime(df['Time'])
    span_days = (times.max() - times.min()).days + 1

    fig, ax = plt.subplots(figsize=figsize)
    # Plot all data
    # ax.plot(times, df['weight'], color=all_color, alpha=all_alpha, lw=lw, label='All data')
    ax.scatter(times, df['weight'], color=all_color, alpha=all_alpha, s=1, label='All data points', zorder=5)
    # Plot reliable if given
    if reliable_df is not None and not reliable_df.empty:
        ax.scatter(pd.to_datetime(reliable_df['Time']), reliable_df['Weight'],
                   color=reliable_color, s=reliable_ms**2, marker=reliable_marker,
                   label='Reliable', zorder=10)

    # X-axis formatting
    if span_days > 2:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,2)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.setp(ax.get_xticklabels(), rotation=45)

    # Vertical dashed lines for date shifts (midnights)
    date_ticks = pd.date_range(times.dt.normalize().min(), times.dt.normalize().max(), freq='D')
    for d in date_ticks[1:]:
        ax.axvline(pd.Timestamp(d), ls='--', color='k', alpha=0.6, lw=1)

    ax.set_ylabel('Weight (g)')
    ax.set_xlabel('Time')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Weight measurements')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    bird_name = 'lp92llb8'  # Replace with actual bird name

    rel_df_full = main(bird_name, start_date='2025-06-11', win_size=10, step=3, std_percentile=None)
    # rel_df_day = main(bird_name, start_date='2025-06-11', win_size=10, step=3, std_percentile=None, daytime_only=True)

    # group by date and compute mean weight for each day
    rel_df_full['date'] = rel_df_full['Time'].dt.date    
    daily_means = rel_df_full.groupby('date')['Weight'].mean().reset_index().round(2)
    print("Daily mean weights using full data:")
    print(daily_means)
    print("mean of means:", daily_means['Weight'].mean().round(2))

    # #compute daily mean weights for daytime data
    # rel_df_day['date'] = rel_df_day['Time'].dt.date    
    # daily_means = rel_df_day.groupby('date')['Weight'].mean().reset_index().round(2)
    # print("Daily mean weights using daytime data only:")
    # print(daily_means)
    # print("mean of means:", daily_means['Weight'].mean().round(2))


    # for day, rel_df in reliable_day_dict.items():
    #     afternoon_weights = rel_df[(rel_df['Time'].dt.hour >= 12) & (rel_df['Time'].dt.hour < 18)]['Weight']
    #     print(f"{day} afternoon weight data description:")
    #     if afternoon_weights.empty:
    #         print(f"No afternoon measurements")
    #         continue
    #     #print the number of afternoon measurements
    #     print(f"{len(afternoon_weights)} afternoon measurements")
    #     print(f"Mean - {afternoon_weights.mean():.2f} g")
    

    df = read_and_filter(bird_name, data_dir='.', low_thrd=0, start_date='2025-06-11')
  

    
    plot_weight_ts(df, reliable_df=rel_df_full)