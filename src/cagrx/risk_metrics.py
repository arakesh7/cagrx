import pandas as pd


def calculate_drawdown(df, column="nav", round_digits=3):
    """
    Calculate the Maximum Drawdown (MDD) and recovery details for a given NAV series.

    :param df: DataFrame with datetime index
    :param column: Column name for NAV values (default: 'nav')
    :param round_digits: Number of decimal places to round max_drawdown to (default: 3)
    :returns: Dictionary containing:
        - 'max_drawdown': float (e.g. -0.327 for -32.7% drop)
        - 'drawdown_start': str ("YYYY-MM-DD", the peak date before the trough)
        - 'drawdown_end': str ("YYYY-MM-DD", the trough date where drawdown bottomed out)
        - 'recovery_date': str ("YYYY-MM-DD", date when NAV recovered to peak) or None
        - 'recovery_days': int (calendar days from trough to recovery date) or None
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    clean_series = df[column].dropna()
    if len(clean_series) < 2:
        raise ValueError("Insufficient data to calculate drawdown.")

    if (clean_series <= 0).any():
        raise ValueError("NAV values must be strictly positive.")

    df_sorted = df.copy()
    if not isinstance(df_sorted.index, pd.DatetimeIndex):
        df_sorted.index = pd.to_datetime(df_sorted.index)
    df_sorted = df_sorted.sort_index()

    nav_series = df_sorted[column]
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax

    min_dd = drawdown.min()
    if min_dd >= 0:
        return {
            "max_drawdown": 0.0,
            "drawdown_start": None,
            "drawdown_end": None,
            "recovery_date": None,
            "recovery_days": None,
        }

    trough_date = drawdown.idxmin()
    peak_val = cummax.loc[trough_date]

    pre_trough = nav_series.loc[:trough_date]
    peak_date = pre_trough[pre_trough == peak_val].index[-1]

    post_trough = nav_series.loc[df_sorted.index > trough_date]
    recovered = post_trough[post_trough >= peak_val]

    if not recovered.empty:
        recovery_date = recovered.index[0]
        recovery_days = int((recovery_date - trough_date).days)
        recovery_date_str = recovery_date.strftime("%Y-%m-%d")
    else:
        recovery_date_str = None
        recovery_days = None

    max_dd_val = float(min_dd)
    if round_digits is not None:
        max_dd_val = round(max_dd_val, round_digits)

    return {
        "max_drawdown": max_dd_val,
        "drawdown_start": peak_date.strftime("%Y-%m-%d"),
        "drawdown_end": trough_date.strftime("%Y-%m-%d"),
        "recovery_date": recovery_date_str,
        "recovery_days": recovery_days,
    }


# Aliases
calculate_max_drawdown = calculate_drawdown
max_drawdown = calculate_drawdown
