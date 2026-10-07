import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare_series(df, column="nav"):
    """
    Single place where the NAV series is cleaned:
    validates the column, drops NaNs, enforces a sorted DatetimeIndex,
    and removes duplicate dates (keeps the last one).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    s = df[column].dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s[~s.index.duplicated(keep="last")]


def _prepare_returns(df, column="nav"):
    """Simple returns. NaNs are dropped BEFORE pct_change so no fake 0% days appear."""
    return _prepare_series(df, column).pct_change().dropna()


def _daily_rf(risk_free_rate, trading_days):
    """Convert an annualized risk-free rate to a daily rate (compounded)."""
    return (1 + risk_free_rate) ** (1 / trading_days) - 1


def _round_or_none(value, round_digits):
    """Return None for undefined values (NaN/inf), otherwise a rounded float."""
    if value is None or not np.isfinite(value):
        return None
    value = float(value)
    return round(value, round_digits) if round_digits is not None else value


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def calculate_drawdown(df, column="nav", top_n=3, round_digits=3):
    """
    Maximum drawdown and recovery details.

    :returns: dict with
        - 'max_drawdown': float (e.g. -0.327 for -32.7%)
        - 'drawdown_start': "YYYY-MM-DD" (peak date before the trough)
        - 'drawdown_end': "YYYY-MM-DD" (trough date)
        - 'recovery_date': "YYYY-MM-DD" or None
        - 'recovery_days': int (calendar days trough -> recovery) or None
        - 'current_drawdown': float (drawdown from latest peak to current day)
        - 'ulcer_index': float (measure of depth and duration of drawdowns)
        - 'top_drawdowns': list of dicts (up to top_n distinct drawdown episodes)
    """
    nav = _prepare_series(df, column)

    if len(nav) < 2:
        raise ValueError("Insufficient data to calculate drawdown.")
    if (nav <= 0).any():
        raise ValueError("NAV values must be strictly positive.")

    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax

    current_dd = _round_or_none(drawdown.iloc[-1], round_digits)
    ulcer_index = (_round_or_none(np.sqrt(((drawdown * 100) ** 2).mean()), round_digits) / 100)

    df_dd = pd.DataFrame({'nav': nav, 'cummax': cummax, 'drawdown': drawdown})
    episodes = []
    
    for _, group in df_dd.groupby('cummax'):
        depth = group['drawdown'].min()
        if depth < 0:
            trough_date = group['drawdown'].idxmin()
            pre_trough = group.loc[:trough_date]
            peaks = pre_trough[pre_trough['drawdown'] == 0]
            peak_date = peaks.index[-1] if not peaks.empty else group.index[0]
            
            post_trough = group.loc[group.index > trough_date]
            recoveries = post_trough[post_trough['drawdown'] == 0]
            
            if not recoveries.empty:
                recovery_date = recoveries.index[0]
            else:
                group_end = group.index[-1]
                future_data = nav.loc[nav.index > group_end]
                if not future_data.empty:
                    recovery_date = future_data.index[0]
                else:
                    recovery_date = None
                    
            if recovery_date is not None:
                recovery_days = int((recovery_date - trough_date).days)
                recovery_date_str = recovery_date.strftime("%Y-%m-%d")
            else:
                recovery_date_str = None
                recovery_days = None
                
            episodes.append({
                'max_drawdown': _round_or_none(depth, round_digits),
                'drawdown_start': peak_date.strftime("%Y-%m-%d"),
                'drawdown_end': trough_date.strftime("%Y-%m-%d"),
                'recovery_date': recovery_date_str,
                'recovery_days': recovery_days
            })

    episodes.sort(key=lambda x: x['max_drawdown'])
    top_episodes = episodes[:top_n]

    if not episodes:
        return {
            "max_drawdown": 0.0,
            "drawdown_start": None,
            "drawdown_end": None,
            "recovery_date": None,
            "recovery_days": None,
            "current_drawdown": current_dd,
            "ulcer_index": ulcer_index,
            "top_drawdowns": []
        }

    worst = top_episodes[0]
    return {
        "max_drawdown": worst["max_drawdown"],
        "drawdown_start": worst["drawdown_start"],
        "drawdown_end": worst["drawdown_end"],
        "recovery_date": worst["recovery_date"],
        "recovery_days": worst["recovery_days"],
        "current_drawdown": current_dd,
        "ulcer_index": ulcer_index,
        "top_drawdowns": top_episodes
    }


# Aliases
calculate_max_drawdown = calculate_drawdown
max_drawdown = calculate_drawdown


# ---------------------------------------------------------------------------
# Volatility, Sharpe, Sortino
# ---------------------------------------------------------------------------

def calculate_volatility(df, column="nav", trading_days=252, round_digits=4):
    """Annualized volatility (sample std of returns * sqrt(trading_days))."""
    returns = _prepare_returns(df, column)
    if len(returns) < 2:
        return None
    return _round_or_none(returns.std(ddof=1) * np.sqrt(trading_days), round_digits)


def calculate_sharpe_ratio(df, risk_free_rate=0.06, column="nav",
                           trading_days=252, round_digits=3):
    """Annualized Sharpe: mean(excess) / std(excess) * sqrt(trading_days)."""
    returns = _prepare_returns(df, column)
    if len(returns) < 2:
        return None

    excess = returns - _daily_rf(risk_free_rate, trading_days)
    std_excess = excess.std(ddof=1)
    if pd.isna(std_excess) or std_excess == 0:
        return None

    return _round_or_none(excess.mean() / std_excess * np.sqrt(trading_days), round_digits)


def calculate_sortino_ratio(df, risk_free_rate=0.06, column="nav",
                            trading_days=252, round_digits=3):
    """
    Annualized Sortino: mean(excess) / downside deviation * sqrt(trading_days).
    Target (MAR) is the daily risk-free rate. Downside deviation averages the
    squared shortfalls over ALL observations (non-negative days count as 0).
    """
    returns = _prepare_returns(df, column)
    if len(returns) < 2:
        return None

    excess = returns - _daily_rf(risk_free_rate, trading_days)
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt((downside ** 2).mean())

    if pd.isna(downside_dev) or downside_dev == 0:
        return None  # no downside observed -> ratio undefined

    return _round_or_none(excess.mean() / downside_dev * np.sqrt(trading_days), round_digits)


# ---------------------------------------------------------------------------
# Calmar
# ---------------------------------------------------------------------------

def calculate_calmar_ratio(df, column="nav", round_digits=3):
    """Calmar = CAGR / |max drawdown|."""
    navs = _prepare_series(df, column)
    if len(navs) < 2:
        return None

    days = (navs.index[-1] - navs.index[0]).days
    if days <= 0:
        return None

    cagr = (navs.iloc[-1] / navs.iloc[0]) ** (365.25 / days) - 1

    dd = calculate_drawdown(df, column=column, round_digits=None)
    max_dd = abs(dd["max_drawdown"])
    if max_dd == 0:
        return None  # no drawdown -> ratio undefined

    return _round_or_none(cagr / max_dd, round_digits)


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------

def _resample_last(navs, rule):
    """Resample to period-end values; falls back to 'M' on pandas < 2.2."""
    try:
        return navs.resample(rule).last().dropna()
    except ValueError:
        legacy = {"ME": "M", "QE": "Q", "YE": "A"}
        if rule in legacy:
            return navs.resample(legacy[rule]).last().dropna()
        raise


def calculate_consistency_metrics(df, column="nav", resample_rule="ME",
                                  drop_incomplete=True, tolerance_days=4):
    """
    Best/worst period and hit rate over complete periods.

    - The first period is never counted: the series starts mid-period, so its
      return would be partial (pct_change naturally starts at the first period end).
    - The last period is dropped if the data ends before the period does
      (e.g. data ending Oct 7 -> October is excluded). `tolerance_days` allows
      for weekends/holidays at the period end.
    """
    navs = _prepare_series(df, column)
    if len(navs) < 2:
        return {}

    period_navs = _resample_last(navs, resample_rule)

    if drop_incomplete and len(period_navs) > 1:
        period_label = period_navs.index[-1]  # period end, e.g. 2026-10-31
        if navs.index[-1] < period_label - pd.Timedelta(days=tolerance_days):
            period_navs = period_navs.iloc[:-1]

    period_returns = period_navs.pct_change().dropna()
    if period_returns.empty:
        return {}

    positive = int((period_returns > 0).sum())
    total = len(period_returns)

    return {
        "best_period_date": period_returns.idxmax().strftime("%Y-%m-%d"),
        "best_period_return": round(float(period_returns.max()), 4),
        "worst_period_date": period_returns.idxmin().strftime("%Y-%m-%d"),
        "worst_period_return": round(float(period_returns.min()), 4),
        "hit_rate": round(positive / total, 4),
        "total_periods": total,
        "positive_periods": positive,
    }