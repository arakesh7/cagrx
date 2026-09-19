import pandas as pd


def cagr(df, column="nav"):
    """
    Calculate the Compound Annual Growth Rate (CAGR) for the given column over the DataFrame's period.
    
    :param df: DataFrame with datetime index
    :param column: Column name to calculate CAGR on (default: 'nav')
    :returns CAGR as a float (e.g., 0.12 for 12% annualized growth)

    note: This method doesn't sort the index of the DataFrame. It is expected that the index is sorted in ascending order.
    """   
    start_value = df[column].iloc[0]
    end_value = df[column].iloc[-1]

    num_years = (df.index[-1] - df.index[0]).days / 365

    if start_value <= 0 or num_years <= 0:
        raise ValueError("Invalid data for CAGR calculation.")
    
    cagr = (end_value / start_value) ** (1 / num_years) - 1
    return round(cagr, 3)


def calculate_trailing_cagr(df, column="nav", periods=None):
    """
    Calculate the Compound Annual Growth Rate (CAGR) of a fund over the given periods.
    
    :param df: DataFrame with datetime index
    :param column: Column name to calculate CAGR on (default: 'nav')
    :returns CAGR as a float (e.g., 0.12 for 12% annualized growth)

    note: -1 period is used to calculate CAGR of MAX period (considering all the data available)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if periods is None:
        periods = [1, 3, 5] 
    
    df = df.sort_index()
    cagr_metrics = {}

    for period in periods:
        if period == -1:
            cagr_metrics[f'Max_CAGR'] = cagr(df, column=column)
            continue
        
        start_date = df.index[-1] - pd.DateOffset(years=period)
        
        # Not enough historical data for this period
        if start_date < df.index[0]:
            cagr_metrics[f'{period}Y_CAGR'] = None
            continue
        
        #calculate cagr for the `n` period
        cagr_metrics[f'{period}Y_CAGR'] = cagr(df.loc[start_date:], column=column)
    
    return cagr_metrics
    
def calculate_rolling_returns(df, column="nav", period=pd.DateOffset(years=1)):
    """
    Calculate the rolling annualised CAGR for the given column over a sliding window.

    Each data point's return is annualised using the CAGR formula so that results
    from windows of different lengths (1Y, 3Y, 5Y) are directly comparable.

      annualised_return = (nav_end / nav_start) ^ (1 / years) - 1

    :param df: DataFrame with datetime index named "date"
    :param column: Column name to calculate rolling returns on (default: 'nav')
    :param period: Window length as a pd.DateOffset (default: 1 year)
    :returns: dict with keys max_returns, max_return_period, min_returns,
              min_return_period, avg_return -- all as annualised decimals
              (e.g. 0.18 means 18% p.a.)
    """
    # Derive the window length in years from the DateOffset so we can annualise.
    # DateOffset(years=n) stores n in the .n attribute; fall back to 1 for
    # offsets that don't expose it (e.g. MonthEnd).
    # DateOffset stores the year count in kwds['years'] when constructed as
    # DateOffset(years=n). The .n attribute is always 1 for this form.
    # Fall back to 1 so that non-year offsets (e.g. MonthEnd) still work.
    years = period.kwds.get('years', 1) if hasattr(period, 'kwds') else 1

    df = df.copy()[[column]]
    dates_in_df = df.index.date
    df['past_date'] = dates_in_df - period

    rolling_df = pd.merge_asof(df, df[['nav']], left_on='past_date', right_on='date', suffixes=('_current', '_past'))
    rolling_df['past_date'] = rolling_df['past_date'].dt.date
    rolling_df.index = dates_in_df

    # Drop rows where historical data for past dates are not available
    rolling_df = rolling_df.dropna(subset=['nav_past'])

    # Annualise using CAGR formula: (end/start)^(1/years) - 1
    rolling_df['returns'] = (
        (rolling_df['nav_current'] / rolling_df['nav_past']) ** (1 / years) - 1
    ).round(4)

    max_row = rolling_df.loc[rolling_df['returns'].idxmax()]
    min_row = rolling_df.loc[rolling_df['returns'].idxmin()]
    metrics = {
        'max_returns': float(max_row['returns']),
        'max_return_period': (str(max_row['past_date']), str(max_row.name)),
        'min_returns': float(min_row['returns']),
        'min_return_period': (str(min_row['past_date']), str(min_row.name)),
        'avg_return': float(rolling_df['returns'].mean().round(4))
    }

    return metrics

def xirr(cashflows, dates, guess=0.1, max_iterations=100, tolerance=1e-6):
    """
    Calculate XIRR (Extended Internal Rate of Return) for irregular cash flows.
    
    Uses the Newton-Raphson method to find the rate that makes NPV = 0.
    
    :param cashflows: List or array of cash flows (negative for investments, positive for returns)
    :param dates: List or array of dates corresponding to each cash flow (datetime objects)
    :param guess: Initial guess for the rate (default: 0.1 or 10%)
    :param max_iterations: Maximum number of iterations (default: 100)
    :param tolerance: Convergence tolerance (default: 1e-6)
    :returns: XIRR as a decimal (e.g., 0.15 for 15% annual return)
    
    Example:
        cashflows = [-5000, -5000, -5000, 17500]
        dates = pd.to_datetime(['2020-01-01', '2020-07-01', '2021-01-01', '2021-12-31'])
        rate = xirr(cashflows, dates)
        print(f"XIRR: {rate * 100:.2f}%")
    """
    if len(cashflows) != len(dates):
        raise ValueError("cashflows and dates must have the same length")
    
    if len(cashflows) < 2:
        raise ValueError("At least 2 cash flows are required")
    
    # Convert dates to pandas datetime if not already
    dates = pd.to_datetime(dates)
    
    # Sort by dates
    sorted_indices = dates.argsort()
    cashflows = [cashflows[i] for i in sorted_indices]
    dates = dates[sorted_indices]
    
    # Calculate days from first date
    start_date = dates[0]
    days = [(date - start_date).days for date in dates]
    
    # Newton-Raphson method
    rate = guess
    
    for iteration in range(max_iterations):
        # Calculate NPV (Net Present Value)
        npv = sum(cf / ((1 + rate) ** (day / 365.0)) for cf, day in zip(cashflows, days))
        
        # Calculate derivative of NPV
        dnpv = sum(-cf * day / 365.0 / ((1 + rate) ** (day / 365.0 + 1)) for cf, day in zip(cashflows, days))
        
        # Check for convergence
        if abs(npv) < tolerance:
            return round(rate, 6)
        
        # Avoid division by zero
        if abs(dnpv) < 1e-10:
            raise ValueError("XIRR calculation failed: derivative too small")
        
        # Update rate using Newton-Raphson formula
        rate = rate - npv / dnpv
        
        # Check if rate is becoming unreasonable
        if rate < -0.99 or rate > 10:  # -99% to 1000% range
            raise ValueError("XIRR calculation failed: rate out of reasonable bounds")
    
    raise ValueError(f"XIRR did not converge after {max_iterations} iterations")

def calculate_sip_returns(sip_cashflows, nav_df, column="nav"):
    """
    Calculate total returns using SIP (Systematic Investment Plan) cashflow.
    
    :param sip_cashflows: DataFrame with datetime index and 'amount' column representing investment amounts
    :param nav_df: DataFrame with datetime index and NAV column
    :param column: Column name for NAV values (default: 'nav')
    :returns: Dictionary with total invested, current value, absolute returns, and return percentage
    
    Example:
        sip_cashflows = pd.DataFrame({
            'amount': [5000, 5000, 5000]
        }, index=pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']))
        
        nav_df = pd.DataFrame({
            'nav': [100, 105, 110, 115]
        }, index=pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01']))
        
        returns = calculate_sip_returns(sip_cashflows, nav_df)
    """
    if 'amount' not in sip_cashflows.columns:
        raise ValueError("sip_cashflows must have 'amount' column")
    if column not in nav_df.columns:
        raise ValueError(f"Column '{column}' not found in nav_df")
    
    # Sort both dataframes by index
    sip_cashflows = sip_cashflows.sort_index()
    nav_df = nav_df.sort_index()
    
    # Merge SIP cashflows with NAV data to get NAV at each investment date
    # direction is set to `forward` because we want the NAV that's available on the 
    # same day or after the investment date
    # eg: If investment is made on a holiday, we want the NAV that's available on the next trading day (not the previous trading day)
    merged = pd.merge_asof(
        sip_cashflows,
        nav_df[[column]],
        left_index=True,
        right_index=True,
        direction='forward' 
    )
    
    # Calculate units purchased at each SIP date
    merged['units'] = merged['amount'] / merged[column]
    
    # Calculate total invested and total units
    total_invested = merged['amount'].sum()
    total_units = merged['units'].sum()
    
    # Get current NAV (last available NAV)
    current_nav = nav_df[column].iloc[-1]
    
    # Calculate current value
    current_value = total_units * current_nav
    
    # Calculate absolute returns
    absolute_returns = current_value - total_invested
    return_percentage = (absolute_returns / total_invested) * 100 if total_invested > 0 else 0
    
    # Calculate annualized returns using XIRR.
    # XIRR is the correct measure for SIPs because each instalment has been
    # invested for a different duration; simple CAGR on the total corpus
    # does not account for this and materially understates the true return.
    #
    # Cash-flow convention: investments are negative (money leaving wallet),
    # the final portfolio value is positive (money coming back).
    investment_cashflows = [-amount for amount in merged['amount']]
    investment_dates     = list(merged.index)

    cashflows_for_xirr = investment_cashflows + [current_value]
    dates_for_xirr     = investment_dates     + [nav_df.index[-1]]

    try:
        xirr_rate = xirr(cashflows_for_xirr, dates_for_xirr)
        annualized_return = round(xirr_rate * 100, 2)
    except (ValueError, ZeroDivisionError):
        annualized_return = None

    return {
        'total_invested': round(total_invested, 2),
        'current_value': round(current_value, 2),
        'absolute_returns': round(absolute_returns, 2),
        'return_percentage': round(return_percentage, 2),
        'annualized_return (xirr %)': annualized_return,
        'total_units': round(total_units, 4),
        'current_nav': round(current_nav, 2),
        'investment_period_days': (nav_df.index[-1] - sip_cashflows.index[0]).days if len(sip_cashflows) > 0 else 0
    }