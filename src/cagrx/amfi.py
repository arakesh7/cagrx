import requests
import csv, os
import pandas as pd
from datetime import datetime, timedelta
from functools import cache

from cagrx.utils import split_into_date_pairs
from cagrx.exceptions import SchemeNotFoundError, MultipleSchemesFoundError
from cagrx.return_metrics import (
    cagr as cagr_fn,
    calculate_trailing_cagr,
    calculate_rolling_returns,
    calculate_sip_returns,
)
from cagrx.risk_metrics import (
    calculate_drawdown,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_consistency_metrics,
)

SCHEMES_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
NAV_HISTORY_URL = "https://www.amfiindia.com/api/nav-history"

class Amfi:

    def __init__(self):
        self.cache_file = os.path.expanduser("~/.cagrx/amfi_navall.csv")
        self.schemes_list = self._load_schemes()
        
    
    def list_all_schemes(self):
        """
        Get all schemes from the AMFI list
        
        :returns: pandas dataframe containing all schemes
        """
        return self.schemes_list

    def get_fund_houses(self):
        """
        Get all the available fund houses from the AMFI list
        
        :returns: set of fund house names
        """
        return set(self.schemes_list["fund_house"].dropna().unique())
        
    def refresh_schemes(self) -> pd.DataFrame:
        """
        Force refresh schemes list from AMFI and update cache.
        """
        self.schemes_list = self._get_schemes_from_amfi()
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.schemes_list.to_csv(self.cache_file, index=False)
        return self.schemes_list

    def get_schemes_by_fund_house(self, fund_house):
        """
        Get available schemes for a given fund house
        
        :param fund_house: name of the fund house
        :returns: pandas dataframe containing scheme codes and names
        """
        return self.schemes_list[self.schemes_list["fund_house"] == fund_house][['scheme_code', 'scheme_name']]

    def get_nav_history(self, scheme_id, start_date, end_date):
        """ 
        Download NAV data for a given mutual fund scheme within the date range
        
        This method fetches the data in chunks of 5 years

        :param start_date: start_date of the requested data period
        :param end_date: end_date of the requested data period
        :param scheme_id: scheme_id of the mutual fund for which the NAV should be fetched
        :param freq: frequency of the data
        
        :returns: pandas dataframe containing nav data
        """
        
        # AMFI allows maximum of five_years to be downloaded at a time
        date_ranges = split_into_date_pairs(start_date, end_date, n_days=365 * 5) 
        nav_records = []

        for from_date, to_date in date_ranges:
            records = self._fetch_historical_nav(scheme_id, from_date, to_date)
            nav_records.extend(records)

        return self._create_dataframe(nav_records)

    get_historical_nav = get_nav_history

    def _match_schemes(
        self,
        query: str,
        plan: str | None = None,
        option: str | None = None,
        fund_house: str | None = None,
    ) -> pd.DataFrame:
        """
        Internal helper to match and filter schemes by query string, code, plan, option, and fund house.

        :param query: Keyword string or scheme code to search for
        :param plan: Optional plan filter (e.g. "direct" or "regular")
        :param option: Optional option filter (e.g. "growth" or "idcw")
        :param fund_house: Optional fund house filter
        :returns: Filtered DataFrame of matching schemes
        """
        query_str = str(query).strip()
        df = self.schemes_list

        # 1. Exact scheme code match
        code_match = df[df["scheme_code"].astype(str) == query_str]
        if not code_match.empty:
            return code_match

        # 2. Token-based search across scheme_name
        tokens = query_str.lower().split()
        if not tokens:
            return df.iloc[0:0]

        mask = pd.Series(True, index=df.index)
        names_lower = df["scheme_name"].astype(str).str.lower()
        for token in tokens:
            mask = mask & names_lower.str.contains(token, regex=False, na=False)

        matches = df[mask]

        # 3. Apply optional filters
        if fund_house:
            matches = matches[
                matches["fund_house"].astype(str).str.lower().str.contains(fund_house.lower(), regex=False, na=False)
            ]

        if plan:
            matches = matches[
                matches["plan"].astype(str).str.lower().str.contains(plan.lower(), regex=False, na=False)
            ]

        if option:
            matches = matches[
                matches["option"].astype(str).str.lower().str.contains(option.lower(), regex=False, na=False)
            ]

        return matches

    def search_schemes(
        self,
        query: str,
        limit: int | None = 10,
        plan: str | None = None,
        option: str | None = None,
        fund_house: str | None = None,
    ) -> list[dict]:
        """
        Search for mutual fund schemes matching a query string or scheme code.

        :param query: Keyword string or scheme code to search for
        :param limit: Maximum number of matches to return (default: 10, None for all)
        :param plan: Optional plan filter (e.g. "direct" or "regular")
        :param option: Optional option filter (e.g. "growth" or "idcw")
        :param fund_house: Optional fund house filter
        :returns: List of scheme dictionaries
        """
        if not query or not str(query).strip():
            return []

        matches = self._match_schemes(query, plan=plan, option=option, fund_house=fund_house)

        if limit is not None and limit > 0:
            matches = matches.head(limit)

        matches = matches.where(pd.notnull(matches), None)
        return matches.to_dict(orient="records")

    def resolve_scheme(
        self,
        scheme: str | int,
        plan: str | None = None,
        option: str | None = None,
    ) -> str:
        """
        Resolve a scheme code or scheme name to an exact AMFI scheme code string.

        :param scheme: AMFI scheme code (e.g. "119551") or scheme name query (e.g. "Parag Parikh Flexi Cap")
        :param plan: Optional plan filter (e.g. "direct", "regular")
        :param option: Optional option filter (e.g. "growth", "idcw")
        :returns: Exact scheme_code as a string (e.g. "119551")
        :raises SchemeNotFoundError: If no scheme matches the input
        :raises MultipleSchemesFoundError: If multiple schemes match and cannot be disambiguated
        """
        scheme_str = str(scheme).strip()

        # 1. Check exact scheme code match in AMFI database
        code_match = self.schemes_list[self.schemes_list["scheme_code"].astype(str) == scheme_str]
        if not code_match.empty:
            return scheme_str

        # If scheme was all numeric digits but not found in the table:
        if scheme_str.isdigit():
            raise SchemeNotFoundError(
                f"Scheme code '{scheme_str}' was not found in the AMFI scheme database. "
                "Call amfi.refresh_schemes() to update the local cache."
            )

        # 2. Check base matches without plan/option to detect existence
        base_matches = self._match_schemes(scheme_str)
        if base_matches.empty:
            raise SchemeNotFoundError(
                f"No schemes found matching '{scheme_str}'. "
                "Try a broader keyword or use amfi.search_schemes() to explore available funds."
            )

        # 3. Check for exact full name match (case-insensitive)
        names_lower = base_matches["scheme_name"].astype(str).str.lower()
        exact_name_match = base_matches[names_lower == scheme_str.lower()]
        if len(exact_name_match) == 1:
            return str(exact_name_match.iloc[0]["scheme_code"])

        # 4. Filter with plan and option
        filtered_matches = self._match_schemes(scheme_str, plan=plan, option=option)

        if filtered_matches.empty:
            raise SchemeNotFoundError(
                f"Found schemes matching '{scheme_str}', but none matched plan='{plan}' and option='{option}'."
            )

        if len(filtered_matches) == 1:
            return str(filtered_matches.iloc[0]["scheme_code"])

        # Multiple matches remain
        match_records = filtered_matches.to_dict(orient="records")
        raise MultipleSchemesFoundError(query=scheme_str, matches=match_records)

    def get_scheme_info(
        self,
        scheme: str | int,
        plan: str | None = None,
        option: str | None = None,
    ) -> dict:
        """
        Get metadata for a single scheme by code or name.
        """
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        row = self.schemes_list[self.schemes_list["scheme_code"].astype(str) == str(scheme_code)]
        if not row.empty:
            row = row.where(pd.notnull(row), None)
            return row.iloc[0].to_dict()
        raise SchemeNotFoundError(f"Scheme code '{scheme_code}' not found.")

    def cagr(
        self,
        scheme: str | int,
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        column: str = "nav",
    ) -> float:
        """
        Calculate CAGR for a scheme by code or name.
        """
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        start_date = start_date or "1990-01-01"
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return cagr_fn(nav_df, column=column)

    def trailing_cagr(
        self,
        scheme: str | int,
        periods: list[int] | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        column: str = "nav",
    ) -> dict:
        """
        Calculate trailing CAGR (e.g. 1Y, 3Y, 5Y) for a scheme by code or name.
        """
        if periods is None:
            periods = [1, 3, 5]
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        positive_periods = [p for p in periods if p > 0]
        max_period = max(positive_periods) if positive_periods else 5
        if -1 in periods:
            start_date = "1990-01-01"
        else:
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int((max_period + 0.5) * 365.25))
            start_date = start_dt.strftime("%Y-%m-%d")
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return calculate_trailing_cagr(nav_df, column=column, periods=periods)

    def rolling_returns(
        self,
        scheme: str | int,
        period: pd.DateOffset | None = None,
        years: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        hurdle_rate: float = 0.08,
        column: str = "nav",
    ) -> dict:
        """
        Calculate rolling returns for a scheme by code or name.
        Accepts either period=pd.DateOffset(years=3) or years=3.
        """
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        if years is not None:
            offset_period = pd.DateOffset(years=years)
        elif period is None:
            offset_period = pd.DateOffset(years=1)
        elif isinstance(period, int):
            offset_period = pd.DateOffset(years=period)
        else:
            offset_period = period

        n_years = offset_period.kwds.get('years', 1) if hasattr(offset_period, 'kwds') else 1
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        if start_date is None:
            fetch_years = max(n_years + 3, 5)
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=int((fetch_years + 0.5) * 365.25))
            start_date = start_dt.strftime("%Y-%m-%d")
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return calculate_rolling_returns(nav_df, column=column, period=offset_period, hurdle_rate=hurdle_rate)

    def drawdown(
        self,
        scheme: str | int,
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        round_digits: int = 3,
        column: str = "nav",
    ) -> dict:
        """
        Calculate maximum drawdown and recovery for a scheme by code or name.
        """
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        start_date = start_date or "2000-01-01"
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return calculate_drawdown(nav_df, column=column, round_digits=round_digits)

    max_drawdown = drawdown

    def sip_returns(
        self,
        scheme: str | int,
        monthly_amount: float,
        start_date: str,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        column: str = "nav",
    ) -> dict:
        """
        Calculate SIP returns for a scheme by code or name.
        """
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        sip_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        sip_cashflows = pd.DataFrame({'amount': monthly_amount}, index=sip_dates)
        return calculate_sip_returns(sip_cashflows, nav_df, column=column)

    calculate_sip = sip_returns

    def volatility(
        self,
        scheme: str | int,
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        trading_days: int = 252,
        round_digits: int = 4,
        column: str = "nav",
    ) -> float | None:
        """Calculate annualized volatility for a scheme."""
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        start_date = start_date or "1990-01-01"
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return calculate_volatility(nav_df, column=column, trading_days=trading_days, round_digits=round_digits)

    def sharpe_ratio(
        self,
        scheme: str | int,
        risk_free_rate: float = 0.06,
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        trading_days: int = 252,
        round_digits: int = 3,
        column: str = "nav",
    ) -> float | None:
        """Calculate the annualized Sharpe Ratio for a scheme."""
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        start_date = start_date or "1990-01-01"
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return calculate_sharpe_ratio(nav_df, risk_free_rate=risk_free_rate, column=column, trading_days=trading_days, round_digits=round_digits)

    def sortino_ratio(
        self,
        scheme: str | int,
        risk_free_rate: float = 0.06,
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        trading_days: int = 252,
        round_digits: int = 3,
        column: str = "nav",
    ) -> float | None:
        """Calculate the annualized Sortino Ratio for a scheme."""
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        start_date = start_date or "1990-01-01"
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return calculate_sortino_ratio(nav_df, risk_free_rate=risk_free_rate, column=column, trading_days=trading_days, round_digits=round_digits)

    def calmar_ratio(
        self,
        scheme: str | int,
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        round_digits: int = 3,
        column: str = "nav",
    ) -> float | None:
        """Calculate the Calmar Ratio for a scheme."""
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        start_date = start_date or "1990-01-01"
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return calculate_calmar_ratio(nav_df, column=column, round_digits=round_digits)

    def consistency_metrics(
        self,
        scheme: str | int,
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        resample_rule: str = "ME",
        drop_incomplete: bool = True,
        tolerance_days: int = 4,
        column: str = "nav",
    ) -> dict:
        """Calculate consistency metrics (best/worst periods, hit rate) for a scheme."""
        scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        start_date = start_date or "1990-01-01"
        nav_df = self.get_nav_history(scheme_code, start_date, end_date)
        if nav_df.empty:
            raise ValueError(f"No NAV data found for scheme {scheme_code} between {start_date} and {end_date}")
        return calculate_consistency_metrics(
            nav_df, 
            column=column, 
            resample_rule=resample_rule, 
            drop_incomplete=drop_incomplete, 
            tolerance_days=tolerance_days
        )

    # ------------------------------------------------------------------
    # Metric dispatch helper (used by compare_funds)
    # ------------------------------------------------------------------

    _DEFAULT_METRICS = [
        "cagr", "volatility", "sharpe_ratio",
        "sortino_ratio", "calmar_ratio", "max_drawdown",
    ]

    def _compute_metric(
        self,
        metric: str,
        nav_df: pd.DataFrame,
        *,
        risk_free_rate: float,
        trading_days: int,
        round_digits: int,
        column: str,
    ) -> float | None:
        """
        Compute a single named metric on a NAV DataFrame.

        :param metric: One of the supported metric names (see _DEFAULT_METRICS).
        :param nav_df: DataFrame with a DatetimeIndex and a NAV column.
        :returns: The computed value, or None on failure.
        """
        if metric == "cagr":
            return round(cagr_fn(nav_df, column=column), round_digits)
        if metric == "volatility":
            return calculate_volatility(
                nav_df, column=column,
                trading_days=trading_days, round_digits=round_digits,
            )
        if metric == "sharpe_ratio":
            return calculate_sharpe_ratio(
                nav_df, risk_free_rate=risk_free_rate, column=column,
                trading_days=trading_days, round_digits=round_digits,
            )
        if metric == "sortino_ratio":
            return calculate_sortino_ratio(
                nav_df, risk_free_rate=risk_free_rate, column=column,
                trading_days=trading_days, round_digits=round_digits,
            )
        if metric == "calmar_ratio":
            return calculate_calmar_ratio(
                nav_df, column=column, round_digits=round_digits,
            )
        if metric == "max_drawdown":
            dd = calculate_drawdown(nav_df, column=column, round_digits=round_digits)
            return dd["max_drawdown"] if dd else None
        if metric == "p10_drawdown":
            dd = calculate_drawdown(nav_df, column=column, round_digits=round_digits)
            return dd["p10_drawdown"] if dd else None
        if metric == "p90_drawdown":
            dd = calculate_drawdown(nav_df, column=column, round_digits=round_digits)
            return dd["p90_drawdown"] if dd else None
        return None

    # ------------------------------------------------------------------
    # Fund comparison
    # ------------------------------------------------------------------

    def compare_funds(
        self,
        schemes: list[str | int],
        start_date: str | None = None,
        end_date: str | None = None,
        plan: str | None = None,
        option: str | None = None,
        metrics: list[str] | None = None,
        risk_free_rate: float = 0.06,
        trading_days: int = 252,
        round_digits: int = 3,
        column: str = "nav",
    ) -> pd.DataFrame:
        """
        Compare multiple funds across various return and risk metrics.

        :param schemes: List of scheme codes or names to compare.
        :param start_date: Start date for the comparison period (YYYY-MM-DD).
        :param end_date: End date for the comparison period (YYYY-MM-DD).
        :param plan: Optional plan filter passed to resolve_scheme (e.g. "direct", "regular").
        :param option: Optional option filter passed to resolve_scheme (e.g. "growth", "idcw").
        :param metrics: List of metric names to compute. Defaults to _DEFAULT_METRICS.
        :param risk_free_rate: Risk-free rate for Sharpe and Sortino ratios.
        :param trading_days: Number of trading days in a year.
        :param round_digits: Number of decimal places to round the results to.
        :param column: The column containing NAV data.
        :returns: A pandas DataFrame with one row per scheme and one column per metric,
                  plus a 'scheme_code' column.  Indexed by scheme name.
        """
        if metrics is None:
            metrics = list(self._DEFAULT_METRICS)

        end_date_str = end_date or datetime.today().strftime("%Y-%m-%d")
        start_date_str = start_date or "1990-01-01"

        results = []
        for scheme in schemes:
            # --- resolve scheme code & name ---
            try:
                scheme_code = self.resolve_scheme(scheme, plan=plan, option=option)
                scheme_info = self.get_scheme_info(scheme_code)
                scheme_name = scheme_info["scheme_name"] if scheme_info else str(scheme)
            except Exception:
                scheme_name = str(scheme)
                scheme_code = None

            row: dict = {"scheme_code": scheme_code, "Scheme Name": scheme_name}

            # If resolution failed, fill metrics with None
            if scheme_code is None:
                row.update({m: None for m in metrics})
                results.append(row)
                continue

            # --- fetch NAV history ---
            try:
                nav_df = self.get_nav_history(scheme_code, start_date_str, end_date_str)
                if nav_df.empty:
                    raise ValueError("No NAV data")
            except Exception:
                row.update({m: None for m in metrics})
                results.append(row)
                continue

            # --- compute each requested metric ---
            for m in metrics:
                try:
                    row[m] = self._compute_metric(
                        m, nav_df,
                        risk_free_rate=risk_free_rate,
                        trading_days=trading_days,
                        round_digits=round_digits,
                        column=column,
                    )
                except Exception:
                    row[m] = None
            results.append(row)

        df = pd.DataFrame(results)
        df.set_index("Scheme Name", inplace=True)
        return df

    _SCHEME_COLUMNS = ["scheme_code", "isin_growth", "isin_reinv", "scheme_name", "plan", "option", "nav", "date", "fund_house"]

    def _get_schemes_from_amfi(self):
        """
        Fetch and parse all schemes from the AMFI NAVAll.txt feed.

        Handles both the current 8-field format and the legacy 6-field format:
          - Current : scheme_code; isin_growth; isin_reinv; scheme_name; plan; option; nav; date
          - Legacy  : scheme_code; isin_growth; isin_reinv; scheme_name; nav; date

        :returns: DataFrame with columns defined in _SCHEME_COLUMNS
        """
        raw_lines = self._fetch_raw_nav_lines()
        current_fund_house = None
        records = []

        for line in raw_lines[1:]:  # row 0 is the header
            line = line.strip()
            if not line:
                continue

            if line.endswith("Mutual Fund"):
                current_fund_house = line
                continue

            row = [field.strip() for field in line.split(";")]
            record = self._parse_scheme_row(row, current_fund_house)
            if record:
                records.append(record)

        return pd.DataFrame(records, columns=self._SCHEME_COLUMNS)

    def _parse_scheme_row(self, row: list, fund_house: str) -> list | None:
        """
        Parse a single semicolon-split AMFI row into a normalised record.

        :param row: list of stripped fields from one line of NAVAll.txt
        :param fund_house: the fund house name currently in scope
        :returns: a list aligned to _SCHEME_COLUMNS, or None if the row is invalid
        """
        MIN_FIELDS = 5
        if len(row) < MIN_FIELDS:
            return None

        scheme_code, isin_growth, isin_reinv = row[0], row[1], row[2]

        if len(row) == 8:
            # Current AMFI format (8 fields)
            base_name, plan, option, nav, date = row[3], row[4], row[5], row[6], row[7]
            scheme_name = self._build_scheme_name(base_name, plan, option)
        else:
            # Legacy AMFI format (6 fields) — no plan/option columns
            base_name, plan, option = row[3], "", ""
            nav, date = row[-2], row[-1]
            scheme_name = base_name

        return [scheme_code, isin_growth, isin_reinv, scheme_name, plan, option, nav, date, fund_house]

    @staticmethod
    def _build_scheme_name(base_name: str, plan: str, option: str) -> str:
        """
        Build a human-readable scheme name from its components.

        Filters out empty strings and bare hyphens before joining so that
        e.g. ("Edelweiss Mid Cap Fund", "Direct Plan", "Growth") becomes
        "Edelweiss Mid Cap Fund - Direct Plan - Growth".

        :returns: combined scheme name string
        """
        parts = [p for p in [base_name, plan, option] if p and p != "-"]
        return " - ".join(parts) if parts else base_name

    @cache
    def _fetch_historical_nav(self, scheme_id, from_date, to_date):
        """ 
        Actual method implementing the network/API call to the AMFI URL

        :param scheme_id: scheme_id of the mutual fund for which the NAV should be fetched
        :param from_date: start_date of the requested data period
        :param to_date: end_date of the requested data period
        """
        
        query_params = {
            "query_type": "historical_period",
            "sd_id": scheme_id,
            "from_date": from_date,
            "to_date": to_date,
        }
        resp = requests.get(
            NAV_HISTORY_URL, params=query_params
        )

        if resp.status_code == 200:
            raw_json = resp.json()
            if "data" in raw_json:
                return raw_json["data"]["nav_groups"][0]["historical_records"]
            return [] #if no data found for the given date range
        else:
            raise ValueError(resp.text)

    def _load_schemes(self):
        """
        Load schemes list from cache if available and valid, otherwise sync from AMFI
        """
        if os.path.exists(self.cache_file):
            try:
                df = pd.read_csv(self.cache_file)
                if not df.empty and "scheme_code" in df.columns:
                    return df
            except Exception:
                pass
        
        return self.refresh_schemes()
        
    def _fetch_raw_nav_lines(self):
        response = requests.get(SCHEMES_URL)
        response.raise_for_status()
        
        return response.text.strip().splitlines()

    def _create_dataframe(self, records):
        """creates pandas dataframe from the raw records"""
        
        df = pd.DataFrame(records)
        df['nav'] = pd.to_numeric(df['nav'])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")
    
    


