import requests
import csv, os
import pandas as pd
from functools import cache

from cagrx.utils import split_into_date_pairs

SCHEMES_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
NAV_HISTORY_URL = "https://www.amfiindia.com/api/nav-history"

class Amfi:

    def __init__(self):
        self.cache_file = "amfi_navall.csv"
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
        Load schemes list from cache if available, otherwise sync from AMFI
        """
        if os.path.exists(self.cache_file):
            return pd.read_csv(self.cache_file)
        
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
    
    


