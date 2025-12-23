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

    def _get_schemes_from_amfi(self):
        """
        Get all schemes from the AMFI list
        
        :returns: pandas dataframe containing all schemes
        """
        # Fetch data
        raw_lines = self._fetch_raw_nav_lines()
        
        current_fund_house = None
        lines = []
        
        for line in raw_lines[1:]: #skip first line containing headers
            line = line.strip()

            if line.endswith("Mutual Fund"):
                current_fund_house = line #if the current line is fund house name, store it and skip the iteration
                continue

            row = line.split(";")
            if len(row) < 5:
                continue #skip invalid or header lines
            
            row.append(current_fund_house)
            lines.append(row)

        columns = ["scheme_code", "isin_growth", "isin_reinv", "scheme_name", "nav", "date", "fund_house"]

        return pd.DataFrame(lines, columns=columns)

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
    
    


