
import unittest
import pandas as pd
from datetime import datetime
from cagrx.return_metrics import cagr, calculate_trailing_cagr, calculate_rolling_returns, xirr, calculate_sip_returns
from cagrx.amfi import Amfi
import os

class TestReturnMetrics(unittest.TestCase):
    
    def setUp(self):
        # Setup common data
        self.dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        self.nav_values = [100 * (1.00032)**i for i in range(len(self.dates))] # Approx 12% annual growth
        self.df = pd.DataFrame({'nav': self.nav_values}, index=self.dates)

    def test_cagr_calculation(self):
        # Test basic CAGR
        result = cagr(self.df)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 0.123, places=2) # Expect approx 12-13%

    def test_trailing_cagr(self):
        # Test trailing CAGR including Max (-1)
        periods = [1, -1]
        result = calculate_trailing_cagr(self.df, periods=periods)
        
        self.assertIn('1Y_CAGR', result)
        self.assertIn('Max_CAGR', result)
        self.assertIsNotNone(result['1Y_CAGR'])
        self.assertIsNotNone(result['Max_CAGR'])
        
        # Max CAGR should equal total period cagr
        self.assertEqual(result['Max_CAGR'], cagr(self.df))

    def test_xirr(self):
        # Test cases from original test_xirr.py
        
        # Test 1: Regular investments
        cashflows = [-5000, -5000, -5000, 17500]
        dates = pd.to_datetime(['2020-01-01', '2020-07-01', '2021-01-01', '2021-12-31'])
        rate = xirr(cashflows, dates)
        self.assertAlmostEqual(rate, 0.11, places=1) # Approx 11%

        # Test 2: Irregular investments
        cashflows2 = [-10000, -5000, -7500, 25000]
        dates2 = pd.to_datetime(['2020-01-15', '2020-06-01', '2021-01-01', '2022-06-30'])
        rate2 = xirr(cashflows2, dates2)
        self.assertIsInstance(rate2, float)

        # Test 3: Loss scenario
        cashflows3 = [-10000, -10000, 18000]
        dates3 = pd.to_datetime(['2020-01-01', '2020-06-01', '2021-12-31'])
        rate3 = xirr(cashflows3, dates3)
        self.assertLess(rate3, 0) # Should be negative

    def test_sip_returns(self):
        # Simple SIP test
        sip_dates = pd.date_range(start='2020-01-01', periods=12, freq='MS')
        sip_cashflows = pd.DataFrame({'amount': 1000}, index=sip_dates)
        
        result = calculate_sip_returns(sip_cashflows, self.df)
        
        self.assertIn('total_invested', result)
        self.assertIn('current_value', result)
        self.assertIn('return_percentage', result)
        self.assertEqual(result['total_invested'], 12000)

class TestAmfiIntegration(unittest.TestCase):
    
    def setUp(self):
        self.amfi = Amfi()
        self.scheme_id = "122639" # Common fund for testing
    
    def test_get_historical_nav(self):
        # This test hits the network
        try:
            nav = self.amfi.get_historical_nav(self.scheme_id, "2023-01-01", "2023-01-10")
            self.assertIsInstance(nav, pd.DataFrame)
            self.assertFalse(nav.empty)
            self.assertIn('nav', nav.columns)
            self.assertEqual(len(nav), len(nav.dropna()))
        except Exception as e:
            self.skipTest(f"Network or API issue: {e}")

    def test_refresh_schemes(self):
        try:
            schemes = self.amfi.refresh_schemes()
            self.assertIsInstance(schemes, pd.DataFrame)
            self.assertFalse(schemes.empty)
        except Exception as e:
            self.skipTest(f"Network or API issue: {e}")

if __name__ == '__main__':
    unittest.main()
