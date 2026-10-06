
import unittest
from unittest.mock import patch
import pandas as pd
from datetime import datetime
from cagrx.return_metrics import (
    cagr,
    calculate_trailing_cagr,
    calculate_rolling_returns,
    xirr,
    calculate_sip_returns,
)
from cagrx.risk_metrics import (
    calculate_drawdown,
    calculate_max_drawdown,
    max_drawdown,
)
from cagrx.amfi import Amfi
from cagrx.exceptions import SchemeNotFoundError, MultipleSchemesFoundError
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
        self.assertIn('Max_CAGR_start_date', result)
        self.assertIn('Max_CAGR_end_date', result)
        self.assertIn('Max_CAGR_years', result)
        self.assertIsNotNone(result['1Y_CAGR'])
        self.assertIsNotNone(result['Max_CAGR'])
        self.assertEqual(result['Max_CAGR_start_date'], '2020-01-01')
        self.assertEqual(result['Max_CAGR_end_date'], '2023-01-01')
        self.assertEqual(result['Max_CAGR_years'], 3.0)
        
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


class TestRiskMetrics(unittest.TestCase):

    def test_drawdown_calculation(self):
        # Test scenario matching user specification
        dates = pd.to_datetime(['2021-01-15', '2021-06-20', '2022-02-10'])
        navs = [100.0, 67.3, 100.0]
        dd_df = pd.DataFrame({'nav': navs}, index=dates)

        result = calculate_drawdown(dd_df)
        expected = {
            "max_drawdown": -0.327,
            "drawdown_start": "2021-01-15",
            "drawdown_end": "2021-06-20",
            "recovery_date": "2022-02-10",
            "recovery_days": 235,
        }
        self.assertEqual(result, expected)
        # Check aliases
        self.assertEqual(calculate_max_drawdown(dd_df), expected)
        self.assertEqual(max_drawdown(dd_df), expected)

    def test_drawdown_unrecovered(self):
        dates = pd.to_datetime(['2021-01-15', '2021-06-20', '2021-09-01'])
        navs = [100.0, 67.3, 85.0]
        dd_df = pd.DataFrame({'nav': navs}, index=dates)

        result = calculate_drawdown(dd_df)
        self.assertEqual(result["max_drawdown"], -0.327)
        self.assertEqual(result["drawdown_start"], "2021-01-15")
        self.assertEqual(result["drawdown_end"], "2021-06-20")
        self.assertIsNone(result["recovery_date"])
        self.assertIsNone(result["recovery_days"])

    def test_drawdown_strictly_increasing(self):
        dates = pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
        navs = [100.0, 101.0, 102.0]
        dd_df = pd.DataFrame({'nav': navs}, index=dates)

        result = calculate_drawdown(dd_df)
        self.assertEqual(result["max_drawdown"], 0.0)
        self.assertIsNone(result["drawdown_start"])
        self.assertIsNone(result["drawdown_end"])
        self.assertIsNone(result["recovery_date"])
        self.assertIsNone(result["recovery_days"])

    def test_backward_compatibility_import(self):
        from cagrx.return_metrics import calculate_drawdown as dd_from_return
        dates = pd.to_datetime(['2021-01-15', '2021-06-20', '2022-02-10'])
        dd_df = pd.DataFrame({'nav': [100.0, 67.3, 100.0]}, index=dates)
        self.assertEqual(dd_from_return(dd_df)["max_drawdown"], -0.327)

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


class TestSchemeAwareAmfi(unittest.TestCase):

    def setUp(self):
        self.amfi = Amfi()
        # Mock NAV DataFrame for calculations
        self.dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        self.nav_values = [100.0 * (1.00032)**i for i in range(len(self.dates))]
        self.mock_nav_df = pd.DataFrame({'nav': self.nav_values}, index=self.dates)
        self.mock_nav_df.index.name = 'date'

    def test_search_schemes(self):
        results = self.amfi.search_schemes('Parag Parikh Flexi', limit=5)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertTrue(any('122639' in str(r['scheme_code']) for r in results))

    def test_resolve_scheme_by_code(self):
        code_str = self.amfi.resolve_scheme('122639')
        self.assertEqual(code_str, '122639')

        code_int = self.amfi.resolve_scheme(122639)
        self.assertEqual(code_int, '122639')

    def test_resolve_scheme_not_found_code(self):
        with self.assertRaises(SchemeNotFoundError):
            self.amfi.resolve_scheme('999999999')

    def test_resolve_scheme_not_found_name(self):
        with self.assertRaises(SchemeNotFoundError):
            self.amfi.resolve_scheme('NonExistentFundSuperXYZ123')

    def test_resolve_scheme_with_plan_and_option(self):
        code = self.amfi.resolve_scheme('Parag Parikh Flexi Cap', plan='direct', option='growth')
        self.assertEqual(code, '122639')

    def test_resolve_scheme_multiple_matches_error(self):
        with self.assertRaises(MultipleSchemesFoundError) as ctx:
            self.amfi.resolve_scheme('Parag Parikh Flexi Cap')
        self.assertGreater(len(ctx.exception.matches), 1)

    @patch.object(Amfi, 'get_nav_history')
    def test_scheme_aware_trailing_cagr(self, mock_get_nav):
        mock_get_nav.return_value = self.mock_nav_df
        result = self.amfi.trailing_cagr('122639', periods=[1, 3])
        self.assertIn('1Y_CAGR', result)
        self.assertIn('3Y_CAGR', result)

    @patch.object(Amfi, 'get_nav_history')
    def test_scheme_aware_rolling_returns(self, mock_get_nav):
        mock_get_nav.return_value = self.mock_nav_df
        result = self.amfi.rolling_returns('Parag Parikh Flexi Cap', years=1, plan='direct', option='growth')
        self.assertIn('max_returns', result)
        self.assertIn('min_returns', result)
        self.assertIn('avg_return', result)

    @patch.object(Amfi, 'get_nav_history')
    def test_scheme_aware_drawdown(self, mock_get_nav):
        mock_get_nav.return_value = self.mock_nav_df
        result = self.amfi.drawdown('122639')
        self.assertIn('max_drawdown', result)
        self.assertIn('drawdown_start', result)
        self.assertIn('drawdown_end', result)

    @patch.object(Amfi, 'get_nav_history')
    def test_scheme_aware_sip_returns(self, mock_get_nav):
        mock_get_nav.return_value = self.mock_nav_df
        result = self.amfi.sip_returns('122639', monthly_amount=5000, start_date='2020-01-01', end_date='2021-01-01')
        self.assertIn('total_invested', result)
        self.assertIn('current_value', result)


if __name__ == '__main__':
    unittest.main()
