# 📈 cagrx

![Status](https://img.shields.io/badge/status-work--in--progress-orange) ![Python](https://img.shields.io/badge/python-3.9%2B-blue)

> **No Cap, only CAGR – Calculate Mutual fund Returns Like a Pro**

*Download, analyze, and unlock insights from Indian mutual funds with ease.*

---
 
> [!WARNING]
> **Work in Progress**: This library is under active development. APIs may change, and some features are still being refined. Contributions and feedback are welcome!

**CagrX** is a powerful Python library to analyze mutual funds using AMFI (Association of Mutual Funds in India) data. Whether you're building a fintech app, conducting investment research, or analyzing portfolio performance, cagrx provides the tools you need to work with mutual fund data effortlessly.

## ✨ Features

- **Download Mutual Fund Data**: Fetch NAV (Net Asset Value) data for any Indian mutual fund scheme
- **Historical Data**: Access historical NAV data with automatic chunking for large date ranges
- **Fund Discovery**: Browse and search through all available mutual fund schemes
- **Performance Analysis**: Built-in metrics for calculating returns:
  - CAGR (Compound Annual Growth Rate)
  - Trailing returns (1Y, 3Y, 5Y, etc.)
  - Rolling returns with max/min/average statistics
  - SIP (Systematic Investment Plan) return calculations
  - Maximum Drawdown (MDD) and recovery tracking
- **Data Persistence**: Automatic caching of schemes list for faster subsequent access

## 🚀 Installation

**Install via pip:**

```bash
pip install cagrx
```

**Or install from source:**

```bash
git clone https://github.com/yourusername/cagrx.git
cd cagrx
pip install -e .
```

## 📦 Requirements

- Python >= 3.9
- pandas
- requests

## 🔧 Usage

### Getting Started with AMFI Data

```python
from cagrx.amfi import Amfi

# Initialize AMFI client
amfi = Amfi()
```

### Browse Available Mutual Funds

```python
# Get all fund houses
fund_houses = amfi.get_fund_houses()
print(fund_houses)

# Get all schemes from a specific fund house
schemes = amfi.get_schemes_by_fund_house("HDFC Mutual Fund")
print(schemes)

# Get complete schemes list (cached locally as amfi_navall.csv)
all_schemes = amfi.list_all_schemes()

# Force refresh the schemes list from AMFI and update cache
# Useful when you want to ensure you have the new funds included
refreshed_schemes = amfi.refresh_schemes()
```

### 🔍 Search & Scheme Resolution

Search across all AMFI funds by keyword, code, plan, or option:

```python
# Search schemes by keywords
matches = amfi.search_schemes("Parag Parikh Flexi", limit=5)

# Resolve scheme code from fund name
scheme_code = amfi.resolve_scheme("Parag Parikh Flexi Cap", plan="direct", option="growth")
# Returns: "122639"
```

### ⚡ Scheme-Aware Analytics

Calculate metrics directly using a scheme code or scheme name without manually downloading NAV DataFrames:

```python
# Trailing CAGR by fund name and plan/option
trailing = amfi.trailing_cagr("Parag Parikh Flexi Cap", plan="direct", option="growth")
print(trailing)

# Rolling returns (pass years=3 or period=pd.DateOffset(years=3))
rolling = amfi.rolling_returns("122639", years=3)
print(rolling)

# Maximum Drawdown and recovery timeline
mdd = amfi.drawdown("122639")
print(mdd)

# SIP returns calculation
sip = amfi.sip_returns("122639", monthly_amount=5000, start_date="2020-01-01")
print(f"Invested: ₹{sip['total_invested']} -> Current: ₹{sip['current_value']}")
```

### Download Historical NAV Data

```python
# Download NAV data for a specific scheme
# Scheme code can be found from list_all_schemes()
nav_data = amfi.get_nav_history(
    scheme_id="122639",  # Example: HDFC Flexi Cap Fund
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Returns a pandas DataFrame with date as index and NAV values
print(nav_data.head())
```

### Calculate Performance Metrics

#### CAGR (Compound Annual Growth Rate)

> [!NOTE]
> You can use `-1` as a period to calculate the CAGR for the entire available historical data (Max CAGR).

```python
from cagrx.return_metrics import cagr, calculate_trailing_cagr

# Calculate overall CAGR
overall_cagr = cagr(nav_data)
print(f"Overall CAGR: {overall_cagr * 100:.2f}%")

# Calculate trailing CAGR for multiple periods
trailing_returns = calculate_trailing_cagr(
    nav_data,
    periods=[-1, 1, 3, 5]  # -1 for Max CAGR, 1Y, 3Y, 5Y
)
print(trailing_returns)
# Output:
# {
#     'Max_CAGR': 0.165,
#     'Max_CAGR_start_date': '2013-05-24',
#     'Max_CAGR_end_date': '2024-10-06',
#     'Max_CAGR_years': 11.37,
#     '1Y_CAGR': 0.123,
#     '3Y_CAGR': 0.156,
#     '5Y_CAGR': 0.142
# }
```

#### Rolling Returns

```python
from cagrx.return_metrics import calculate_rolling_returns
import pandas as pd

# Calculate 1-year rolling returns
rolling_metrics = calculate_rolling_returns(
    nav_data,
    period=pd.DateOffset(years=1)
)

print(f"Max rolling return: {rolling_metrics['max_returns'] * 100:.2f}%")
print(f"Period: {rolling_metrics['max_return_period']}")
print(f"Min rolling return: {rolling_metrics['min_returns'] * 100:.2f}%")
print(f"Average rolling return: {rolling_metrics['avg_return'] * 100:.2f}%")
```

#### SIP Returns (What-If Analysis)

Analyze how your periodic investments would have performed:

```python
from cagrx.amfi import Amfi
from cagrx.return_metrics import calculate_sip_returns
import pandas as pd

# First, get the NAV data for the fund
amfi = Amfi()
nav_data = amfi.get_nav_history(
    scheme_id="122639",
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Example 1: Regular monthly SIP of ₹5,000 for 3 years
sip_dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
sip_cashflows = pd.DataFrame({'amount': 5000}, index=sip_dates)

returns = calculate_sip_returns(sip_cashflows, nav_data)
print(f"Invested: ₹{returns['total_invested']:,.0f} → Current: ₹{returns['current_value']:,.0f}")
print(f"Returns: {returns['return_percentage']:.2f}% (Annualized: {returns['annualized_return']:.2f}%)")

# Example 2: Irregular investments (lump sum + step-up SIP)
irregular_investments = pd.DataFrame({
    'amount': [50000, 5000, 7500, 10000, 15000]
}, index=pd.to_datetime([
    '2020-01-15',   # Initial lump sum
    '2020-06-01',   # ₹5k after 6 months
    '2021-01-01',   # Stepped up to ₹7.5k
    '2021-06-01',   # Stepped up to ₹10k
    '2022-01-01'    # Stepped up to ₹15k
]))

irregular_returns = calculate_sip_returns(irregular_investments, nav_data)
print(f"Irregular Returns: {irregular_returns['return_percentage']:.2f}%")
```

#### Maximum Drawdown and Recovery

Analyze peak-to-trough drops and recovery timelines:

```python
from cagrx.risk_metrics import calculate_drawdown

drawdown = calculate_drawdown(nav_data)
print(drawdown)
# Output:
# {
#     "max_drawdown": -0.327,
#     "drawdown_start": "2021-01-15",
#     "drawdown_end": "2021-06-20",
#     "recovery_date": "2022-02-10",
#     "recovery_days": 235
# }
```

### Complete Example

```python
from cagrx.amfi import Amfi
from cagrx.return_metrics import cagr, calculate_trailing_cagr, calculate_rolling_returns
import pandas as pd

# Initialize
amfi = Amfi()

# Browse HDFC schemes
hdfc_schemes = amfi.get_schemes_by_fund_house("HDFC Mutual Fund")
print("Available HDFC Schemes:")
print(hdfc_schemes.head())

# Download NAV data for HDFC Flexi Cap Fund
nav_data = amfi.get_nav_history(
    scheme_id="122639",
    start_date="2018-01-01",
    end_date="2024-12-31"
)

# Save to CSV for future use
nav_data.to_csv("hdfc_flexi_cap_nav.csv")

# Calculate performance metrics
print("\n=== Performance Metrics ===")

# Overall CAGR
overall_return = cagr(nav_data)
print(f"\nOverall CAGR: {overall_return * 100:.2f}%")

# Trailing returns
trailing = calculate_trailing_cagr(nav_data, periods=[1, 3, 5])
print("\nTrailing Returns:")
for period, value in trailing.items():
    if value:
        print(f"  {period}: {value * 100:.2f}%")
    else:
        print(f"  {period}: Insufficient data")

# Rolling returns
rolling = calculate_rolling_returns(nav_data, period=pd.DateOffset(years=1))
print(f"\n1-Year Rolling Returns:")
print(f"  Max: {rolling['max_returns'] * 100:.2f}%")
print(f"  Min: {rolling['min_returns'] * 100:.2f}%")
print(f"  Avg: {rolling['avg_return'] * 100:.2f}%")
```

## 📊 Data Sources

All data is sourced from official AMFI (Association of Mutual Funds in India) APIs:
- **Schemes List**: https://www.amfiindia.com/spages/NAVAll.txt
- **Historical NAV**: https://www.amfiindia.com/api/nav-history

## 🏗️ Project Structure

```
cagrx/
├── src/
│   └── cagrx/
│       ├── __init__.py          # Main package entry point
│       ├── amfi.py              # AMFI data fetching and management
│       ├── return_metrics.py    # Performance calculation utilities
│       ├── risk_metrics.py      # Downside risk and drawdown calculations
│       └── utils.py             # Helper functions
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## 🧪 Development

### Running Tests

```bash
# Run the test file
python src/cagrx/amfi_tests.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data provided by AMFI (Association of Mutual Funds in India)
- Built with Python, pandas, and requests

## 📮 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This library is for educational and research purposes. Please verify all calculations independently before making investment decisions.