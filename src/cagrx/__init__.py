from cagrx.amfi import Amfi
from cagrx.return_metrics import (
    cagr,
    calculate_trailing_cagr,
    calculate_rolling_returns,
    calculate_sip_returns,
    xirr,
)
from cagrx.risk_metrics import (
    calculate_drawdown,
    calculate_max_drawdown,
    max_drawdown,
)
from cagrx.exceptions import (
    CagrxError,
    SchemeNotFoundError,
    MultipleSchemesFoundError,
)

__all__ = [
    "Amfi",
    "cagr",
    "calculate_trailing_cagr",
    "calculate_rolling_returns",
    "calculate_sip_returns",
    "xirr",
    "calculate_drawdown",
    "calculate_max_drawdown",
    "max_drawdown",
    "CagrxError",
    "SchemeNotFoundError",
    "MultipleSchemesFoundError",
]

def main() -> None:
    print("Hello from cagrx!")
