from cagrx.amfi import Amfi
from cagrx.return_metrics import (
    cagr,
    calculate_trailing_cagr,
    calculate_rolling_returns,
    calculate_sip_returns,
    xirr
)

__all__ = [
    "Amfi",
    "cagr",
    "calculate_trailing_cagr",
    "calculate_rolling_returns",
    "calculate_sip_returns",
    "xirr",
]

def main() -> None:
    print("Hello from cagrx!")
