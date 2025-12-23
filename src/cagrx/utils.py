import time
from datetime import datetime, timedelta

def split_into_date_pairs(start_date_str, end_date_str, n_days=100):
    """
    Splits the range between start_date and end_date into `n_days=n` intervals
    Returns a list of (start, end) date pairs.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    pairs = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=n_days - 1), end_date)
        pairs.append(
            (current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d"))
        )
        current_start = current_end + timedelta(days=1)

    return pairs