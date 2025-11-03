# utils/formatting.py
import pandas as pd

def records_to_df(records) -> pd.DataFrame:
    return pd.DataFrame(records) if records else pd.DataFrame()

def to_date(series: pd.Series):
    return pd.to_datetime(series, errors="coerce").dt.date

def to_num(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")
