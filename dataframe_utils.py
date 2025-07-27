import pandas as pd
from pandas import DatetimeTZDtype
from pandas.api.types import is_datetime64_any_dtype
from typing import Dict, List
import streamlit as st

@st.cache_data
def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up placeholders, formats datetimes as strings, and trims whitespace.
    """
    # Trim whitespace from all object columns
    df = df.replace(["-", "N/A", "NULL", "", None], pd.NA)
    for col in df.columns:
        dtype = df[col].dtype
        if is_datetime64_any_dtype(dtype) or isinstance(dtype, DatetimeTZDtype):
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.convert_dtypes()

@st.cache_data
def get_simple_diff(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame listing all row-level and cell-level changes based on index.
    """
    rows = []
    old_idx = set(old_df.index)
    new_idx = set(new_df.index)

    # Added rows
    for idx in sorted(new_idx - old_idx):
        rows.append({
            "action": "added", "row_index": str(idx), "column": None,
            "old_value": None, "new_value": new_df.loc[idx].to_dict()
        })

    # Removed rows
    for idx in sorted(old_idx - new_idx):
        rows.append({
            "action": "removed", "row_index": str(idx), "column": None,
            "old_value": old_df.loc[idx].to_dict(), "new_value": None
        })

    # Changed cells
    for idx in sorted(old_idx & new_idx):
        for col in old_df.columns:
            a, b = old_df.at[idx, col], new_df.at[idx, col]
            if not ((pd.isna(a) and pd.isna(b)) or a == b):
                rows.append({
                    "action": "changed", "row_index": str(idx), "column": col,
                    "old_value": a, "new_value": b
                })

    return pd.DataFrame(rows)


@st.cache_data
def get_keyed_diff(old_df: pd.DataFrame, new_df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Compares two DataFrames based on key columns, identifying added, removed, and changed rows.
    """
    if not all(col in old_df.columns and col in new_df.columns for col in key_cols):
        raise ValueError("Key columns must exist in both DataFrames.")

    # Set index to align rows by content, not position
    old_df = old_df.set_index(key_cols, drop=False)
    new_df = new_df.set_index(key_cols, drop=False)
    
    rows = []
    old_idx = set(old_df.index)
    new_idx = set(new_df.index)
    
    # Added rows (keys present in new_df but not in old_df)
    for key in sorted(new_idx - old_idx):
        rows.append({
            "action": "added", "row_index": str(key), "column": None,
            "old_value": None, "new_value": new_df.loc[key].to_dict()
        })
        
    # Removed rows (keys present in old_df but not in new_df)
    for key in sorted(old_idx - new_idx):
        rows.append({
            "action": "removed", "row_index": str(key), "column": None,
            "old_value": old_df.loc[key].to_dict(), "new_value": None
        })
        
    # Changed cells (keys present in both)
    data_cols = [col for col in old_df.columns if col not in key_cols]
    for key in sorted(old_idx & new_idx):
        # Use .iloc[0] because the key might not be unique if key_cols isn't a true primary key
        old_row = old_df.loc[key].iloc[0]
        new_row = new_df.loc[key].iloc[0]
        for col in data_cols:
            a, b = old_row.get(col), new_row.get(col)
            if not ((pd.isna(a) and pd.isna(b)) or a == b):
                rows.append({
                    "action": "changed", "row_index": str(key), "column": col,
                    "old_value": a, "new_value": b
                })
                
    return pd.DataFrame(rows)

@st.cache_data
def summarize_diff(diff_df: pd.DataFrame) -> Dict[str, int]:
    """
    Takes a diff DataFrame and returns a dictionary summarizing the changes.
    """
    if diff_df.empty:
        return {"added": 0, "removed": 0, "changed": 0}
        
    counts = diff_df['action'].value_counts().to_dict()
    return {
        "added": counts.get("added", 0),
        "removed": counts.get("removed", 0),
        "changed": counts.get("changed", 0)
    }
    
def get_schema_diff(cols_before: list, rename_map_before: dict, cols_after: list, rename_map_after: dict) -> dict:
    """
    Compares two schema versions and returns a dictionary of added and removed columns.
    """
    # Determine the final set of column names for each version
    final_names_before = {rename_map_before.get(col, col) for col in cols_before}
    final_names_after = {rename_map_after.get(col, col) for col in cols_after}
    
    # Calculate added and removed columns using set operations
    added = sorted(list(final_names_after - final_names_before))
    removed = sorted(list(final_names_before - final_names_after))
    
    return {'added': added, 'removed': removed}
