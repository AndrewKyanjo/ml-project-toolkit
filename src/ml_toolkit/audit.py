import pandas as pd
import numpy as np


def target_report(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Returns the class distribution and percentages of the target variable."""
    summary = (
        df[target_col]
        .value_counts(dropna=False)
        .rename_axis("class")
        .reset_index(name="count")
    )
    summary["percentage"] = (summary["count"] / len(df)) * 100
    
    return summary


def id_report(df: pd.DataFrame, id_col: str) -> dict:
    """Audits the unique identifier column for missing or duplicate values."""
    return {
        "total_rows": len(df),
        "unique_ids": int(df[id_col].nunique()),
        "missing_ids": int(df[id_col].isna().sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum())
    }


def duplicate_report(df: pd.DataFrame) -> int:
    """Returns the total number of exact duplicate rows in the dataset."""
    return int(df.duplicated().sum())


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of columns that contain missing values, 
    including their counts and percentages.
    """
    # Note: I wrote the actual missingness logic here, as your 
    # snippet accidentally duplicated the row logic!
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].reset_index()
    
    if missing_counts.empty:
        return pd.DataFrame(columns=["feature", "missing_count", "percentage"])
        
    missing_counts.columns = ["feature", "missing_count"]
    missing_counts["percentage"] = (missing_counts["missing_count"] / len(df)) * 100
    
    return missing_counts.sort_values("missing_count", ascending=False).reset_index(drop=True)


def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a master audit table summarizing column types, missingness, and uniqueness."""
    audit_table = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
        "n_unique": df.nunique(dropna=False),
    })
    audit_table["unique_pct"] = (audit_table["n_unique"] / len(df)) * 100
    return audit_table.sort_values("missing_pct", ascending=False)

