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


def hidden_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Detects blank strings or whitespace disguised as valid data in categorical columns."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    blank_summary = {}
    
    for column in categorical_cols:
        blanks = df[column].astype(str).str.strip().eq("").sum()
        if blanks > 0:
            blank_summary[column] = blanks
            
    result = pd.Series(blank_summary, name="hidden_missing_count").to_frame()
    result["percentage"] = (result["hidden_missing_count"] / len(df)) * 100
    return result.sort_values("hidden_missing_count", ascending=False)


def infinite_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Checks all numeric columns for infinite values."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    infinite_counts = pd.Series({
        column: np.isinf(df[column]).sum()
        for column in numeric_cols
    })
    
    result = infinite_counts[infinite_counts > 0].reset_index()
    if result.empty:
        return pd.DataFrame(columns=["feature", "infinite_count"])
        
    result.columns = ["feature", "infinite_count"]
    return result


def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Reports the unique value count for categorical features."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    
    return pd.DataFrame({
        "feature": categorical_cols,
        "n_unique": [df[col].nunique(dropna=False) for col in categorical_cols]
    }).sort_values("n_unique", ascending=False).reset_index(drop=True)


def constant_feature_report(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies constant features and near-constant (highly dominant) features."""
    dominant_pct = {
        col: df[col].value_counts(normalize=True, dropna=False).iloc[0] * 100
        for col in df.columns
    }
    
    report = pd.DataFrame(list(dominant_pct.items()), columns=["feature", "dominant_percentage"])
    report["is_strictly_constant"] = report["dominant_percentage"] == 100.0
    
    return report.sort_values("dominant_percentage", ascending=False).reset_index(drop=True)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Enhances the standard describe() output with median, skew, and unique counts."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        return pd.DataFrame()
        
    summary = df[numeric_cols].describe().T
    summary["median"] = df[numeric_cols].median()
    summary["skew"] = df[numeric_cols].skew()
    summary["n_unique"] = df[numeric_cols].nunique()
    
    return summary.sort_values("skew", ascending=False)
