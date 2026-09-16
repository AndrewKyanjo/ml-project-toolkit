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
