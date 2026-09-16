import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. TARGET & BASIC DISTRIBUTIONS
# ==========================================


def target_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Returns the absolute and percentage distribution of the target variable."""
    counts = df[target].value_counts(dropna=False).sort_index()
    pcts = df[target].value_counts(normalize=True, dropna=False).sort_index() * 100

    return pd.DataFrame({"count": counts, "percentage": pcts})


def zero_percentage(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculates the percentage of exact zero values in numeric features."""
    zeros = df[columns].eq(0).mean() * 100
    return (
        zeros[zeros > 0]
        .sort_values(ascending=False)
        .to_frame("zero_pct")
        .reset_index(names="feature")
    )


# ==========================================
# 2. OUTLIERS & ANOMALIES
# ==========================================


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculates IQR bounds and returns the count/percentage of outliers per feature."""
    records = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        mask = (df[col] < lower) | (df[col] > upper)

        records.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": mask.sum(),
                "outlier_pct": mask.mean() * 100,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("outlier_pct", ascending=False)
        .reset_index(drop=True)
    )
