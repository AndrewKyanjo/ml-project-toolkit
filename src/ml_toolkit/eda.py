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
