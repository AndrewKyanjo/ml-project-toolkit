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
    
    
def category_normalization_check(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Detects categories that would merge if stripped of whitespace and lowercased."""
    records = []
    for col in columns:
        orig = df[col].nunique(dropna=False)
        norm = df[col].astype(str).str.strip().str.lower().nunique(dropna=False)
        
        if norm < orig:
            records.append({
                "feature": col, 
                "original_unique": orig, 
                "normalized_unique": norm
            })
            
    return pd.DataFrame(records)


# ==========================================
# 3. TARGET RELATIONSHIPS (RISK ANALYSIS)
# ==========================================

def categorical_target_rate(df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    """Calculates the positive rate (e.g., default rate) for each category."""
    summary = df.groupby(feature, dropna=False)[target].agg(
        count="count",
        positives="sum",
        target_rate="mean"
    ).sort_values("target_rate", ascending=False)
    
    summary["target_rate_pct"] = summary["target_rate"] * 100
    return summary.reset_index()


def numeric_target_rate_by_quantile(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> pd.DataFrame:
    """Bins a continuous feature into quantiles and calculates the target rate per bin."""
    temp = df[[feature, target]].dropna().copy()
    temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop")
    
    result = temp.groupby("bin", observed=True)[target].agg(
        count="count",
        positives="sum",
        target_rate="mean"
    )
    result["target_rate_pct"] = result["target_rate"] * 100
    return result.reset_index()



# ==========================================
# 4. CORRELATIONS
# ==========================================

def correlation_matrix(df: pd.DataFrame, columns: list[str], method: str = "pearson") -> pd.DataFrame:
    """Returns the correlation matrix for the specified numerical columns."""
    return df[columns].corr(method=method)


def high_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Extracts feature pairs that exceed the absolute correlation threshold."""
    pairs = []
    cols = corr_matrix.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) >= threshold:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": corr
                })
                
    if not pairs:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
        
    return pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)

