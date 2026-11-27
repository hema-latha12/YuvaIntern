"""Assignment 4: Industry best practices.

Demonstrates cleaning, reproducibility, train/test splitting without data
leakage, and basic validation checks. Run:  python 04_best_practices.py
"""

import numpy as np
import pandas as pd


# Task 1: Reproducibility ---------------------------------------------------
# Always seed randomness and use a local generator rather than global state.
def make_dataset(seed=42, n=200):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),  # unique key, so only injected copies are duplicates
            "age": rng.integers(18, 70, n).astype(float),
            "income": rng.normal(50_000, 12_000, n).round(2),
        }
    )
    df["bought"] = (df["income"] + rng.normal(0, 8_000, n) > 52_000).astype(int)
    # Inject realistic mess: missing values and a duplicated row.
    df.loc[rng.choice(n, 10, replace=False), "income"] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


# Task 2: Cleaning -----------------------------------------------------------
def clean(df):
    out = df.drop_duplicates().copy()
    return out


# Task 3: Validation checks (fail loudly, early) ----------------------------
def validate(df):
    assert df["age"].between(0, 120).all(), "age out of range"
    assert set(df["bought"].unique()) <= {0, 1}, "target must be binary"
    assert not df.duplicated().any(), "duplicates remain"
    return True


# Task 4: Split BEFORE computing any statistics from the data ---------------
def train_test_split_idx(n, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    return idx[:cut], idx[cut:]


# Task 5: Avoid data leakage -------------------------------------------------
# Impute missing values using the TRAIN median only, then apply to both sets.
def impute_without_leakage(train, test, col):
    fill = train[col].median()  # learned from train only
    train = train.assign(**{col: train[col].fillna(fill)})
    test = test.assign(**{col: test[col].fillna(fill)})
    return train, test, fill


# Task 6: Documented, reusable pipeline -------------------------------------
def run_pipeline(seed=42):
    df = clean(make_dataset(seed))
    validate(df)
    tr_idx, te_idx = train_test_split_idx(len(df), seed=seed)
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    train, test, fill = impute_without_leakage(train, test, "income")
    return train, test, fill


if __name__ == "__main__":
    # Reproducibility: same seed -> identical data; different seed -> different.
    assert make_dataset(1).equals(make_dataset(1))
    assert not make_dataset(1).equals(make_dataset(2))

    raw = make_dataset()
    cleaned = clean(raw)
    assert len(raw) - len(cleaned) == 1
    assert validate(cleaned)

    train, test, fill = run_pipeline()
    assert train["income"].isna().sum() == 0 and test["income"].isna().sum() == 0
    assert len(set(train.index) & set(test.index)) == 0  # no row in both sets

    # The fill value came from train only, so it should match the train median
    # of the ORIGINAL (pre-imputation) train column, not the full dataset's.
    df = clean(make_dataset())
    tr_idx, _ = train_test_split_idx(len(df))
    assert fill == df.iloc[tr_idx]["income"].median()

    print("04_best_practices: all tasks passed")
    print(f"  train={len(train)} rows, test={len(test)} rows, train-median fill={fill:.2f}")
