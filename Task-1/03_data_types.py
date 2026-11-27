"""Assignment 3: Data types in data science.

Covers numerical vs categorical, ordinal vs nominal, dates, and how pandas
represents each. Run:  python 03_data_types.py
"""

import pandas as pd

# Task 1: Classify variables ----------------------------------------------
# Return the statistical type for each column of a small survey table.
CLASSIFICATION = {
    "age": "numerical (continuous/discrete ratio)",
    "city": "categorical (nominal)",
    "satisfaction": "categorical (ordinal)",
    "signup_date": "datetime",
    "is_member": "boolean (binary categorical)",
}


def build_survey():
    return pd.DataFrame(
        {
            "age": ["25", "31", "47", "29"],  # stored as text on purpose
            "city": ["Warangal", "Hyderabad", "Warangal", "Chennai"],
            "satisfaction": ["low", "high", "medium", "high"],
            "signup_date": ["2026-01-05", "2026-02-11", "2026-03-20", "2026-04-02"],
            "is_member": ["yes", "no", "yes", "yes"],
        }
    )


# Task 2: Convert to the right pandas dtypes ------------------------------
def fix_dtypes(df):
    out = df.copy()
    out["age"] = pd.to_numeric(out["age"])
    out["city"] = out["city"].astype("category")
    out["satisfaction"] = pd.Categorical(
        out["satisfaction"], categories=["low", "medium", "high"], ordered=True
    )
    out["signup_date"] = pd.to_datetime(out["signup_date"])
    out["is_member"] = out["is_member"].map({"yes": True, "no": False})
    return out


# Task 3: Ordinal comparison only works on ordered categoricals ----------
def high_satisfaction_rows(df):
    return df[df["satisfaction"] >= "medium"]


# Task 4: Encoding categoricals for models -------------------------------
# Nominal -> one-hot; ordinal -> integer codes that respect the order.
def encode(df):
    out = pd.get_dummies(df, columns=["city"], prefix="city", dtype=int)
    out["satisfaction"] = out["satisfaction"].cat.codes
    out["is_member"] = out["is_member"].astype(int)
    out["signup_month"] = out["signup_date"].dt.month
    return out.drop(columns=["signup_date"])


# Task 5: Structured vs unstructured -------------------------------------
STRUCTURE_EXAMPLES = {
    "structured": ["SQL table of orders", "CSV of sensor readings"],
    "semi-structured": ["JSON API response", "XML feed"],
    "unstructured": ["customer review text", "images", "audio recordings"],
}


if __name__ == "__main__":
    raw = build_survey()
    assert not pd.api.types.is_numeric_dtype(raw["age"])  # text before conversion

    df = fix_dtypes(raw)
    assert pd.api.types.is_numeric_dtype(df["age"])
    assert isinstance(df["city"].dtype, pd.CategoricalDtype)
    assert df["satisfaction"].cat.ordered
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])
    assert df["is_member"].dtype == bool

    assert len(high_satisfaction_rows(df)) == 3  # medium + two high

    enc = encode(df)
    assert {"city_Warangal", "city_Hyderabad", "city_Chennai"} <= set(enc.columns)
    assert enc["satisfaction"].tolist() == [0, 2, 1, 2]

    print("03_data_types: all tasks passed")
    print(df.dtypes.to_string())
