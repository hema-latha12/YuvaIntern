# Data Science Fundamentals Assessment

A hands-on assessment covering the core foundations of data science. Each
assignment is a self-contained Python script with tasks, solutions, and
`assert` checks so you can see immediately that everything works.

## Objectives

1. Demonstrate core data science concepts: Python basics, statistical
   foundations, data types, and industry best practices.
2. Complete assignments covering all fundamentals.

## Assignments

| # | File | Topics |
|---|------|--------|
| 1 | `01_python_basics.py` | Data structures, loops, functions, comprehensions, recursion, error handling |
| 2 | `02_statistics.py` | Mean/median/mode, variance, std, IQR, correlation, z-scores, outliers, t-test, confidence intervals |
| 3 | `03_data_types.py` | Numerical vs categorical, ordinal vs nominal, datetimes, pandas dtypes, encoding |
| 4 | `04_best_practices.py` | Reproducibility, cleaning, validation, train/test split, avoiding data leakage |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python 01_python_basics.py
python 02_statistics.py
python 03_data_types.py
python 04_best_practices.py
```

Each script prints `all tasks passed` when every check succeeds.

## Key takeaways

- **Python:** comprehensions and dictionaries cover most day-to-day data wrangling.
- **Statistics:** pick the measure that fits the data (median over mean when outliers are present); report uncertainty with confidence intervals.
- **Data types:** the type decides the valid operations and encoding (nominal -> one-hot, ordinal -> ordered codes).
- **Best practices:** seed randomness, validate early, split before computing statistics, and never let test data influence training.
