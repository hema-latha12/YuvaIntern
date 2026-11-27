"""Assignment 2: Statistical foundations.

Implements core statistics from scratch, then cross-checks against NumPy/SciPy.
Run:  python 02_statistics.py
"""

import math
from collections import Counter

import numpy as np
from scipy import stats


# Task 1: Measures of central tendency ------------------------------------
def mean(xs):
    return sum(xs) / len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def mode(xs):
    counts = Counter(xs)
    top = max(counts.values())
    return sorted(k for k, v in counts.items() if v == top)


# Task 2: Spread -----------------------------------------------------------
def variance(xs, sample=True):
    m = mean(xs)
    ss = sum((x - m) ** 2 for x in xs)
    return ss / (len(xs) - 1 if sample else len(xs))


def std(xs, sample=True):
    return math.sqrt(variance(xs, sample))


def iqr(xs):
    q1, q3 = np.percentile(xs, [25, 75])
    return q3 - q1


# Task 3: Correlation ------------------------------------------------------
def pearson(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den


# Task 4: Z-scores and outliers --------------------------------------------
def zscores(xs):
    m, s = mean(xs), std(xs)
    return [(x - m) / s for x in xs]


def outliers_iqr(xs):
    q1, q3 = np.percentile(xs, [25, 75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return [x for x in xs if x < low or x > high]


# Task 5: Hypothesis testing -----------------------------------------------
# Two-sample Welch t-test: do the two groups have different means at alpha=0.05?
def welch_t_test(a, b, alpha=0.05):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p), "reject_null": bool(p < alpha)}


# Task 6: Confidence interval for a mean -----------------------------------
def mean_ci(xs, confidence=0.95):
    n = len(xs)
    se = std(xs) / math.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    m = mean(xs)
    return (m - t_crit * se, m + t_crit * se)


if __name__ == "__main__":
    data = [4, 8, 6, 5, 3, 8, 9, 5, 8, 7]

    assert math.isclose(mean(data), np.mean(data))
    assert math.isclose(median(data), np.median(data))
    assert mode(data) == [8]
    assert math.isclose(variance(data), np.var(data, ddof=1))
    assert math.isclose(variance(data, sample=False), np.var(data))
    assert math.isclose(std(data), np.std(data, ddof=1))

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    assert math.isclose(pearson(x, y), np.corrcoef(x, y)[0, 1])

    assert math.isclose(sum(zscores(data)), 0, abs_tol=1e-9)
    assert outliers_iqr([10, 12, 11, 13, 12, 11, 95]) == [95]

    rng = np.random.default_rng(42)
    a = rng.normal(50, 5, 100)
    b = rng.normal(55, 5, 100)
    result = welch_t_test(a, b)
    assert result["reject_null"]  # means truly differ by 5, n=100

    lo, hi = mean_ci(a)
    assert lo < a.mean() < hi
    print("02_statistics: all tasks passed")
    print(f"  Welch t-test: t={result['t']:.2f}, p={result['p']:.2e}")
    print(f"  95% CI for mean of a: ({lo:.2f}, {hi:.2f})")
