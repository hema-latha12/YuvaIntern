"""Assignment 1: Python basics for data science.

Each task has a short description, a solution, and an assert that checks it.
Run:  python 01_python_basics.py
"""

from collections import Counter


# Task 1: Data structures -------------------------------------------------
# Given a list of (name, score) tuples, build a dict {name: score}
# and return the name with the highest score.
def top_scorer(records):
    scores = dict(records)
    return max(scores, key=scores.get)


# Task 2: Loops + conditionals --------------------------------------------
# FizzBuzz-style: return a list for 1..n using "Fizz", "Buzz", "FizzBuzz".
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


# Task 3: Functions + dictionaries ----------------------------------------
# Count word frequencies (case-insensitive, punctuation ignored).
def word_freq(text):
    cleaned = "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in text)
    return dict(Counter(cleaned.split()))


# Task 4: Comprehensions --------------------------------------------------
# Squares of the even numbers in a list, as a list comprehension.
def even_squares(nums):
    return [x**2 for x in nums if x % 2 == 0]


# Task 5: Dict comprehension ----------------------------------------------
# Map each word to its length, keeping only words longer than 3 characters.
def long_word_lengths(words):
    return {w: len(w) for w in words if len(w) > 3}


# Task 6: Recursion -------------------------------------------------------
def factorial(n):
    if n < 0:
        raise ValueError("n must be non-negative")
    return 1 if n in (0, 1) else n * factorial(n - 1)


# Task 7: Error handling --------------------------------------------------
# Convert strings to floats, returning None for values that cannot be parsed.
def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Task 8: File-free string processing -------------------------------------
# Check whether a string is a palindrome, ignoring case and non-alphanumerics.
def is_palindrome(s):
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]


if __name__ == "__main__":
    assert top_scorer([("Asha", 82), ("Ravi", 91), ("Meera", 77)]) == "Ravi"
    assert fizzbuzz(15)[-1] == "FizzBuzz" and fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"]
    assert word_freq("The cat. The hat!") == {"the": 2, "cat": 1, "hat": 1}
    assert even_squares([1, 2, 3, 4, 5, 6]) == [4, 16, 36]
    assert long_word_lengths(["a", "data", "science"]) == {"data": 4, "science": 7}
    assert factorial(5) == 120 and factorial(0) == 1
    assert safe_float("3.5") == 3.5 and safe_float("abc") is None and safe_float(None) is None
    assert is_palindrome("A man, a plan, a canal: Panama") and not is_palindrome("data")
    print("01_python_basics: all tasks passed")
