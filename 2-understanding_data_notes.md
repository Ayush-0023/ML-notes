# Understanding Your Data — Complete Notes

---

## Why Understanding Data Matters

Most people load a dataset and immediately start building models. That's a mistake. A model is only as good as the data you feed it — and you can't make good decisions about preprocessing, feature engineering, or model selection without first understanding what you're working with.

Understanding your data answers three questions:
- **What do I have?** — size, types, structure
- **What's wrong with it?** — missing values, duplicates, outliers, wrong types
- **What patterns exist?** — distributions, correlations, class balance

---

## The Complete Checklist

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
```

---

## 1. How Big is the Data?

```python
# Shape
df.shape
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Memory usage — important for large datasets
memory_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory: {memory_mb:.2f} MB")
```

**Why memory matters:** A dataset with 100k rows and 500 columns can easily exceed RAM. Knowing memory usage upfront helps you decide whether to use chunking, reduce dtypes, or sample the data.

---

## 2. What Does the Data Look Like?

```python
# sample() is better than head()
df.sample(5, random_state=42)
# head() always shows the first 5 rows — often the cleanest, most "normal" rows
# sample() shows random rows — more representative of the full dataset

# head() and tail() together reveal if data is sorted
df.head()   # first rows
df.tail()   # last rows — check if sorted by time, ID etc.
```

---

## 3. Data Types and Memory Optimization

```python
df.info()
# Shows: column names, non-null counts, data types, memory usage
```

**What to look for and fix:**

```python
# Float columns that should be int — wastes memory
df['Age'] = df['Age'].astype(int)

# Object columns that are actually dates
df['date'] = pd.to_datetime(df['date'])

# Object columns with few unique values — use category type
# Stores the string once, uses integers internally — huge memory saving
df['gender'] = df['gender'].astype('category')
df['city'] = df['city'].astype('category')

# Check memory before and after optimization
print(df.memory_usage(deep=True))
```

**Rule of thumb:** If a column has fewer than 50 unique values and is stored as object, convert to category. Memory can drop by 50-80%.

---

## 4. Missing Values

```python
# Count of missing values
df.isnull().sum()

# Percentage — much more meaningful than count
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
print(missing_pct[missing_pct > 0])  # only show columns with missing values

# Columns with more than 5% missing
high_missing = missing_pct[missing_pct > 5]
print(high_missing)

# Visualise missing value pattern
import missingno as msno
msno.matrix(df)  # shows WHERE missing values are located
# Random scatter = MCAR (safe to drop/impute)
# Clustered pattern = MAR or MNAR (need careful handling)
```

**Why percentage matters:** 500 missing values means nothing without context. Is that 1% or 50% of your data? The percentage tells you how seriously to treat it.

**Why pattern matters:** Are missing values random throughout the dataset? Or do they cluster in specific rows/columns? Clustering suggests the missingness is not random — which changes how you handle it.

---

## 5. Statistical Summary

```python
# Numerical columns
df.describe()

# All columns including categorical
df.describe(include='all')
```

**What to actively look for in describe():**

| What you see | What it means | What to do |
|---|---|---|
| max is 10x the mean | Likely outlier | Investigate with boxplot |
| std very high relative to mean | High variance | May need scaling or transformation |
| 25th and 75th percentile very close | Low variance feature | Consider dropping |
| count < total rows | Missing values present | Handle missing values |
| min is negative | Check if meaningful | Domain knowledge needed |
| mean and median (50%) very different | Skewed distribution | May need log transform |

---

## 6. Duplicate Values

```python
# Count full row duplicates
df.duplicated().sum()

# Duplicates based on specific columns only
# (same customer appearing twice with different dates is NOT a duplicate)
df.duplicated(subset=['name', 'email']).sum()

# View the duplicates
df[df.duplicated()]

# Remove duplicates — keeps first occurrence
df.drop_duplicates(inplace=True)

# Keep last occurrence instead
df.drop_duplicates(keep='last', inplace=True)
```

**Important distinction:** Full row duplicates are almost always data errors. Duplicates on a subset of columns may be legitimate (same customer, multiple orders).

---

## 7. Unique Values

```python
# Number of unique values per column
df.nunique()

# Interpretation:
# High nunique → continuous or high cardinality → might need encoding or binning
# Low nunique → categorical → good candidate for value_counts()
# nunique == 1 → constant column → zero variance → drop it
# nunique == n_rows → likely an ID column → drop it

# Actual unique values for categorical columns
df['city'].unique()
df['gender'].unique()
```

---

## 8. Categorical Column Distributions

```python
# Frequency of each category
df['gender'].value_counts()

# As percentage
df['gender'].value_counts(normalize=True) * 100

# Including missing values
df['gender'].value_counts(dropna=False)

# Loop through all categorical columns
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col} ({df[col].nunique()} unique values):")
    print(df[col].value_counts())
```

**What to look for:**
- **Class imbalance in target column** — if 95% are class 0 and 5% are class 1, your model will predict 0 for everything and still look accurate
- **Rare categories** — categories with very few samples; the model can't learn from 3 examples. Group these into "Others"
- **Unexpected values** — typos, inconsistent capitalisation (Male vs male vs MALE)

---

## 9. Skewness and Kurtosis

```python
# Skewness
df.skew(numeric_only=True)
# 0 = normal distribution
# Positive = right skewed (long tail on right)
# Negative = left skewed (long tail on left)
# Rule of thumb: |skew| > 0.5 is considered skewed, > 1 is highly skewed

# Kurtosis
df.kurtosis(numeric_only=True)
# High kurtosis = heavy tails, more outliers than normal distribution
# Low kurtosis = light tails, fewer outliers
```

**Why this matters:** Many ML algorithms assume normally distributed features. High skewness tells you which columns need transformation (log, Box-Cox, Yeo-Johnson) before modelling.

---

## 10. Correlation

```python
# Pearson correlation — default, measures linear relationship
df.corr(numeric_only=True)

# Spearman — better for non-linear monotonic relationships
df.corr(method='spearman', numeric_only=True)

# Kendall — better for small datasets with outliers
df.corr(method='kendall', numeric_only=True)

# Single column correlation with target
df.corr(numeric_only=True)['target'].sort_values(ascending=False)

# Heatmap — far more readable than raw numbers
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    mask=np.triu(np.ones_like(df.corr(numeric_only=True)))  # lower triangle only
)
plt.show()
```

**When to use which correlation:**

| Method | Use When |
|---|---|
| Pearson | Linear relationship, no outliers |
| Spearman | Monotonic but not necessarily linear relationship |
| Kendall | Small dataset, many outliers |

**Critical limitation of correlation:**
> Correlation only measures linear relationships. Two columns can have zero Pearson correlation but a strong non-linear relationship. Always plot — don't rely on numbers alone.

---

## The Complete Understanding Data Script

Run this on any new dataset before doing anything else:

```python
def understand_data(df):

    print("=" * 60)
    print("1. SHAPE & MEMORY")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n" + "=" * 60)
    print("2. DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("3. MISSING VALUES (%)")
    print("=" * 60)
    missing = (df.isnull().sum() / len(df) * 100).round(2)
    print(missing[missing > 0] if missing.any() else "No missing values")

    print("\n" + "=" * 60)
    print("4. DUPLICATES")
    print("=" * 60)
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("5. UNIQUE VALUES PER COLUMN")
    print("=" * 60)
    print(df.nunique())

    print("\n" + "=" * 60)
    print("6. STATISTICAL SUMMARY")
    print("=" * 60)
    print(df.describe(include='all'))

    print("\n" + "=" * 60)
    print("7. SKEWNESS (numerical columns)")
    print("=" * 60)
    print(df.skew(numeric_only=True).round(2))

    print("\n" + "=" * 60)
    print("8. CATEGORICAL DISTRIBUTIONS")
    print("=" * 60)
    for col in df.select_dtypes(include='object').columns:
        print(f"\n{col}:")
        print(df[col].value_counts())

understand_data(df)
```

---

## Quick Reference — What Each Method Tells You

| Method | What it tells you | Key thing to look for |
|---|---|---|
| df.shape | Size of dataset | Enough rows for the number of features? |
| df.memory_usage() | RAM consumption | Do I need chunking or dtype optimization? |
| df.sample() | Random snapshot | Unexpected values, data format issues |
| df.info() | Types + non-null counts | Wrong types, hidden missing values |
| df.isnull().sum() | Missing value count | Which columns need imputation? |
| df.describe() | Statistical summary | Outliers, low variance, skewness |
| df.duplicated() | Duplicate rows | Data entry errors |
| df.nunique() | Unique value counts | ID columns, constant columns, cardinality |
| df.value_counts() | Category frequencies | Class imbalance, rare categories |
| df.skew() | Distribution shape | Which columns need transformation? |
| df.corr() | Linear relationships | Multicollinearity, useful features |

---

## Key Insights That Come From This Step

**If you find class imbalance in target column:**
→ Don't use accuracy as metric. Use precision, recall, F1. Consider SMOTE or class weights.

**If you find high skewness in a numerical column:**
→ Apply log transform (right skew) or square transform (left skew) before modelling with linear algorithms.

**If you find high correlation between two input features:**
→ Multicollinearity risk. Consider dropping one or using Ridge regression.

**If you find near-zero variance in a column:**
→ That feature contributes almost nothing. Consider dropping it.

**If you find a column with nunique == n_rows:**
→ That's an ID column. Drop it — it will cause severe overfitting.

**If you find unexpected categories (typos, inconsistent capitalisation):**
→ Clean and standardise before encoding.

Understanding your data isn't a step you do once and move on. Every time something goes wrong with your model, you come back here first.
