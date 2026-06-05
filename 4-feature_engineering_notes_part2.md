# Feature Engineering — Part 2 Complete Notes

---

## 1. Machine Learning Pipelines

### What is a Pipeline?

A Pipeline chains multiple preprocessing steps and a model together so that the output of each step is used as input of the next step. It ensures that the same transformations applied during training are also applied during testing and deployment.

**Why use Pipelines?**
- Prevents data leakage — fit/transform happens correctly on train/test automatically
- Cleaner code — one object handles everything
- Easy deployment — export one object instead of many separate transformers
- Works seamlessly with GridSearchCV and cross-validation

### Key Rule — Column Index vs Column Name

When chaining ColumnTransformers in a pipeline, pass **column indices** (not names) to the second and later transformers. This is because the output of a ColumnTransformer is a numpy array, not a DataFrame — so column names don't exist anymore.

Also remember: ColumnTransformer outputs transformed columns first, remainder columns after. Keep this in mind when specifying indices in the next step.

### Complete Pipeline Example — Titanic Dataset

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier

# Drop irrelevant columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['Survived']), df['Survived'],
    test_size=0.2, random_state=42
)

# Step 1 — Imputation
# Age (index 2) has missing values → fill with mean
# Embarked (index 6) has missing values → fill with most frequent
trf1 = ColumnTransformer([
    ('impute_age', SimpleImputer(), [2]),
    ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])
], remainder='passthrough')

# Step 2 — One Hot Encoding
# After trf1, Sex and Embarked are at new indices due to column reordering
# handle_unknown='ignore' → if test data has unseen category, output all zeros
trf2 = ColumnTransformer([
    ('ohe_sex_embarked', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), [1, 3])
], remainder='passthrough')

# Step 3 — Scaling
# slice(0,10) means columns 0 through 9
trf3 = ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0, 10))
], remainder='passthrough')

# Step 4 — Feature Selection
# SelectKBest keeps the k best features based on chi2 score
trf4 = SelectKBest(score_func=chi2, k=8)

# Step 5 — Model
trf5 = DecisionTreeClassifier()

# Create Pipeline — Method 1 (explicit names)
pipe = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2),
    ('trf3', trf3),
    ('trf4', trf4),
    ('trf5', trf5)
])

# Create Pipeline — Method 2 (auto names with make_pipeline)
pipe = make_pipeline(trf1, trf2, trf3, trf4, trf5)

# Fit — applies all transformations + trains model
pipe.fit(X_train, y_train)

# Inspect pipeline steps
pipe.named_steps
pipe.named_steps['trf1']

# Predict
y_pred = pipe.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

### handle_unknown='ignore' — Why It Matters

In training data, Embarked might have values: S, C, Q.
In test/production data, you might encounter a new value never seen during training.

Without handle_unknown='ignore' → error.
With handle_unknown='ignore' → new category gets all zeros (treated as unknown). Safe for deployment.

### Why We Don't Drop OHE Column in Decision Trees

Multicollinearity (dummy variable trap) matters most in linear models because it makes XᵀX non-invertible. Decision trees don't use matrix inversion — they just ask yes/no questions at each node. Multicollinearity doesn't break decision trees, so we skip drop='first'.

### Cross Validation with Pipeline

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X, y, cv=10, scoring='accuracy')
print(f"Mean Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

Pipeline ensures no data leakage during cross-validation — each fold fits transformers only on training folds and transforms validation fold using those fitted transformers.

### Hyperparameter Tuning with Pipeline

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'decisiontreeclassifier__max_depth': [3, 5, 7, None],
    'decisiontreeclassifier__min_samples_split': [2, 5, 10]
}
# Format: stepname__parametername (double underscore)

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

### Exporting and Loading a Pipeline

```python
import pickle

# Save
pickle.dump(pipe, open('pipe.pkl', 'wb'))  # wb = write binary

# Load
pipe = pickle.load(open('pipe.pkl', 'rb'))  # rb = read binary

# Use on new data
test_input = pd.DataFrame({
    'Pclass': [2],
    'Sex': ['male'],
    'Age': [31.0],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [10.5],
    'Embarked': ['S']
})

pipe.predict(test_input)
```

**Why pickle?** The pipeline stores the fitted state of every transformer (the mean learned by SimpleImputer, the categories learned by OHE, etc.). Pickling preserves all of this so you don't need to retrain on deployment.

---

## 2. Function Transformer & Power Transformer

### Why Mathematical Transformation?

Many ML models (Linear Regression, Logistic Regression, LDA) assume that input features are normally distributed. Real data is often skewed. Mathematical transformations reshape the distribution to be closer to normal.

**Decision trees don't benefit** from transformations for the same reason they don't need scaling — rank order doesn't change.

### How to Check if Data is Normal

```python
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Method 1 — Histogram + KDE
sns.histplot(x='column', kde=True, stat='density', data=df)

# Method 2 — Skewness
df['column'].skew()
# 0 = normal, positive = right skewed, negative = left skewed
# Rule of thumb: |skew| > 0.5 is considered skewed

# Method 3 — Q-Q Plot (most reliable)
stats.probplot(df['column'], dist="norm", plot=plt)
plt.show()
# If all points lie on the 45° diagonal line → normal distribution
# Deviations from line → deviation from normality
```

### Sklearn's Three Transformation Tools

| Tool | Transforms | When to Use |
|---|---|---|
| FunctionTransformer | Log, Reciprocal, Square, Square Root, Custom | When you visually know the skew type |
| PowerTransformer | Box-Cox, Yeo-Johnson | When you want automated optimization |
| QuantileTransformer | Maps to uniform/normal distribution | Rarely used |

---

### FunctionTransformer

#### 1. Log Transform

**Formula:** x' = log(x) or log(1 + x)

**Use when:** Data is right-skewed (long tail on the right), all values strictly positive.

**Why log1p over log:**
- log(0) = -infinity → error if any value is 0
- log1p(x) = log(1 + x) → safe when x = 0

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

trf = FunctionTransformer(func=np.log1p)

X_train_transformed = pd.DataFrame(
    trf.fit_transform(X_train),
    columns=X_train.columns
)
X_test_transformed = pd.DataFrame(
    trf.transform(X_test),
    columns=X_test.columns
)
```

#### 2. Reciprocal Transform

**Formula:** x' = 1/x

**Effect:** Small values become large, large values become small. Compresses the right tail aggressively.

**Limitation:** Cannot be used when x = 0 (division by zero).

#### 3. Square Transform (x²)

**Formula:** x' = x²

**Use when:** Data is left-skewed (long tail on the left).

#### 4. Square Root Transform (√x)

**Formula:** x' = √x

**Use when:** Mild right skew. Less aggressive than log transform.

**Rule of thumb:** Try all transforms, compare Q-Q plots and skewness scores, pick the one that gives the most normal result. The code is small — testing all four takes minutes.

---

### PowerTransformer

More general and automated than FunctionTransformer. Internally uses MLE to find the optimal lambda parameter.

#### Box-Cox Transform

**Formula:** Uses lambda parameter optimized by MLE.

**Restriction:** Only works for strictly positive values (x > 0). Zero and negative values are invalid.

#### Yeo-Johnson Transform

**Extends Box-Cox** to handle zero and negative values. No restrictions on input range.

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox
pt = PowerTransformer(method='box-cox')  # only for x > 0

# Yeo-Johnson
pt = PowerTransformer(method='yeo-johnson')  # works for all values

X_train_transformed = pd.DataFrame(
    pt.fit_transform(X_train),
    columns=X_train.columns
)
X_test_transformed = pd.DataFrame(
    pt.transform(X_test),
    columns=X_test.columns
)
```

### When to Use Which

| Situation | Use |
|---|---|
| Strong right skew, strictly positive, log makes intuitive sense | FunctionTransformer (log1p) |
| Left skew | FunctionTransformer (x²) |
| Skewness is tricky, unsure which transform | PowerTransformer (Yeo-Johnson) |
| Strictly positive data, want automated optimization | PowerTransformer (Box-Cox) |
| Has zeros or negative values | PowerTransformer (Yeo-Johnson) |

---

## 3. Binning and Binarization

### Why Convert Numerical to Categorical?

Sometimes the exact numerical value is less important than the range it belongs to. Converting continuous features to categories can:
- Handle outliers (extreme values get placed in a boundary bin)
- Improve value spread (more uniform distribution)
- Capture non-linear relationships more easily
- Make patterns more interpretable

**Example:** Ages 18, 20, 24, 76, 32, 98 → 'Young', 'Middle Aged', 'Old'

### Binarization

Converts numerical values into two categories based on a threshold.

```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=50)  # values > 50 → 1, values ≤ 50 → 0
X_transformed = binarizer.fit_transform(X)
```

**Example:** Marks → Pass (1) or Fail (0) based on threshold 40.

### Discretization / Binning

Converts continuous values into intervals (bins).

#### Types of Binning

**1. Unsupervised Binning (doesn't use target variable)**

| Type | Description | Use When |
|---|---|---|
| Equal Width | All bins have same range | Data is uniformly distributed |
| Equal Frequency (Quantile) | Each bin has same number of observations | Data is skewed |
| KMeans | Uses KMeans clustering to group similar values | Natural clusters exist |

**2. Supervised Binning**
- **Decision Tree Binning** — A decision tree identifies optimal split points based on target prediction. Most powerful — uses label information.

**3. Custom Binning**
- Bins manually defined using domain knowledge or business rules.
- Example: Age → [0-18: Child], [18-35: Young Adult], [35-60: Middle Aged], [60+: Senior]

### Implementation with KBinsDiscretizer

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

# Parameters:
# n_bins — number of bins
# strategy — 'uniform' (equal width), 'quantile' (equal frequency), 'kmeans'
# encode — 'ordinal' (integers 0,1,2...) or 'onehot' (binary columns)

kbin_age = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
kbin_fare = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

trf = ColumnTransformer([
    ('age_bins', kbin_age, [0]),
    ('fare_bins', kbin_fare, [1])
])

X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)

# Inspect bin edges
trf.named_transformers_['age_bins'].bin_edges_

# Visualise the binning
output = pd.DataFrame({
    'age': X_train['Age'],
    'age_bin': X_train_trf[:, 0],
    'fare': X_train['Fare'],
    'fare_bin': X_train_trf[:, 1]
})

output['age_labels'] = pd.cut(
    x=X_train['Age'],
    bins=trf.named_transformers_['age_bins'].bin_edges_[0].tolist()
)
```

---

## 4. Handling Date and Time Variables

By default pandas reads dates as 'object' (string). You must convert to datetime to extract useful features.

### Working with Dates

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.month_name()
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek      # 0=Monday, 6=Sunday
df['day_name'] = df['date'].dt.day_name()
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['date'].dt.dayofweek >= 5  # True for Saturday/Sunday
df['week_of_year'] = df['date'].dt.isocalendar().week
```

### Working with Time

```python
df['datetime'] = pd.to_datetime(df['datetime'])

df['time'] = df['datetime'].dt.time
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['second'] = df['datetime'].dt.second

# Useful derived features
df['is_business_hours'] = df['hour'].between(9, 17)
df['time_of_day'] = pd.cut(
    df['hour'],
    bins=[0, 6, 12, 17, 21, 24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night2']
)
```

### Elapsed Time Features

```python
# Days since a reference date
reference_date = pd.Timestamp('2024-01-01')
df['days_since_ref'] = (df['date'] - reference_date).dt.days

# Time between two dates
df['duration_days'] = (df['end_date'] - df['start_date']).dt.days
```

### Why Extract These Features?

ML models can't understand raw datetime objects — they need numbers. But more importantly, extracted features capture meaningful patterns:
- Hour of day → captures time-of-day effects (rush hour, business hours)
- Day of week → captures weekly seasonality
- Month → captures seasonal patterns
- Is weekend → captures behaviour differences between weekdays and weekends

---

## 5. Handling Missing Data

### Three Types of Missingness

Understanding WHY data is missing determines HOW you handle it:

| Type | What it means | Example | Safe to drop? |
|---|---|---|---|
| **MCAR** (Missing Completely At Random) | Missing has no relationship with any data | Random sensor failure | Yes — safe to drop |
| **MAR** (Missing At Random) | Missing depends on other observed columns | Income missing more often for younger people | Impute using other columns |
| **MNAR** (Missing Not At Random) | Missing depends on the missing value itself | High earners don't report income | Dangerous — dropping biases results |

### How to Check if Missing is Random

```python
# Numerical — compare distribution before and after dropping missing rows
df['Age'].plot(kind='kde', label='Original')
df['Age'].dropna().plot(kind='kde', label='After dropping NaN')
plt.legend()
# If distributions are similar → missing is likely random (MCAR/MAR)
# If distributions differ → missing is not random (MNAR)

# Categorical — compare category ratios before and after
df['Embarked'].value_counts(normalize=True)
df.dropna(subset=['Embarked'])['Embarked'].value_counts(normalize=True)
# If ratios are similar → missing is likely random
```

---

### Method 1 — Complete Case Analysis (CCA)

Remove all rows with any missing value. Also called listwise deletion.

```python
# Check how much data survives
print(f"Original: {len(df)} rows")
print(f"After CCA: {len(df.dropna())} rows")
print(f"Retained: {len(df.dropna()) / len(df) * 100:.1f}%")

# Only drop rows missing in specific columns
cols_with_few_missing = [col for col in df.columns
                         if 0 < df[col].isnull().mean() < 0.05]
df_clean = df[cols_with_few_missing].dropna()
```

**When to use:** Missing is MCAR, missing percentage is small (<5%), enough data remains after dropping.

**When NOT to use:** Large percentage missing, MNAR data, small dataset where you can't afford to lose rows.

---

### Method 2 — Univariate Imputation

Filling missing values using only that same column.

#### Numerical Columns

**Mean Imputation:**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
# Use when: data is normally distributed, missing is random, <5% missing
```

**Median Imputation:**
```python
imputer = SimpleImputer(strategy='median')
# Use when: data is skewed or has outliers (median is robust)
```

**Arbitrary Value Imputation:**
```python
imputer = SimpleImputer(strategy='constant', fill_value=-999)
# Use when: you want the model to learn that -999 means "missing"
# Some tree-based models can learn this pattern
```

**Disadvantages of mean/median imputation:**
- Reduces variance — the imputed values are all the same
- Changes correlation between columns
- Changes the distribution shape

**Always verify after imputing:**
```python
# Check variance
print('Before:', df['Age'].var())
print('After:', df['Age_imputed'].var())

# Check distribution
df['Age'].plot(kind='kde', label='Original')
df['Age_imputed'].plot(kind='kde', label='Imputed')
plt.legend()
plt.show()

# Check correlation
print(df[['Age', 'Fare']].corr())
print(df[['Age_imputed', 'Fare']].corr())
```

#### Categorical Columns

**Mode Imputation (most frequent):**
```python
imputer = SimpleImputer(strategy='most_frequent')
# Use when: one category clearly dominates, missing is random
```

**Constant 'Missing' Category:**
```python
imputer = SimpleImputer(strategy='constant', fill_value='Missing')
# Use when: many non-random missing values
# The 'Missing' category itself becomes informative
```

---

### Method 3 — Random Imputation

Fill missing values with random samples drawn from the non-missing values of the same column.

**Advantage:** Preserves the original variance (unlike mean/median which reduce it).

**Disadvantage:** Reduces correlation between columns (introduces randomness).

```python
# Numerical
X_train['Age_imputed'] = X_train['Age'].copy()
X_test['Age_imputed'] = X_test['Age'].copy()

# Fill training missing values with random samples from training data
X_train['Age_imputed'][X_train['Age_imputed'].isnull()] = (
    X_train['Age'].dropna()
    .sample(X_train['Age'].isnull().sum(), replace=True)
    .values
)

# Fill test missing values with random samples from TRAINING data (not test)
X_test['Age_imputed'][X_test['Age_imputed'].isnull()] = (
    X_train['Age'].dropna()
    .sample(X_test['Age'].isnull().sum(), replace=True)
    .values
)
# Critical: always sample from X_train, never X_test — avoid data leakage
```

---

### Method 4 — Missing Indicator

Create a new binary column flagging whether the original value was missing.

```python
from sklearn.impute import SimpleImputer

# add_indicator=True automatically creates _missing indicator columns
si = SimpleImputer(strategy='mean', add_indicator=True)

X_train_transformed = si.fit_transform(X_train)
X_test_transformed = si.transform(X_test)

# This creates:
# Original columns with imputed values
# + New boolean columns: Age_missing, Fare_missing etc.
```

**When it works:** When missingness itself is informative. For example, if expensive houses are less likely to have a garage quality rating because the field was left blank intentionally — the missingness tells you something about the house.

---

### Method 5 — KNN Imputer (Multivariate)

Uses K nearest neighbours to fill missing values. For each row with a missing value, it finds K most similar rows (based on other columns) and fills with their average.

**Advantage:** Generally performs better than univariate methods because it uses relationships between columns.

**Disadvantages:**
- Slow on large datasets — computes distances between all rows
- Requires storing entire training dataset in memory at prediction time

```python
from sklearn.impute import KNNImputer

knn = KNNImputer(
    n_neighbors=3,
    weights='distance'  # closer neighbours have more influence
    # weights='uniform' → all neighbours equally weighted
)

X_train_trf = knn.fit_transform(X_train)
X_test_trf = knn.transform(X_test)
```

---

### Method 6 — Iterative Imputer / MICE (Multivariate)

MICE = Multivariate Imputation by Chained Equations.

Models each column with missing values as a function of all other columns. Iterates multiple times until convergence.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

imputer = IterativeImputer(
    estimator=BayesianRidge(),  # model used to predict missing values
    max_iter=10,                # number of iterations
    random_state=42
)

X_train_trf = imputer.fit_transform(X_train)
X_test_trf = imputer.transform(X_test)
```

**When to use:** Complex datasets where columns are strongly related, moderate missing percentage, accuracy matters more than speed.

---

### Complete Imputation Decision Guide

```
Missing data detected
        │
        ├── Is it MNAR?
        │     └── Yes → Missing indicator + domain expertise required
        │
        ├── Is it <5% and MCAR?
        │     └── Yes → CCA (drop rows) is safe
        │
        ├── Numerical column?
        │     ├── Normal distribution → Mean imputation
        │     ├── Skewed / outliers → Median imputation
        │     ├── Want to preserve variance → Random imputation
        │     └── Complex relationships → KNN or MICE
        │
        └── Categorical column?
              ├── One dominant category → Mode imputation
              └── Many missing, non-random → 'Missing' category
```

---

### Automatic Strategy Selection with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('model', DecisionTreeClassifier())
])

param_grid = {
    'imputer__strategy': ['mean', 'median', 'most_frequent']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_params_)
# Automatically finds which imputation strategy works best for your data
```

---

## Quick Reference — All Methods Summary

| Method | Preserves Variance | Preserves Correlation | Speed | Best For |
|---|---|---|---|---|
| CCA (Drop rows) | ✓ | ✓ | Fast | MCAR, <5% missing |
| Mean/Median | ✗ | ✗ | Fast | Simple, quick baseline |
| Mode | ✗ | ✗ | Fast | Categorical, dominant category |
| Random | ✓ | ✗ | Fast | Preserving distribution shape |
| Missing Indicator | N/A | N/A | Fast | When missingness is informative |
| KNN | ✓ | ✓ | Slow | Best accuracy, small-medium data |
| MICE | ✓ | ✓ | Slowest | Best accuracy, complex relationships |
