# Feature Engineering — Complete Notes

## What is Feature Engineering?

Feature engineering is the process of using domain knowledge to extract, transform, and create features from raw data to improve the performance of ML models.

It can be divided into:
- **Feature Transformation** — Changes in existing features like missing value imputation, handling categorical features, outlier detection, feature scaling
- **Feature Construction** — Creating new features from existing ones
- **Feature Selection** — Choosing the most relevant features
- **Feature Extraction** — Reducing dimensions while preserving information (e.g. PCA)

---

## 1. Feature Scaling

Feature scaling is a technique to standardize the independent features present in the data in a fixed range.

### Why Do We Need Feature Scaling?

Some ML algorithms are sensitive to the scale of features, others are not.

**Algorithms that NEED scaling:**
- **KNN** — uses Euclidean distance. A salary column (0–100,000) will completely dominate an age column (0–100) without scaling. The distance calculation becomes meaningless.
- **Logistic Regression & Linear Regression with Gradient Descent** — gradient updates happen at different rates for features with different scales. Without scaling, weights update very differently for each feature, making convergence slow or unstable.
- **PCA** — requires variance control and mean centring. Without scaling, features with larger ranges dominate the principal components.
- **K-Means Clustering** — uses distance, same problem as KNN.
- **ANN/Neural Networks** — gradient descent based, same issue as above.

**Algorithms that DON'T need scaling:**
- **Decision Trees** — split data by asking "is feature X above or below threshold T?" This is a rank-based operation. Scaling changes the values but never changes the rank order. So the splits remain identical regardless of scale.
- **Random Forest** — made of decision trees, same reasoning.
- **Gradient Boosting (XGBoost, LightGBM)** — tree-based, doesn't need scaling.
- **Naive Bayes** — probability-based, not distance-based.

### Critical Rule — Fit Only on Training Data

This is one of the most common mistakes in ML:

```python
# WRONG — leaks test information into training
scaler.fit(X)  # fits on entire dataset including test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CORRECT — learn only from training data
scaler.fit(X_train)  # fits ONLY on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why?** If you fit on all data, your scaler has "seen" the test set — its min, max, mean, std. In production you never have test data upfront. Fitting on all data gives artificially optimistic results. This is called **data leakage**.

---

### Standardization (Z-Score Normalization)

**Formula:**
> Xi' = (Xi - X̄) / σ

**Result:** New column has mean = 0 and std dev = 1.

**Geometric Intuition:** We do mean centring of the data (shift so mean is at 0), then squish the data in scale of std dev such that most data falls between -3 and +3.

**Effect on Outliers:** Outliers still remain outliers even after scaling because scaling only changes the scale. The relative difference between normal data points and outliers remains the same.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], 
    test_size=0.3, random_state=0
)

scaler = StandardScaler()
scaler.fit(X_train)  # learn mean and std from training only

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame (transform returns numpy array)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Verify: mean ≈ 0, std ≈ 1
np.round(X_train_scaled.describe(), 1)
```

**When to use:** KNN, Logistic Regression, PCA, Gradient Descent, ANN — whenever features need to be comparable in scale but you don't need a fixed range.

---

### Normalization

Normalization is a technique to change the values of numeric columns to use a common scale, without distorting differences in the ranges of values or losing information.

#### MinMax Scaling

**Formula:**
> Xi' = (Xi - Xmin) / (Xmax - Xmin)

**Result:** New column has range [0, 1].

**Geometric Intuition:** All data points are squished inside a square of size 1×1 in case of 2D, a cube of size 1×1×1 in case of 3D, and so on.

**Effect on Outliers:** Sensitive to outliers. One extreme value compresses all other values into a tiny range.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Verify: min=0, max=1
np.round(X_train_scaled.describe(), 1)
```

**When to use:** Neural networks (especially with sigmoid/tanh activations), image pixel values (0–255 → 0–1).

---

#### Mean Normalization

**Formula:**
> Xi' = (Xi - Xmean) / (Xmax - Xmin)

**Result:** Range is [-1, 1] with mean at 0. Both mean centring and bounded range.

---

#### MaxAbs Scaling

**Formula:**
> Xi' = Xi / |Xmax|

**Result:** Range is [-1, 1].

**When to use:** Data where many zeros are present (sparse data). Doesn't shift/centre the data, so sparsity is preserved. Commonly used in text data (TF-IDF).

---

#### Robust Scaling

**Formula:**
> Xi' = (Xi - Xmedian) / IQR

**Result:** Robust to outliers.

**Why it's robust:** StandardScaler uses mean and std — both heavily influenced by outliers. One extreme value shifts the mean and inflates std, distorting the entire scaling. RobustScaler uses median and IQR — both resistant to outliers. The median doesn't move when you add extreme values. IQR only captures the middle 50% of data.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

**When to use:** Data with significant outliers that you don't want to remove.

---

### Complete Scaling Comparison

| Scaler | Formula | Range | Outlier Robust | Use When |
|---|---|---|---|---|
| StandardScaler | (x - mean) / std | (-∞, +∞) | No | Gaussian-like data, PCA, gradient descent |
| MinMaxScaler | (x - min) / (max - min) | [0, 1] | No | Neural networks, image data |
| MeanNormalization | (x - mean) / (max - min) | [-1, 1] | No | Mean centred + bounded range |
| MaxAbsScaler | x / \|max\| | [-1, 1] | No | Sparse data with many zeros |
| RobustScaler | (x - median) / IQR | Varies | **Yes** | Data with significant outliers |

---

## 2. Encoding Categorical Data

ML algorithms expect numbers. Categorical data (strings) must be converted to numbers. This is called encoding.

Three main types:
- **Ordinal Encoding** — for ordinal categorical data (has a natural order)
- **Label Encoding** — same as ordinal but for the output/target column
- **One Hot Encoding** — for nominal categorical data (no natural order)

---

### Ordinal Encoding

Used when categories have a meaningful order. We specify the order manually.

Example: Poor < Average < Good → 0, 1, 2

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[
    ['Poor', 'Average', 'Good'],    # order for review column
    ['School', 'UG', 'PG']         # order for education column
])

oe.fit(X_train)
X_train = oe.transform(X_train)
X_test = oe.transform(X_test)
```

**Important:** Always specify categories explicitly with the correct order. If you don't, OrdinalEncoder assigns order alphabetically — which may not reflect the true ordering.

---

### Label Encoding

Same concept as ordinal encoding but used specifically for the **target/output column**.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()  # Order is decided automatically — alphabetical
le.fit(y_train)

le.classes_  # Check which class got what number

y_train = le.transform(y_train)
y_test = le.transform(y_test)
```

**Why only for target column?**

If you use LabelEncoder on input features:
- City column might get: Delhi=0, Mumbai=1, Chennai=2
- The model thinks Mumbai is "between" Delhi and Chennai
- And Chennai is "twice" Mumbai
- That ordering is completely meaningless — you've introduced false information into the model

For nominal input features → always OneHotEncoder.
For ordinal input features → OrdinalEncoder with explicit order.
For target column → LabelEncoder is fine.

---

### One Hot Encoding

For nominal categorical data — categories with no natural order (colours, cities, gender).

It creates separate binary columns for each category.

Example: Red, Green, Blue →
- Red = [1, 0, 0]
- Green = [0, 1, 0]
- Blue = [0, 0, 1]

#### The Dummy Variable Trap

With n categories, OHE creates n columns. But this causes **multicollinearity** — if Red=0 and Green=0, we already know it's Blue. So the Blue column is perfectly predictable from Red and Green — perfect linear dependency.

This makes XᵀX non-invertible → OLS breaks down.

**Fix:** Drop one column (drop='first'). We're left with n-1 columns — no redundancy.

#### Method 1 — Pandas get_dummies (NOT recommended for ML)

```python
# Basic OHE
pd.get_dummies(df, columns=['fuel', 'owner'])

# K-1 encoding (drop first column to avoid multicollinearity)
pd.get_dummies(df, columns=['fuel', 'owner'], drop_first=True)
```

**Why not recommended:** It doesn't remember which position it put the columns in. You can get different column orders between train and test. This breaks your model silently.

#### Method 2 — Sklearn OneHotEncoder (recommended)

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' handles dummy variable trap
ohe.fit(X_train[['fuel', 'owner']])

X_train_encoded = ohe.transform(X_train[['fuel', 'owner']])
X_test_encoded = ohe.transform(X_test[['fuel', 'owner']])  # uses same encoding as training
```

Sklearn remembers the category order and always produces consistent encoding.

---

### Handling High Cardinality — Dimensionality Reduction for Categories

Sometimes a column has many categories but some appear very rarely. A model can't learn from 3 examples.

**Approach 1 — Top Categories (keep most frequent)**

```python
# Keep top categories, group the rest as 'Others'
counts = df['brand'].value_counts()
threshold = 100

repl = counts[counts <= threshold].index  # categories with fewer than 100 occurrences

pd.get_dummies(df['brand'].replace(repl, 'Others'))
```

**Approach 2 — Target Encoding (replace category with mean of target)**

```python
import category_encoders as ce
encoder = ce.TargetEncoder(cols=['brand'])
# Warning: can cause data leakage without cross-validation
```

**Approach 3 — Frequency Encoding (replace category with its frequency)**

```python
freq = df['brand'].value_counts(normalize=True)
df['brand_encoded'] = df['brand'].map(freq)
# No leakage risk, preserves information about category prevalence
```

---

## 3. Column Transformer

When you have different columns requiring different transformations, applying them separately creates multiple numpy arrays that are hard to manage.

**Column Transformer** applies different transformations to different columns simultaneously and combines results into one clean output.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

transformer = ColumnTransformer(
    transformers=[   # list of tuples: (name, transformer, columns)
        ('impute_fever', SimpleImputer(), ['fever']),
        ('encode_cough', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
        ('encode_nominal', OneHotEncoder(sparse_output=False, drop='first'), ['gender', 'city'])
    ],
    remainder='passthrough'  # what to do with untouched columns
    # remainder='drop'       → drop untouched columns entirely
    # remainder='passthrough' → keep untouched columns as-is
)

# Fit and transform training data
X_train_transformed = transformer.fit_transform(X_train)

# Transform test data using same fitted transformer
X_test_transformed = transformer.transform(X_test)
```

**Important:** Transformed columns come first in the output, remainder columns come last. Column order changes after ColumnTransformer — keep this in mind when interpreting results.

---

## The Big Picture — Where Everything Fits

```
Raw Data
    │
    ├── Missing Values → Imputation (SimpleImputer, KNN, MICE)
    │
    ├── Categorical → Encoding (Ordinal, OHE, Target)
    │
    ├── Numerical → Scaling (Standard, MinMax, Robust)
    │
    ├── Outliers → Detection + Removal/Capping
    │
    ├── Distribution → Transformation (Log, Box-Cox, Yeo-Johnson)
    │
    ├── New Features → Construction + Splitting
    │
    └── Dimensionality → PCA, Feature Selection
```

All of this gets wrapped in a **Pipeline** — so transformations apply consistently to train and test without manual steps. That's why pipelines matter.

---

## Quick Reference — When to Use What

| Situation | Use |
|---|---|
| Features have different scales, no outliers | StandardScaler |
| Need values between 0 and 1, no outliers | MinMaxScaler |
| Data has significant outliers | RobustScaler |
| Sparse data (many zeros) | MaxAbsScaler |
| Ordinal categories (has natural order) | OrdinalEncoder |
| Target column with string labels | LabelEncoder |
| Nominal categories (no natural order) | OneHotEncoder |
| Many rare categories | Top-N encoding or Frequency encoding |
| Multiple different transformations | ColumnTransformer |
