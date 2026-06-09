# Feature Engineering — Part 3: Outliers, Curse of Dimensionality & PCA

---

## 1. Outliers

### What is an Outlier?

A data point that is significantly different from the rest of the data. Outliers can arise from:
- Measurement errors — faulty sensors, data entry mistakes
- Natural variation — genuine extreme values (a billionaire in an income dataset)
- Data corruption — wrong units, wrong scale

**Why outliers matter:**
- Mean and std are heavily affected by outliers — making StandardScaler unreliable
- Linear regression tries to minimise squared errors — one outlier with huge error pulls the line toward it
- Distance-based algorithms (KNN, K-Means) are heavily distorted by outliers
- Decision trees are relatively robust to outliers

---

### How to Detect Outliers

Choice of detection method depends on the distribution of your data:

| Distribution | Method |
|---|---|
| Normal | Z-score method |
| Skewed | IQR method |
| Unknown | Percentile / Winsorization |

Always visualise first with a boxplot or histogram before applying any method.

---

### Method 1 — Z-Score Method (Normal Distribution)

Any data point beyond ±3 standard deviations from the mean is considered an outlier.

**Why ±3σ?** In a normal distribution, 99.7% of data falls within ±3σ. Anything beyond is in the extreme 0.3% — almost certainly an outlier.

```python
import numpy as np
import pandas as pd

# Find boundary values
upper_limit = df['cgpa'].mean() + 3 * df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3 * df['cgpa'].std()

print(f"Upper limit: {upper_limit:.2f}")
print(f"Lower limit: {lower_limit:.2f}")

# Find outliers
outliers = df[(df['cgpa'] > upper_limit) | (df['cgpa'] < lower_limit)]
print(f"Number of outliers: {len(outliers)}")

# Trimming — remove outliers completely
df_trimmed = df[(df['cgpa'] < upper_limit) & (df['cgpa'] > lower_limit)]

# OR using Z-score directly
df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean()) / df['cgpa'].std()
df_trimmed = df[(df['cgpa_zscore'] < 3) & (df['cgpa_zscore'] > -3)]

# Capping — replace outliers with boundary values (Winsorization)
df['cgpa'] = np.where(
    df['cgpa'] > upper_limit,
    upper_limit,
    np.where(
        df['cgpa'] < lower_limit,
        lower_limit,
        df['cgpa']
    )
)
```

**Limitation:** Only valid for normally distributed data. Applying Z-score to skewed data gives wrong boundaries because mean and std are not meaningful measures for skewed distributions.

---

### Method 2 — IQR Method (Skewed Distribution)

Uses quartiles instead of mean/std — robust to skewness because quartiles don't get pulled by extreme values.

**Rule:** Any point beyond Q3 + 1.5×IQR or below Q1 - 1.5×IQR is an outlier.

**Why 1.5?** John Tukey (inventor of the boxplot) chose 1.5 empirically — it captures roughly 99.3% of normally distributed data. It's a convention, not a mathematical law.

```python
import seaborn as sns

# Visualise first
sns.boxplot(df['placement_exam_marks'])

# Calculate IQR
Q1 = df['placement_exam_marks'].quantile(0.25)
Q3 = df['placement_exam_marks'].quantile(0.75)
IQR = Q3 - Q1

# Boundaries
upper_limit = Q3 + 1.5 * IQR
lower_limit = Q1 - 1.5 * IQR

print(f"Upper limit: {upper_limit:.2f}")
print(f"Lower limit: {lower_limit:.2f}")

# Find outliers
upper_outliers = df[df['placement_exam_marks'] > upper_limit]
lower_outliers = df[df['placement_exam_marks'] < lower_limit]
print(f"Upper outliers: {len(upper_outliers)}, Lower outliers: {len(lower_outliers)}")

# Trimming
df_trimmed = df[
    (df['placement_exam_marks'] < upper_limit) &
    (df['placement_exam_marks'] > lower_limit)
]

# Capping
df_capped = df.copy()
df_capped['placement_exam_marks'] = np.where(
    df_capped['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        df_capped['placement_exam_marks'] < lower_limit,
        lower_limit,
        df_capped['placement_exam_marks']
    )
)
```

---

### Method 3 — Percentile Method / Winsorization

Instead of using a statistical rule, you simply cap values at chosen percentiles — typically 1st and 99th.

**When to use:** When distribution is neither normal nor clearly skewed, or when you want full control over the boundaries.

```python
# Cap at 1st and 99th percentile
lower_limit = df['column'].quantile(0.01)
upper_limit = df['column'].quantile(0.99)

df['column'] = np.where(df['column'] > upper_limit, upper_limit,
               np.where(df['column'] < lower_limit, lower_limit,
               df['column']))

# Using scipy directly
from scipy.stats.mstats import winsorize
df['column_winsorized'] = winsorize(df['column'], limits=[0.01, 0.01])
# limits=[0.01, 0.01] → cap bottom 1% and top 1%
```

**Why "Winsorization"?** Named after statistician Charles Winsor. Instead of removing extreme values, it replaces them with the nearest boundary value — preserving the row count while reducing extreme influence.

---

### Trimming vs Capping — When to Use Which

| | Trimming | Capping (Winsorization) |
|---|---|---|
| **What it does** | Removes outlier rows entirely | Replaces outlier values with boundary values |
| **Effect on dataset size** | Reduces rows | Keeps all rows |
| **When to use** | Outliers are genuine errors, dataset is large enough | Outliers might be real, can't afford to lose rows |
| **Risk** | Losing potentially valid data | Distorting the distribution at boundaries |

---

### Outlier Treatment Decision Guide

```
Outlier detected
        │
        ├── Is it a data entry error?
        │     └── Yes → Trim (remove it)
        │
        ├── Is it a genuine extreme value?
        │     ├── Using linear model → Cap (it pulls the line)
        │     ├── Using tree model → Leave it (trees are robust)
        │     └── Using distance model → Cap (distorts distances)
        │
        └── Unsure?
              └── Try both, compare model performance
```

---

## 2. Curse of Dimensionality

### What is it?

As you add more features (dimensions), the data becomes increasingly sparse in that high-dimensional space. The same number of data points covers less and less of the feature space.

**The core problem:** There is an optimal number of features for maximum model performance. Adding features beyond that point doesn't improve performance — it makes it worse.

### Why Does Sparsity Hurt?

- **Distance-based algorithms** (KNN, K-Means, SVM) break down — in high dimensions, all points become roughly equidistant from each other. "Nearest neighbour" loses meaning.
- **More features → more parameters → more data needed** to learn reliably. With limited data, the model overfits.
- **Irrelevant features** add noise — the model tries to learn patterns from random variation.

### Visualising the Problem

In 1D: 10 points cover a line reasonably well.
In 2D: you need 100 points (10²) to cover a grid with the same density.
In 3D: you need 1000 points (10³).
In 100D: you need 10¹⁰⁰ points — more than atoms in the universe.

### Solution — Dimensionality Reduction

Two approaches:

**1. Feature Selection** — Keep a subset of existing features. The features remain interpretable.
- Forward Selection — start with no features, add one at a time (the one that improves performance most)
- Backward Elimination — start with all features, remove one at a time (the one that hurts performance least)
- Embedded methods — Lasso (zeroes out irrelevant features automatically)

**2. Feature Extraction** — Create a new, smaller set of features from existing ones. New features are combinations of originals — less interpretable but more powerful.
- PCA — Principal Component Analysis
- LDA — Linear Discriminant Analysis
- t-SNE — for visualisation only
- UMAP — modern alternative to t-SNE

---

## 3. Principal Component Analysis (PCA)

### What is PCA?

PCA is a feature extraction technique that reduces the number of features while preserving as much information (variance) as possible.

It transforms your original features into a new set of features called **Principal Components** — ordered by how much variance they explain.

### The Core Idea — Variance = Information

PCA assumes that the directions of maximum variance in your data are the most informative directions.

Think of it like this: if all your data points vary a lot along one direction, that direction carries a lot of information. A direction where data barely varies carries little information and can be discarded.

### Geometric Intuition

Imagine your data as a cloud of points in 2D:

```
        ●  ●
      ●   ●  ●
    ●  ●   ●  ●      → data spreads most in this diagonal direction
      ●  ●  ●
        ●  ●
```

PCA finds the axis of maximum spread (PC1), then the axis perpendicular to it (PC2), then PC3 perpendicular to both, and so on.

You keep the first k components that explain most of the variance and discard the rest.

### Step-by-Step Mathematical Process

**Step 1 — Standardise the data**
PCA is variance-based. Features with larger scales dominate. Always standardise first.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Step 2 — Compute the Covariance Matrix**
The covariance matrix captures how each feature varies with every other feature.

> Cov(X) = (1/n) XᵀX

**Step 3 — Compute Eigenvalues and Eigenvectors**
- Eigenvectors = the directions of the principal components
- Eigenvalues = how much variance each component explains

Larger eigenvalue → more variance → more important component.

**Step 4 — Sort by Eigenvalue**
Sort components in descending order of eigenvalue. PC1 explains most variance, PC2 second most, and so on.

**Step 5 — Select k Components**
Choose k based on how much variance you want to retain (typically 95%).

**Step 6 — Transform the Data**
Project original data onto the k principal components.

### Implementation

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Step 1 — Always scale first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Step 2 — Fit PCA (keep all components first to analyse)
pca = PCA()
pca.fit(X_scaled)

# Step 3 — Check explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Variance explained by each component:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.3f} ({cumulative_variance[i]:.3f} cumulative)")

# Step 4 — Scree Plot (visualise explained variance)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance)+1), explained_variance)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance')
plt.legend()
plt.show()

# Step 5 — Choose k components that explain 95% variance
pca = PCA(n_components=0.95)  # automatically selects k for 95% variance
# OR
pca = PCA(n_components=10)    # explicitly choose 10 components

X_train_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(scaler.transform(X_test))

print(f"Original shape: {X_train.shape}")
print(f"After PCA shape: {X_train_pca.shape}")
```

### The Scree Plot — How to Choose k

The scree plot shows variance explained vs number of components. Look for the "elbow" — the point where adding more components gives diminishing returns.

Common rule: choose k such that cumulative explained variance ≥ 95%.

### PCA in a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
print(f"Components selected: {pipe.named_steps['pca'].n_components_}")
```

### What PCA Does NOT Do

- PCA does not use the target variable — it's completely unsupervised. It finds directions of maximum variance, not directions most useful for prediction.
- PCA components are not interpretable — PC1 is a combination of all original features, not one specific feature.
- PCA does not remove noise automatically — it removes low-variance directions, which are often but not always noise.

### PCA vs LDA

| | PCA | LDA |
|---|---|---|
| **Type** | Unsupervised | Supervised |
| **Optimises for** | Maximum variance | Maximum class separation |
| **Uses target variable** | No | Yes |
| **Use when** | No labels, compression | Classification, want better separation |

LDA finds components that maximise the ratio of between-class variance to within-class variance. It knows about your target classes — PCA doesn't.

### t-SNE — For Visualisation Only

t-SNE (t-Distributed Stochastic Neighbour Embedding) is a dimensionality reduction technique used exclusively for visualisation — reducing to 2D or 3D to see cluster structure.

**Critical rule:** Never use t-SNE as a preprocessing step for ML. It's non-linear, non-deterministic, and not invertible. Use only to visualise high-dimensional data.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE Visualisation')
plt.show()
```

---

## Complete Feature Engineering Summary

```
Raw Data
    │
    ├── 1. Handle Missing Values
    │         → CCA, Mean/Median/Mode, Random, KNN, MICE
    │
    ├── 2. Handle Outliers
    │         → Detect (Z-score / IQR / Percentile)
    │         → Treat (Trim / Cap)
    │
    ├── 3. Encode Categorical Features
    │         → Ordinal Encoder, Label Encoder, One Hot Encoder
    │
    ├── 4. Transform Distributions
    │         → Log, Square, Sqrt, Box-Cox, Yeo-Johnson
    │
    ├── 5. Scale Features
    │         → Standard, MinMax, Robust, MaxAbs
    │
    ├── 6. Reduce Dimensionality
    │         → Feature Selection (Forward/Backward/Lasso)
    │         → Feature Extraction (PCA, LDA)
    │
    └── 7. Wrap Everything in a Pipeline
              → Consistent, leak-free, deployable
```
