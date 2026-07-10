# Random Forest — Complete Notes

---

## What is Random Forest?

Random Forest is an ensemble learning method that builds a large number of decision trees and combines their outputs for more accurate, stable predictions.

**Breaking down the name:**
- **Random** — Random Forest is a bagging-based technique. Bagging involves bootstrapping (random sampling of data) for each base model — hence "Random"
- **Forest** — A collection of many decision trees

**Key distinction from generic bagging:** In bagging, base models can be ANY algorithm (KNN, SVM, decision tree). Random Forest specifically uses **decision trees** as the base model, AND adds an extra layer of randomness (feature sampling at every node — explained below).

---

## How Random Forest Works

### Setup Example
1000 rows, 100 decision tree base models.

### Step 1 — Bootstrapping (Creating Diverse Data for Each Tree)

Three sampling strategies:

**Row Sampling**
Randomly select a subset of rows (e.g., 500 out of 1000) for each tree.
- **With replacement** — a row can be picked multiple times (standard bagging approach)
- **Without replacement** — no duplicates within a single sample

**Column Sampling (Feature Sampling)**
Randomly select a subset of columns (e.g., 5 out of total features) for each tree.
- **With replacement** — a feature can be selected multiple times
- **Without replacement** — no duplicate features within a sample

**Combination Sampling**
Sample both rows AND columns simultaneously — the standard Random Forest approach.

Since every tree sees different data (different rows and/or columns), each tree turns out structurally different — creating the diversity needed for the ensemble to work.

### Step 2 — Aggregation

Same as standard bagging:
- **Classification** → majority vote across all trees
- **Regression** → average of all trees' predictions

---

## Why Random Forest Performs So Well

Same underlying mechanism as bagging: decision trees (especially deep/unpruned ones) have **low bias but high variance**. Random Forest reduces this variance by training many decorrelated trees and averaging their predictions — the random fluctuations cancel out while the true signal (low bias) is preserved.

**The added randomness (feature sampling) makes trees even MORE decorrelated than plain bagging** — which is precisely why Random Forest typically outperforms plain Bagging with decision trees.

---

## Random Forest vs Bagging — The Critical Difference

This is the single most important distinction to understand.

**Bagging (with decision trees as base learners):**
Feature sampling happens **once, before the tree is built** (tree-level sampling).

Example: Tree D1 is assigned columns [1, 3] before training starts. The ENTIRE tree D1 — every single split at every node — can only use columns 1 and 3. No other columns are ever considered for any split in this tree.

**Random Forest:**
Feature sampling happens **at every single node**, fresh each time (node-level sampling).

Example: At the root node, a random subset of features (say 2 out of 5) is selected, and the best split is found among just those. At the next node (child), a NEW random subset of 2 features is selected — possibly different from the root's subset. This repeats at every node throughout the tree.

### Visual Comparison

```
BAGGING (tree-level sampling):
Tree D1 assigned columns [1,3] ONCE
    Root: split using col 1 or col 3
        Node A: split using col 1 or col 3 (same restricted set)
            Node B: split using col 1 or col 3 (same restricted set)

RANDOM FOREST (node-level sampling):
Tree D1 — fresh sampling at EVERY node
    Root: randomly pick 2 cols (say [1,3]) → split using best of these
        Node A: randomly pick 2 cols AGAIN (say [2,4]) → split using best of these
            Node B: randomly pick 2 cols AGAIN (say [1,5]) → split using best of these
```

### Why This Matters

Node-level sampling injects **more randomness** into each tree's structure:
- Creates more diverse trees across the forest
- Trees become less correlated with each other
- Less correlation → greater variance reduction when aggregating (recall: Var(average) = ρσ² + (1-ρ)σ²/B — lower ρ means lower variance)
- Result: better generalisation, less overfitting compared to plain Bagging

**The key hyperparameter controlling this:** `max_features` — the number of features randomly sampled at each split.

---

## Random Forest Hyperparameters

### Tree-Level Hyperparameters (inherited from Decision Trees)

| Parameter | Meaning |
|---|---|
| `max_depth` | Maximum depth of each individual tree |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required at a leaf |
| `criterion` | 'gini' or 'entropy' for classification; 'squared_error' etc. for regression |

### Forest-Level Hyperparameters (specific to Random Forest)

| Parameter | Meaning | Typical Values |
|---|---|---|
| `n_estimators` | Number of trees in the forest | 100-1000 |
| `max_features` | Number of features sampled at each split | 'sqrt' (classification), 'log2', or a fraction |
| `bootstrap` | Whether to bootstrap samples (True = bagging applied) | True (default) |
| `max_samples` | Fraction of samples per bootstrap (when bootstrap=True) | 1.0 (default) or less |
| `oob_score` | Calculate out-of-bag score | True/False |
| `n_jobs` | Number of CPU cores to use (-1 = all) | -1 for speed |

### Default max_features Values

- **Classification:** `max_features='sqrt'` → √(total features)
- **Regression:** `max_features=1.0` (all features) — though tuning this often helps

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=2,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
```

---

## Hyperparameter Tuning — GridSearchCV vs RandomizedSearchCV

### GridSearchCV — Exhaustive Search

Tries **every single combination** of specified hyperparameters.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

**Total combinations here:** 3 × 4 × 2 × 3 = 72 combinations × 5 folds = 360 model fits. Exhaustive but computationally expensive — grows exponentially with more hyperparameters.

### RandomizedSearchCV — Random Sampling

Instead of trying every combination, randomly samples a fixed number of combinations from the specified distributions.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [5, 10, 15, 20, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,         # only try 50 random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

### GridSearchCV vs RandomizedSearchCV — When to Use Which

| | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| **Search method** | Exhaustive — every combination | Random sampling of combinations |
| **Computational cost** | High (exponential growth) | Controlled (set by n_iter) |
| **Best for** | Few hyperparameters, small ranges | Many hyperparameters, large ranges |
| **Guarantee** | Finds the best within the grid | May miss the absolute best, but usually close |
| **Practical use** | Final fine-tuning on narrowed range | Initial broad search |

**Best practice:** Use RandomizedSearchCV first to find a promising region, then use GridSearchCV to fine-tune within that narrower region.

---

## OOB Score — Out-of-Bag Evaluation

Since each tree is trained on a bootstrap sample (~63.2% of data), the remaining ~36.8% of rows (OOB samples) were never seen by that tree.

**How OOB score works:**
For each training point, identify which trees did NOT include it in their bootstrap sample. Use only those trees to predict that point. Compare predictions to actual labels.

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_:.4f}")
```

**Why OOB score matters:**
- Provides a validation estimate without needing a separate validation set
- Especially valuable with smaller datasets where you want to use all data for training
- Generally correlates well with test set performance — a useful sanity check before evaluating on the actual test set

---

## Feature Importance in Random Forest

### How It's Calculated

For each feature, sum up the **impurity reduction** (Gini decrease or entropy decrease) it contributes across all splits, across all trees in the forest. Normalise so all importances sum to 1.

> Importance(feature) = (1/n_trees) × Σ_trees Σ_nodes_using_feature [weighted impurity decrease at that node]

**Intuition:** A feature that consistently creates large impurity reductions across many trees and many nodes is considered highly important.

```python
import pandas as pd
import matplotlib.pyplot as plt

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

print(importances)

importances.plot(kind='barh')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.show()
```

### Why Random Forest Feature Importance is More Reliable Than a Single Decision Tree

A single decision tree's feature importance can be unstable — small data changes can completely change which features appear important (because the tree structure itself is unstable, as we discussed in bagging).

Random Forest **averages feature importance across hundreds of trees**, each trained on different bootstrap samples and different feature subsets. This averaging makes the importance estimates much more stable and reliable.

### Important Caveat — Bias Toward High Cardinality Features

Like single decision trees, Random Forest feature importance is still biased toward features with many unique values (more potential split points = more chances to appear important) and toward features at the top of trees.

**Better alternative for unbiased importance:** **Permutation Importance**

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

perm_importances = pd.Series(result.importances_mean, index=feature_names)
perm_importances.sort_values(ascending=False)
```

**How permutation importance works:** Shuffle one feature's values (breaking its relationship with the target) and measure how much model performance drops. Larger performance drop = more important feature. This method doesn't have the high-cardinality bias of impurity-based importance.

---

## Complete Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=2,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")
print(classification_report(y_test, y_pred))

# Cross-validation for robustness check
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(importances)
```

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    max_features=1.0,  # often all features for regression
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
```

---

## Random Forest vs Single Decision Tree vs Bagging — Full Comparison

| | Single Decision Tree | Bagging (with trees) | Random Forest |
|---|---|---|---|
| **Number of models** | 1 | Many | Many |
| **Row sampling** | No (uses all data) | Yes (bootstrap) | Yes (bootstrap) |
| **Feature sampling** | No | No (tree-level only if specified) | Yes — node-level, every split |
| **Variance** | High | Reduced | Further reduced |
| **Bias** | Low | Low (preserved) | Low (preserved) |
| **Overfitting risk** | High | Moderate | Lower |
| **Interpretability** | High | Low | Low |
| **Training speed** | Fast | Moderate | Moderate (parallelisable) |

---

## Interview One-Liners

**What is Random Forest?**
"A bagging ensemble that uses decision trees as base learners, with an added layer of randomness — feature sampling at every node split, not just at the tree level. This extra randomness decorrelates trees more than plain bagging, leading to better variance reduction."

**Random Forest vs Bagging — what's the actual difference?**
"In bagging with decision trees, each tree gets a fixed subset of features chosen ONCE before training. In Random Forest, a fresh random subset of features is chosen at EVERY node split. This node-level sampling creates more diverse, less correlated trees, leading to better generalisation."

**How is feature importance calculated in Random Forest?**
"Sum the impurity reduction each feature contributes across all splits in all trees, then average and normalise. More reliable than single-tree importance because it's averaged across many trees, but still biased toward high-cardinality features — permutation importance is a more unbiased alternative."

**What is OOB score?**
"Since bootstrap samples use ~63.2% of data, the remaining ~36.8% per tree can validate that tree's predictions — giving essentially free validation without a separate holdout set."

**GridSearchCV vs RandomizedSearchCV?**
"GridSearchCV exhaustively tries every combination — thorough but expensive, especially with many hyperparameters. RandomizedSearchCV samples a fixed number of random combinations — faster, good for initial broad search. Common practice: use RandomizedSearchCV first, then GridSearchCV to fine-tune."
