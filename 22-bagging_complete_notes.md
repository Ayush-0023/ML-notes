# Bagging (Bootstrap Aggregating) — Complete Notes

---

## What is Bagging?

Bagging = **B**ootstrap **Agg**regat**ing**.

An ensemble technique that improves stability and accuracy by combining multiple versions of the **same model**, each trained on a slightly different (randomly sampled) dataset.

**Two components:**

1. **Bootstrapping** — Create multiple random subsets of the original dataset using sampling **with replacement**. Each subset is called a bootstrap sample.
2. **Aggregation** — Combine predictions: majority voting for classification, averaging for regression.

---

## Bootstrapping in Detail

**Sampling with replacement** means: when you pick a row, you put it back before picking the next one. So the same row can be picked multiple times in a single bootstrap sample.

**Example:** Original dataset has 1000 rows. To create a bootstrap sample of size 1000:
- Randomly pick a row, record it, put it back
- Repeat 1000 times

**Mathematical property:** On average, a bootstrap sample contains about **63.2%** of the unique rows from the original dataset (some rows appear multiple times, others not at all).

> Probability a specific row is NOT selected in one draw = (1 - 1/n)
> Probability it's NOT selected in n draws = (1 - 1/n)ⁿ ≈ e⁻¹ ≈ 0.368 (for large n)
> Probability it IS selected at least once = 1 - 0.368 = **0.632**

The remaining ~36.8% of rows not included in a bootstrap sample are called **Out-of-Bag (OOB)** samples — useful for validation without needing a separate test set.

---

## Why Bagging Improves Performance — The Bias-Variance Connection

### The Setup

Recall the bias-variance tradeoff:
- **High bias, low variance** → underfitting (model too simple)
- **Low bias, high variance** → overfitting (model too complex, e.g., fully grown decision tree)
- **Ideal:** low bias AND low variance

**Bagging specifically targets models with low bias but high variance** — like fully grown (unpruned) decision trees.

### How It Works

High variance models are very sensitive to small changes in training data — even minor changes can drastically alter predictions.

**Bagging's mechanism:**
1. Train multiple high-variance models on different bootstrap samples
2. Since each model sees different data, their errors become **less correlated**
3. When you average/vote across models, random fluctuations (variance) cancel out
4. The underlying true pattern (which all models still roughly capture — low bias preserved) remains

### Concrete Example

Train ONE deep decision tree on 10,000 rows. Change just 100 rows → the entire tree structure can change drastically (high variance — sensitive to data).

Train MANY deep decision trees on different bootstrap samples of the same 10,000 rows. Changing 100 rows now only affects *some* of the trees, not all of them. The aggregate (averaged/voted) prediction remains stable.

### The Mathematical Intuition

If you have B models, each with variance σ², and their errors are **independent**:

> Variance of average = σ²/B

If models are **correlated** with correlation ρ:

> Variance of average = ρσ² + (1-ρ)σ²/B

**Key insight:** Even with some correlation, increasing B reduces variance (the second term shrinks). This is exactly why bagging works — more diverse bootstrap samples → lower correlation (ρ) → more variance reduction.

**Result:** Bagging significantly reduces variance **without increasing bias** — leading to more stable, accurate predictions on unseen data.

---

## Bagging Classifier

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),  # base model (default is decision tree)
    n_estimators=100,        # number of base models
    max_samples=1.0,         # fraction of samples per bootstrap (1.0 = same size as original)
    max_features=1.0,        # fraction of features per bootstrap
    bootstrap=True,          # sampling with replacement (True = bagging)
    bootstrap_features=False,# whether to bootstrap features too
    oob_score=True,          # calculate out-of-bag score
    n_jobs=-1,                # use all CPU cores
    random_state=42
)

bagging_clf.fit(X_train, y_train)
y_pred = bagging_clf.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score: {bagging_clf.oob_score_:.4f}")  # validation without separate test set
```

**Comparing single tree vs bagged trees:**

```python
# Single decision tree (high variance)
single_tree = DecisionTreeClassifier()
single_tree.fit(X_train, y_train)
print(f"Single Tree: {accuracy_score(y_test, single_tree.predict(X_test)):.4f}")

# Bagged trees (lower variance)
print(f"Bagged Trees: {accuracy_score(y_test, bagging_clf.predict(X_test)):.4f}")
# Bagged version typically performs better and more consistently across runs
```

---

## Bagging Regressor

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

bagging_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    max_samples=0.8,    # use 80% of data per bootstrap sample
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

bagging_reg.fit(X_train, y_train)
y_pred = bagging_reg.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"OOB Score (R²): {bagging_reg.oob_score_:.4f}")
```

---

## Key Hyperparameters

| Parameter | Meaning | Effect |
|---|---|---|
| `n_estimators` | Number of base models | More models → lower variance, more compute |
| `max_samples` | Fraction/count of samples per bootstrap | Lower → more diversity, possibly more bias |
| `max_features` | Fraction/count of features per bootstrap | Lower → more diversity (this becomes Random Forest's key idea) |
| `bootstrap` | Sample with replacement? | True = bagging. False = no resampling |
| `bootstrap_features` | Bootstrap features too? | Adds another layer of randomness |
| `oob_score` | Calculate out-of-bag validation score | Free validation without separate test set |

---

## Out-of-Bag (OOB) Score — Free Validation

Since each bootstrap sample uses ~63.2% of data, the remaining ~36.8% (OOB samples) were never seen by that particular model during training.

**This lets you validate without a separate validation set:**
- For each training point, find all models that did NOT include it in their bootstrap sample
- Use those models to predict that point
- Compare to actual value → OOB score

```python
print(f"OOB Score: {bagging_clf.oob_score_:.4f}")
# This is essentially a built-in cross-validation, computed almost for free
```

**Why this matters:** With small datasets, you don't want to "waste" data on a separate validation set. OOB gives you validation while using all data for training.

---

## Bagging Works Best With High-Variance Base Models

| Base Model | Bias | Variance | Bagging Benefit |
|---|---|---|---|
| Deep Decision Tree | Low | High | ✓✓✓ Excellent — this is the classic use case |
| Shallow Decision Tree | High | Low | ✗ Limited — bias remains, little to reduce |
| Linear Regression | Often high | Often low | ✗ Limited benefit |
| KNN (small K) | Low | High | ✓✓ Good |
| KNN (large K) | High | Low | ✗ Limited |

**This is exactly why Random Forest (Bagging + Decision Trees) is so effective** — decision trees are naturally high-variance, making them the perfect base learner for bagging.

---

## Bagging vs Voting — Key Difference

| | Voting | Bagging |
|---|---|---|
| **Base models** | Different algorithms | Same algorithm |
| **Training data** | Same dataset for all | Different bootstrap samples |
| **Goal** | Combine diverse perspectives | Reduce variance of one algorithm |
| **Diversity source** | Algorithm choice | Data sampling |

---

## Interview One-Liners

**What is bagging?**
"Bootstrap Aggregating — train multiple copies of the same model on different bootstrap samples (sampled with replacement) of the training data, then combine predictions via voting or averaging. Reduces variance without increasing bias."

**Why does bagging reduce variance?**
"High-variance models are sensitive to small changes in training data. Training on different bootstrap samples decorrelates the models' errors. Averaging decorrelated errors cancels out the random fluctuations while preserving the true underlying pattern."

**What is OOB score?**
"Since each bootstrap sample uses only ~63.2% of data on average, the remaining ~36.8% (out-of-bag samples) can be used to validate that model — giving a free validation estimate without a separate holdout set."

**Why are decision trees the ideal base learner for bagging?**
"Fully grown decision trees have low bias but high variance — exactly the profile bagging is designed to fix. This combination (bagging + decision trees) is the foundation of Random Forest."

**What's the math behind why more bootstrap samples helps?**
"Each row has probability (1-1/n)ⁿ ≈ 36.8% of being excluded from a given bootstrap sample. With independent models, variance of the averaged prediction shrinks as 1/B (B = number of models). Even with correlated models, increasing B still reduces variance — just not all the way to zero."
