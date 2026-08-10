# XGBoost — Complete Notes

---

## The Historical Context

**1970s-80s:** ML algorithms worked well only on specific data types and small datasets — narrow, specialised tools.

**1990s:** Random Forest, SVM, Gradient Boosting emerged — generalised well across many dataset types. But they suffered from **scalability** problems (slow on large data) and **overfitting** issues.

**2014 onward:** XGBoost (eXtreme Gradient Boosting) — currently among the best-performing algorithms for structured/tabular data.

**Critical distinction:** XGBoost is NOT a new algorithm. It's a **highly optimised library/implementation** built on top of the existing Gradient Boosting algorithm.

> **XGBoost = Gradient Boosting (the ML algorithm) + Software Engineering optimisations (the implementation)**

This is why XGBoost dominates Kaggle competitions and production systems — it takes a proven mathematical approach and makes it dramatically faster, more memory-efficient, and more robust to overfitting through careful engineering.

---

## XGBoost's Three Pillars

### 1. Performance (Accuracy & Generalisation)

- **Regularised learning objective** — penalises model complexity directly in the loss function (unlike vanilla Gradient Boosting which relies only on tree depth/learning rate for control)
- **Handling missing values** — automatically learns the best direction (left/right) to send missing values during training, no manual imputation needed
- **Sparsity-aware split finding** — efficiently skips zero/missing entries when finding splits, crucial for sparse data (e.g., one-hot encoded categorical features)
- **Efficient split finding** — uses weighted quantile sketch (approximate histogram-based method) + approximate tree learning for speed on large datasets
- **Tree pruning** — uses a "max_depth first, then prune backward" strategy (different from greedy stopping in vanilla GB) — prevents missing good splits that initially look bad but lead to better splits downstream

### 2. Speed

- **Parallel processing** — even though boosting is sequential at the tree level, the construction of each individual tree (split finding across features) is parallelised
- **Optimised data structures** — uses a compressed column-block structure for fast column access during split finding
- **Cache awareness** — algorithms designed to minimise cache misses on CPU
- **Out-of-core computing** — can handle datasets too large to fit in RAM by efficient disk usage
- **Distributed computing** — scales across multiple machines/clusters
- **GPU support** — can leverage GPU acceleration for training

### 3. Flexibility

- Cross-platform support (Windows, Linux, Mac)
- Multiple language support (Python, R, Java, Scala, C++)
- Integration with other ML tools and pipelines (sklearn-compatible API)
- Supports all types of ML problems — classification, regression, ranking, etc.
- Allows **custom loss functions** — you can define your own objective if the built-in ones don't fit your problem

---

## LightGBM — A Sibling Implementation

LightGBM is another optimised implementation of Gradient Boosting, similar in spirit to XGBoost but with different internal optimisations.

**Key differences from XGBoost:**
- **Leaf-wise tree growth** (vs XGBoost's level-wise/depth-wise growth) — LightGBM grows the leaf with the highest loss reduction first, which can lead to deeper, more asymmetric trees but often better accuracy
- **Targeted for:** faster training speed, higher efficiency, lower memory usage — especially valuable on very large datasets
- Uses histogram-based algorithms more aggressively than XGBoost's default settings

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1  # -1 means no limit, controlled by num_leaves instead
)
lgb_model.fit(X_train, y_train)
```

**When to choose LightGBM over XGBoost:** Very large datasets where training speed/memory is the bottleneck. XGBoost is often preferred for smaller-to-medium datasets where its more conservative depth-wise growth gives more stable, less overfit-prone trees.

---

## XGBoost for Regression

### The Overall Process (Same Skeleton as Gradient Boosting)

1. Start with initial prediction = mean of y
2. Calculate residuals
3. Build a tree to predict residuals
4. Update prediction = previous + learning_rate × new tree's prediction
5. Repeat

### The Key Difference — How Trees Are Built

In vanilla Gradient Boosting, the decision trees are built using standard criteria (Gini impurity or entropy/variance reduction) — exactly like a normal decision tree.

In XGBoost, trees are built using a special metric called **Similarity Score**:

> **Similarity Score = (Σ residuals)² / (n + λ)**

Where:
- Σ residuals = sum of residuals in that node
- n = number of residuals (samples) in that node
- λ (lambda) = regularisation parameter

### Why This Formula?

This is derived from a **second-order Taylor expansion** of the loss function around the current prediction (explained in the math section below). It allows XGBoost to evaluate the quality of a potential split WITHOUT actually building the full tree and checking error — making split-finding much faster and more theoretically grounded.

### Tree Building — Finding the Best Split

**Step 1 — Calculate the Similarity Score for the parent node** (before any split).

**Step 2 — For each candidate split point**, divide data into left and right children, and calculate:

> **Gain = Similarity(Left) + Similarity(Right) - Similarity(Parent)**

**Step 3 — Choose the split with the maximum Gain.**

This directly parallels Information Gain in standard decision trees — but using the Similarity Score formula instead of entropy/Gini.

**Split-finding algorithms:**
- **Exact Greedy Algorithm** — tries every possible split point. Works well for smaller datasets but is computationally expensive for large ones.
- **Approximate Algorithm** — uses percentile-based candidate split points (weighted quantile sketch) instead of every possible value. Much faster, used automatically for large datasets.

### Worked Numerical Example

Suppose a node has 4 residuals: [4, -2, 3, -1], λ = 1

> Σ residuals = 4 + (-2) + 3 + (-1) = 4
> n = 4
> Similarity = 4² / (4 + 1) = 16/5 = **3.2**

If we split this node into Left=[4,-2] and Right=[3,-1]:

> Similarity(Left) = (4-2)²/(2+1) = 4/3 ≈ 1.33
> Similarity(Right) = (3-1)²/(2+1) = 4/3 ≈ 1.33
> Gain = 1.33 + 1.33 - 3.2 = -0.54

A negative gain means this particular split made things worse — XGBoost would consider other split points and potentially **not split at all** if no split yields positive gain (this connects to the `gamma` hyperparameter — the minimum gain required to make a split).

### The Role of Lambda (λ)

Notice λ sits in the denominator of the Similarity Score. As λ increases:
- Similarity Score decreases for all nodes
- This makes it HARDER for any split to show meaningful gain
- Effectively, trees become simpler/shallower — **λ is a direct overfitting control knob**

This is analogous to L2 regularisation in linear regression — λ penalises complexity directly in the optimisation objective.

---

## XGBoost for Classification

### The Key Adaptation

Same overall structure as regression, but the base prediction works in **log-odds space** (exactly like Gradient Boosting for classification, and conceptually like logistic regression).

**Step 1 — Initial prediction:** log(odds) instead of mean.
> F₀(x) = log(p / (1-p))

**Step 2 — Convert to probability** via sigmoid for actual predictions:
> P(x) = 1 / (1 + e^(-F(x)))

**Step 3 — Calculate residuals** (actual label - predicted probability):
> residual = y - P(x)

**Step 4 — Build tree on residuals**, but now the Similarity Score formula changes:

> **Similarity Score = (Σ residuals)² / (Σ [prev_prob × (1 - prev_prob)] + λ)**

### Why the Denominator Changes

The denominator (Σ prev_prob × (1-prev_prob)) comes from the **second derivative (Hessian)** of the log-loss function — it represents how "confident" the current model already is at each point.

**Intuition:** prev_prob × (1-prev_prob) is maximised at prev_prob=0.5 (maximum uncertainty) and minimised near 0 or 1 (high confidence). This means:
- Points the model is already confident about contribute LESS to the denominator → don't destabilise the similarity score much
- Points the model is uncertain about (prob near 0.5) contribute MORE to the denominator → similarity score is more conservative for these, requiring stronger evidence to justify a split

This naturally weights the tree-building process by how much "work" still needs to be done on each point.

---

## Mathematics Behind XGBoost — The Full Derivation

### The Objective Function

XGBoost minimises a regularised objective:

> **Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)**

Where:
- L(yᵢ, ŷᵢ) is the loss function (e.g., squared error for regression, log-loss for classification)
- Ω(fₖ) is a regularisation term for each tree fₖ, penalising complexity:

> **Ω(f) = γT + (1/2)λΣⱼwⱼ²**

Where T = number of leaves, wⱼ = weight (output value) of leaf j, γ and λ are regularisation hyperparameters.

### Second-Order Taylor Approximation

For each boosting round, instead of computing the exact loss reduction (which requires evaluating the loss for every possible tree structure — computationally infeasible), XGBoost approximates the loss using a **second-order Taylor expansion** around the current prediction:

> L(yᵢ, ŷᵢ^(t-1) + fₜ(xᵢ)) ≈ L(yᵢ, ŷᵢ^(t-1)) + gᵢfₜ(xᵢ) + (1/2)hᵢfₜ(xᵢ)²

Where:
- **gᵢ = ∂L/∂ŷ** (first derivative / gradient) — this is the "residual-like" quantity
- **hᵢ = ∂²L/∂ŷ²** (second derivative / Hessian) — this captures curvature, i.e., confidence

**This is XGBoost's key mathematical innovation over standard Gradient Boosting** — using BOTH first and second derivatives (Newton's method style optimisation) rather than just the first derivative (gradient descent style) used in vanilla Gradient Boosting. This generally leads to faster, more accurate convergence.

### Deriving the Similarity Score from This

For a fixed tree structure, the optimal weight for leaf j (minimising the Taylor-approximated objective) is:

> **wⱼ* = -Gⱼ / (Hⱼ + λ)**

Where Gⱼ = Σ(gᵢ for i in leaf j) and Hⱼ = Σ(hᵢ for i in leaf j).

Plugging this back into the objective gives the optimal loss reduction for that leaf structure:

> **Obj* = -(1/2) Σⱼ (Gⱼ² / (Hⱼ + λ)) + γT**

The term **Gⱼ²/(Hⱼ+λ)** is exactly the Similarity Score formula (up to a sign and constant factor)!

**For squared error loss (regression):** gᵢ = -(residual), hᵢ = 1 (constant) → Similarity Score = (Σresiduals)²/(n+λ) — matching exactly what was used in the regression section above (n appears because hᵢ=1 for every point, so Hⱼ = n).

**For log-loss (classification):** gᵢ = (P(x)-y), hᵢ = P(x)(1-P(x)) → Similarity Score = (Σresiduals)²/(Σ[P(1-P)] + λ) — matching the classification formula above.

**This is the unifying mathematical insight:** the Similarity Score formula isn't arbitrary — it falls directly out of optimally solving the second-order Taylor-approximated objective function, for ANY differentiable loss function. Just plug in the correct g and h for your chosen loss.

---

## Complete Implementation

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    reg_lambda=1.0,        # L2 regularisation (lambda)
    reg_alpha=0.0,         # L1 regularisation
    gamma=0,               # minimum loss reduction to make a split
    subsample=0.8,         # row sampling (like bagging within boosting)
    colsample_bytree=0.8,  # column sampling per tree
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Regression
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    reg_lambda=1.0,
    objective='reg:squarederror',
    random_state=42
)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Early stopping (using validation set)
xgb_clf_es = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=20,
    eval_metric='logloss',
    random_state=42
)
xgb_clf_es.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print(f"Best iteration: {xgb_clf_es.best_iteration}")
```

---

## Key Hyperparameters

| Parameter | Meaning | Effect |
|---|---|---|
| `n_estimators` | Number of boosting rounds | More → complex model, overfitting risk |
| `learning_rate` (eta) | Shrinkage per round | Lower → need more estimators, better generalisation |
| `max_depth` | Max depth of each tree | Typically 3-10. Deeper → more complex, overfitting risk |
| `reg_lambda` (λ) | L2 regularisation on leaf weights | Higher → simpler trees, more conservative splits |
| `reg_alpha` | L1 regularisation on leaf weights | Higher → can zero out some leaf weights (sparsity) |
| `gamma` | Minimum gain required to make a split | Higher → fewer splits, simpler trees |
| `subsample` | Fraction of rows sampled per tree | < 1.0 adds randomness, reduces overfitting |
| `colsample_bytree` | Fraction of columns sampled per tree | < 1.0 adds randomness, like Random Forest's feature sampling |
| `min_child_weight` | Minimum sum of Hessian (h) needed in a child node | Higher → more conservative, prevents tiny unreliable splits |

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_lambda': uniform(0, 5),
    'gamma': uniform(0, 5)
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

---

## XGBoost vs Gradient Boosting vs LightGBM — Complete Comparison

| | Gradient Boosting (vanilla) | XGBoost | LightGBM |
|---|---|---|---|
| **Tree building criteria** | Variance reduction (standard) | Similarity Score (2nd-order Taylor) | Similarity Score + leaf-wise growth |
| **Optimisation order** | First-order (gradient only) | Second-order (gradient + Hessian) | Second-order |
| **Tree growth** | Depth-wise | Depth-wise (level-wise) | Leaf-wise |
| **Regularisation** | Implicit (via depth, learning rate) | Explicit (λ, α, γ in objective) | Explicit |
| **Missing value handling** | Manual imputation needed | Automatic (learned direction) | Automatic |
| **Speed** | Slower | Fast | Fastest (especially large data) |
| **Memory usage** | Higher | Moderate | Lowest |
| **Best for** | Small-medium data, baseline | Medium-large structured data | Very large datasets |

---

## Interview One-Liners

**What is XGBoost?**
"An optimised, regularised implementation of Gradient Boosting — not a new algorithm, but a combination of the Gradient Boosting mathematical framework with significant software engineering improvements: parallelisation, cache awareness, sparsity handling, and built-in regularisation."

**How does XGBoost build trees differently from vanilla Gradient Boosting?**
"Instead of standard variance/Gini-based splitting, XGBoost uses a Similarity Score derived from a second-order Taylor expansion of the loss function — using both the gradient (first derivative) and Hessian (second derivative). This is more like Newton's method than plain gradient descent, generally converging faster and more accurately."

**What is the Similarity Score?**
"Gⱼ²/(Hⱼ+λ) where G is the sum of gradients and H is the sum of Hessians in a node, and λ is a regularisation parameter. It measures how much loss reduction a node split would achieve, derived directly from optimally solving the Taylor-approximated objective function."

**What does lambda (λ) control?**
"It's an L2 regularisation parameter that sits in the denominator of the Similarity Score. Higher λ shrinks the similarity score for all nodes, making splits harder to justify — directly controlling tree complexity and overfitting, similar to Ridge regression's regularisation."

**XGBoost vs LightGBM?**
"Both are optimised Gradient Boosting implementations. XGBoost grows trees depth-wise (level by level); LightGBM grows leaf-wise (always splitting the leaf with highest loss reduction first), which can be faster and more memory-efficient on very large datasets, though sometimes more prone to overfitting on smaller ones."

**Why does XGBoost use second-order derivatives (Hessian)?**
"The Hessian captures the curvature of the loss function — essentially how confident the current model already is. Using both gradient and Hessian (Newton's method style) gives a more accurate approximation of the true loss reduction for a potential split than using gradient alone, leading to better split decisions and faster convergence."
