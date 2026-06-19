# Ridge & Lasso Regression — Complete Notes

---

## Regularisation — The Core Idea

### What is Regularisation?

Regularisation is a technique used in machine learning to reduce overfitting by adding a penalty to the model's complexity. It prevents the model from learning noise in the training data by discouraging large coefficients.

**Why large coefficients = overfitting:**
In linear regression, large coefficients mean the model is extremely sensitive to small changes in input. A tiny change in x causes a huge swing in prediction. This is the mathematical signature of a model that has memorised training data rather than learned the true pattern.

**The fix:** Add a penalty term to the loss function that punishes large coefficients. Now the model must balance fitting the data well AND keeping coefficients small.

---

### Mathematical Intuition

**Standard OLS loss:**
> J = Σ(yᵢ - ŷᵢ)²

**With regularisation:**
> J = Σ(yᵢ - ŷᵢ)² + λ × (penalty on coefficients)

λ (lambda/alpha) is the **regularisation hyperparameter:**
- λ = 0 → pure OLS, no regularisation
- λ → ∞ → coefficients forced to zero → underfitting
- Optimal λ → balance between bias and variance

**The key insight:** The model is forced to choose a line that has both low error AND small coefficients. If a line with small coefficients fits only slightly worse than the "best" line with large coefficients, regularisation will prefer the former.

---

### Types of Regularisation

| Type | Penalty Term | Name |
|---|---|---|
| L2 | λΣwⱼ² | Ridge Regression |
| L1 | λΣ\|wⱼ\| | Lasso Regression |
| L1 + L2 | λ₁Σ\|wⱼ\| + λ₂Σwⱼ² | ElasticNet |

**Why L1 and L2?**
Named after the mathematical norm used:
- L2 norm = √(Σwⱼ²) → squaring gives L2 penalty
- L1 norm = Σ|wⱼ| → absolute value gives L1 penalty
- Could theoretically have L1.5, L3, etc. but L1 and L2 have the most useful properties

---

## Ridge Regression (L2 Regularisation)

### Definition

Ridge Regression adds the **sum of squared coefficients** as a penalty to the loss function:

> **J = Σ(yᵢ - ŷᵢ)² + λΣwⱼ²**

### Mathematical Derivation — 2D Case

For 1 input column:
> J = Σ(yᵢ - (mxᵢ + b))² + λm²

Taking partial derivative with respect to m and setting to zero:
> ∂J/∂m = -2Σxᵢ(yᵢ - mxᵢ - b) + 2λm = 0

Solving for m:
> **m = Σxᵢ(yᵢ - b) / (Σxᵢ² + λ)**

Compared to OLS where:
> m_OLS = Σxᵢ(yᵢ - b) / Σxᵢ²

**The only difference:** λ is added to the denominator. This is why Ridge shrinks m but can never make it exactly zero — you can't make a denominator zero by adding a positive number to it.

Since b depends on m:
> b = ȳ - m·x̄

b also changes when m changes.

### Matrix Form Derivation — nD Case

For n input columns, in matrix notation:

**Loss function:**
> J = (y - Xw)ᵀ(y - Xw) + λwᵀw

**Taking derivative and setting to zero:**
> ∂J/∂w = -2Xᵀ(y - Xw) + 2λw = 0

**Solving for w:**
> XᵀXw + λw = Xᵀy
> (XᵀX + λI)w = Xᵀy
> **w = (XᵀX + λI)⁻¹Xᵀy**

**Beautiful property:** Adding λI to XᵀX makes it **always invertible** — even when multicollinearity is present. This is why Ridge elegantly solves the multicollinearity problem that breaks OLS.

### 5 Key Points About Ridge Regression

**1. Coefficients shrink but never reach zero**
The L2 penalty applies smooth, continuous shrinkage. As λ increases, coefficients approach but never exactly reach zero. The quadratic penalty becomes infinitesimally small as w → 0, so the algorithm never has enough incentive to push it all the way.

**2. Higher coefficients are affected more**
If W₁ = 1000 and W₂ = 100, the penalty for W₁ is 1000² = 1,000,000 vs W₂'s 100² = 10,000. The rate of shrinkage is proportional to the coefficient's magnitude — large coefficients get penalised far more.

**3. Bias-Variance tradeoff**
- λ small (→0) → bias low, variance high → overfitting
- λ large (→∞) → bias high, variance low → underfitting
- Optimal λ → minimum total error

Find optimal λ by plotting bias and variance vs λ. Choose λ slightly before their intersection point where both are acceptably low.

**4. Effect on loss function shape**
As λ increases, the bowl-shaped loss function gets narrower and the minimum shifts toward the origin. The minimum never reaches zero — it just gets closer.

**5. Why called "Ridge"**
The λ penalty term λΣwⱼ² represents a circle (or sphere in higher dimensions) centered at the origin in coefficient space. The OLS solution (unconstrained minimum) typically lies outside this circle. Ridge regression constrains the solution to lie on the boundary of this circular "ridge." The solution is always on the ridge — hence the name.

---

### Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Always scale before Ridge — regularisation is scale-sensitive
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge with specific alpha (lambda)
ridge = Ridge(alpha=1.0)  # alpha = lambda
ridge.fit(X_train_scaled, y_train)

print(f"Coefficients: {ridge.coef_}")
print(f"Intercept: {ridge.intercept_}")

# Finding optimal alpha using cross-validation
from sklearn.linear_model import RidgeCV

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {ridge_cv.alpha_}")
```

---

## Lasso Regression (L1 Regularisation)

### Definition

Lasso (**L**east **A**bsolute **S**hrinkage and **S**election **O**perator) adds the **sum of absolute values of coefficients** as a penalty:

> **J = Σ(yᵢ - ŷᵢ)² + λΣ|wⱼ|**

### Mathematical Derivation — 2D Case

For 1 input column:
> J = Σ(yᵢ - (mxᵢ + b))² + λ|m|

Since |m| is not differentiable at m = 0, we split into 3 cases:

**Case 1: m > 0 (|m| = m)**
> ∂J/∂m = -2Σxᵢ(yᵢ - mxᵢ - b) + λ = 0
> m = (ΣxᵢYᵢ - λ/2) / Σxᵢ²

**Case 2: m = 0**
> λ|m| = 0 → same as simple regression

**Case 3: m < 0 (|m| = -m)**
> ∂J/∂m = -2Σxᵢ(yᵢ - mxᵢ - b) - λ = 0
> m = (ΣxᵢYᵢ + λ/2) / Σxᵢ²

### Why Lasso Creates Sparsity (Coefficients = Exactly Zero)

This is the critical difference from Ridge. Let's trace through numerically.

Let ΣxᵢYᵢ = 100 and Σxᵢ² = 50.

Using Case 1 formula (m > 0): m = (100 - λ/2) / 50

| λ | m |
|---|---|
| 0 | 2.0 |
| 10 | 1.9 |
| 100 | 1.0 |
| 200 | 0.0 |
| 300 | -1.0 ← but m < 0, use Case 3 |

When λ = 300 gives m = -1 using Case 1, but m < 0 violates the Case 1 assumption. Switching to Case 3: m = (100 + 150) / 50 = 5. Now m > 0 which violates Case 3. The only consistent solution is **m = 0**.

**Why this happens:**
- In Ridge: λ is in the **denominator** → denominator can never be zero → m can never be zero
- In Lasso: λ is in the **numerator** → at some λ value, λ cancels the numerator entirely → m = 0

This is why Lasso creates sparsity and Ridge doesn't. Once a coefficient hits zero in Lasso, the algorithm stops there — moving in either direction would increase the loss.

### Why Sparsity Matters

When you have high-dimensional data (100, 200 columns) or polynomial features, many features may be irrelevant or redundant. 

**Ridge:** All coefficients remain non-zero. The model uses all features — even irrelevant ones get small but non-zero weights. No feature selection.

**Lasso:** Irrelevant features get coefficient = exactly 0. They're completely removed from the model. Lasso **automatically performs feature selection** — reducing dimensionality without manual intervention.

This is Lasso's biggest advantage over Ridge.

### 4 Key Points About Lasso Regression

**1. Coefficients can reach exactly zero**
As λ increases, coefficients shrink — and the less important ones hit zero first. More important features (larger coefficients) are harder to push to zero.

**2. Higher coefficients are affected more**
Same as Ridge — larger coefficients face proportionally larger penalty. But the absolute value penalty is less "gentle" than the quadratic — coefficients drop faster.

**3. Bias-Variance tradeoff**
Same pattern as Ridge:
- λ small → overfitting (high variance, low bias)
- λ large → underfitting (low variance, high bias)
- Optimal λ → minimum total error → select in the region where both bias and variance are low

**4. Effect on loss function**
As λ increases, the loss minimum shifts toward zero. After a certain λ value, the minimum fixes itself at zero and the curve loses its bowl shape (flattens). This never happened in Ridge — the bowl always maintained its shape.

### No Closed Form Solution

Unlike Ridge which has β = (XᵀX + λI)⁻¹Xᵀy, Lasso has **no closed form solution**.

Why? Because |w| has a sharp corner at zero — not differentiable at that point. You can't set derivative to zero and solve analytically.

**Solution:** Coordinate Descent — optimise one coefficient at a time while holding others fixed. Iterate until convergence.

### Implementation

```python
from sklearn.linear_model import Lasso, LassoCV

# Always scale before Lasso
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso with specific alpha
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)

print(f"Coefficients: {lasso.coef_}")
print(f"Number of non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
print(f"Features selected: {np.sum(lasso.coef_ != 0)} out of {X_train.shape[1]}")

# Finding optimal alpha
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Optimal alpha: {lasso_cv.alpha_}")
```

---

## ElasticNet Regression

### When to Use

- **Ridge:** When all features are important — you want to shrink but keep all
- **Lasso:** When only a subset of features matter — you want automatic selection
- **ElasticNet:** When you have many features and don't know which matter — combines both

### Definition

> **J = Σ(yᵢ - ŷᵢ)² + λ₁Σ|wⱼ| + λ₂Σwⱼ²**

Two hyperparameters: λ₁ (L1 strength) and λ₂ (L2 strength).

In sklearn, controlled by:
- `alpha` — overall regularisation strength
- `l1_ratio` — mix between L1 and L2 (0 = pure Ridge, 1 = pure Lasso)

### Implementation

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

elasticnet = ElasticNet(alpha=1.0, l1_ratio=0.5)
elasticnet.fit(X_train_scaled, y_train)

# Finding optimal parameters
elasticnet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
    alphas=[0.001, 0.01, 0.1, 1.0],
    cv=5
)
elasticnet_cv.fit(X_train_scaled, y_train)
print(f"Optimal alpha: {elasticnet_cv.alpha_}")
print(f"Optimal l1_ratio: {elasticnet_cv.l1_ratio_}")
```

---

## Complete Comparison Table

| | OLS | Ridge (L2) | Lasso (L1) | ElasticNet |
|---|---|---|---|---|
| **Penalty** | None | λΣwⱼ² | λΣ\|wⱼ\| | λ₁Σ\|wⱼ\| + λ₂Σwⱼ² |
| **Coefficients reach zero** | No | No | Yes | Yes |
| **Feature selection** | No | No | Yes | Yes |
| **Handles multicollinearity** | No | Yes | Partial | Yes |
| **Closed form solution** | Yes | Yes | No | No |
| **Solver** | Normal Equation | Normal Equation | Coordinate Descent | Coordinate Descent |
| **Use when** | No overfitting | All features matter | Many irrelevant features | Unknown feature importance |

---

## Why Scaling is Mandatory Before Regularisation

Regularisation penalises large coefficients. But coefficient size depends on feature scale:
- A salary feature (0–100,000) will have tiny coefficients
- An age feature (0–100) will have larger coefficients for same "importance"

Without scaling, regularisation unfairly penalises features with larger scales — not features that are actually less important.

**Always StandardScaler before Ridge/Lasso/ElasticNet.**

---

## Geometric Intuition — Why L1 Creates Zeros and L2 Doesn't

The OLS solution sits somewhere in coefficient space. Regularisation constrains the solution to lie within a region:

**L2 constraint region:** A **circle** (sphere in higher dimensions)
- Smooth boundary, no corners
- The loss function ellipse touches the circle at a smooth point
- Coefficients get pushed toward zero but rarely land exactly on an axis

**L1 constraint region:** A **diamond** (with corners on the axes)
- Sharp corners sit exactly on the axes (where one coefficient = 0)
- The loss function ellipse is very likely to first touch the diamond at a corner
- When it touches at a corner, one coefficient = exactly zero → sparsity

This geometric difference is the fundamental reason why L1 creates sparsity and L2 doesn't.

---

## Interview One-Liners

**What is regularisation?**
"Adding a penalty term to the loss function to discourage large coefficients — preventing the model from memorising training noise and improving generalisation."

**Ridge vs Lasso?**
"Both add a penalty to the loss function. Ridge uses squared coefficients (L2) — shrinks all coefficients but never to zero, solves multicollinearity elegantly. Lasso uses absolute values (L1) — can shrink coefficients to exactly zero, performing automatic feature selection. The geometric reason: L2's circular constraint has no corners, L1's diamond constraint has corners on the axes where coefficients hit zero."

**Why does Lasso create sparsity but Ridge doesn't?**
"In Ridge, λ is in the denominator — you can't make a denominator zero. In Lasso, λ is in the numerator — at a certain λ value it cancels the numerator entirely, pushing the coefficient to exactly zero."

**When to use which?**
"Ridge when all features are important. Lasso when you believe many features are irrelevant and want automatic feature selection. ElasticNet when you're unsure — it combines both penalties."

**Why must you scale before regularisation?**
"Regularisation penalises large coefficients. Without scaling, features with larger ranges get unfairly penalised regardless of their actual importance. StandardScaler ensures the penalty is applied fairly across all features."

**What is λ (lambda)?**
"The regularisation strength hyperparameter. λ = 0 → pure OLS. λ → ∞ → all coefficients zero → underfitting. Optimal λ balances bias and variance — found using cross-validation."
