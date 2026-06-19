# ElasticNet Regression — Complete Notes

---

## What is ElasticNet?

ElasticNet Regression combines both L1 (Lasso) and L2 (Ridge) regularisation into a single loss function. It's particularly useful when you have many correlated features or when you're unsure whether Ridge or Lasso is more appropriate.

---

## Loss Function

> **J = Σ(yᵢ - ŷᵢ)² + a·Σwⱼ² + b·Σ|wⱼ|**

Where:
- **a** = amount of Ridge (L2) regularisation applied
- **b** = amount of Lasso (L1) regularisation applied

### sklearn Hyperparameters

In sklearn, ElasticNet is controlled by two hyperparameters:

> **a = λ × (1 - l1_ratio)**
> **b = λ × l1_ratio**

| Hyperparameter | Default | Meaning |
|---|---|---|
| `alpha` (λ) | 1.0 | Overall regularisation strength |
| `l1_ratio` | 0.5 | Mix between L1 and L2 |

**Examples:**
- l1_ratio = 0.5 → 50% Ridge + 50% Lasso (default)
- l1_ratio = 0.9 → 10% Ridge + 90% Lasso
- l1_ratio = 0.0 → pure Ridge
- l1_ratio = 1.0 → pure Lasso

**Note:** In your notes it says "if l1_ratio increases, Ridge also increases" — this is actually reversed. Higher l1_ratio means more Lasso and less Ridge. l1_ratio directly controls the Lasso proportion.

---

## When to Use ElasticNet

**Case 1 — High dimensional data (too many columns):**
When the dataset has too many features and you don't know which ones are important. ElasticNet will zero out truly irrelevant features (via L1) while handling correlated ones gracefully (via L2).

**Case 2 — Multicollinearity:**
When input columns are highly correlated with each other (e.g., Height and Weight). Lasso alone tends to arbitrarily pick one correlated feature and zero out the others — even if both carry real information. Ridge distributes weight across correlated features. ElasticNet does both — selects relevant features AND distributes weight among correlated ones.

---

## Ridge vs Lasso vs ElasticNet — Decision Guide

| Situation | Use |
|---|---|
| All features are genuinely important | Ridge |
| Many irrelevant features, want automatic selection | Lasso |
| Many features, some correlated, unsure what matters | ElasticNet |
| No overfitting issue | Standard Linear Regression |

---

## Implementation

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler

# Always scale first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet with specific hyperparameters
en = ElasticNet(alpha=1.0, l1_ratio=0.5)
en.fit(X_train_scaled, y_train)

print(f"Coefficients: {en.coef_}")
print(f"Non-zero coefficients: {(en.coef_ != 0).sum()}")

# Finding optimal hyperparameters using cross-validation
en_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
    cv=5
)
en_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {en_cv.alpha_}")
print(f"Optimal l1_ratio: {en_cv.l1_ratio_}")
```

---

## Complete Regularisation Summary

| | OLS | Ridge (L2) | Lasso (L1) | ElasticNet |
|---|---|---|---|---|
| **Penalty** | None | λΣwⱼ² | λΣ\|wⱼ\| | λ₁Σ\|wⱼ\| + λ₂Σwⱼ² |
| **Coefficients → 0** | No | No (approaches) | Yes (exactly) | Yes |
| **Feature selection** | No | No | Yes | Yes |
| **Handles multicollinearity** | No | Yes | Partial | Yes |
| **Closed form** | Yes | Yes | No | No |
| **Solver** | Normal Equation | Normal Equation | Coordinate Descent | Coordinate Descent |

---

## Interview One-Liners

**What is ElasticNet?**
"ElasticNet combines L1 and L2 penalties — it gets feature selection from Lasso and stability from Ridge. Controlled by alpha (overall strength) and l1_ratio (mix between L1 and L2)."

**When to use ElasticNet over Ridge or Lasso?**
"When you have many features with unknown importance, especially when features are correlated. Lasso arbitrarily drops correlated features — ElasticNet handles them more gracefully by distributing weight while still zeroing truly irrelevant ones."
