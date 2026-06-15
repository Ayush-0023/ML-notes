# Regression Metrics — Complete Notes

---

## Why Do We Need Metrics?

You've built a model. You have residuals. But how do you summarise "how good is this model" in one number you can compare across different models?

That's what metrics do. Each metric answers a slightly different question — which is why you need more than one.

**The key insight:** Never rely on a single metric. Always evaluate multiple metrics together.

---

## 1. MAE — Mean Absolute Error

> **MAE = (1/n) Σ |yᵢ - ŷᵢ|**

Average of absolute errors. Every error contributes equally regardless of size.

**Intuition:** On average, your predictions are off by MAE units. Directly interpretable — same units as target variable.

### Pros
- Simple to interpret — "predictions are off by X rupees on average"
- Robust to outliers — a huge error doesn't dominate
- Suitable when all prediction errors should be treated equally

### Cons
- Not differentiable at zero — can't use calculus cleanly (though subgradient methods work)
- Doesn't emphasise large errors — less useful when large deviations are critical

### When to use
When your data has outliers you don't want to dominate the evaluation, or when all errors are equally important.

---

## 2. MSE — Mean Squared Error

> **MSE = (1/n) Σ (yᵢ - ŷᵢ)²**

Average of squared errors.

**Intuition:** Penalises large errors heavily — an error of 10 contributes 100, an error of 1 contributes 1.

### Pros
- Emphasises large errors — useful when large deviations are critical
- Differentiable everywhere — mathematically convenient for optimisation
- Directly corresponds to OLS objective function

### Cons
- Sensitive to outliers — one large error dominates
- Units are squared — harder to interpret (rupees² doesn't make intuitive sense)

### When to use
When large errors are especially undesirable — you'd rather have many small errors than one catastrophic one.

---

## 3. RMSE — Root Mean Squared Error

> **RMSE = √MSE = √((1/n) Σ (yᵢ - ŷᵢ)²)**

Square root of MSE.

**Why does RMSE exist?**
MSE is in squared units — if predicting salary in rupees, MSE is in rupees². RMSE brings it back to original units — rupees. Makes it interpretable like MAE but retains MSE's property of penalising large errors.

### Pros
- Same units as target variable — directly interpretable
- Still penalises large errors more than MAE
- Most commonly reported metric in practice

### Cons
- Still sensitive to outliers (inherited from MSE)

### MAE vs RMSE
- Both are in same units
- RMSE ≥ MAE always
- **Large gap between RMSE and MAE = outliers dominating**. If they're similar, errors are uniformly distributed.

---

## 4. R² — Coefficient of Determination

> **R² = 1 - (RSS / TSS)**

Where:
- **RSS** = Residual Sum of Squares = Σ(yᵢ - ŷᵢ)² — error of your model
- **TSS** = Total Sum of Squares = Σ(yᵢ - ȳ)² — error of baseline model (just predicts mean)

**Intuition:**
The dumbest possible model ignores x completely and always predicts ȳ (the mean of y). TSS measures how bad that model is. RSS measures how bad your model is. R² = how much better you are than the dumb baseline.

- R² = 1 → perfect model, RSS = 0
- R² = 0 → your model is as bad as just predicting the mean
- R² < 0 → your model is worse than predicting the mean (very bad model)

**Range:** (-∞, 1]

### Pros
- Intuitive — represents proportion of variance explained
- Useful for comparing models on the same dataset
- Scale-independent — works regardless of units

### Cons
- Sensitive to outliers
- **Critical flaw:** R² always increases or stays the same when you add more features — even useless ones. A model with 50 random features will have higher R² than a model with 5 useful ones.
- Doesn't tell you if the model is good in absolute terms — only relative to the mean baseline

### sklearn
```python
model.score(X_test, y_test)  # returns R²
```

---

## 5. Adjusted R²

> **Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]**

Where:
- n = number of observations
- p = number of predictors

**Why it exists:**
Adding any variable — even random noise — will increase R² slightly. Adjusted R² penalises you for adding features. It only increases when the new feature improves the model more than expected by chance.

**The penalty mechanism:**
As p increases, the denominator (n - p - 1) decreases, which multiplies (1 - R²) by a larger number, pulling Adjusted R² down — unless R² increased enough to compensate.

**Key property:**
- Adjusted R² < R² always
- Adjusted R² increases only when new variable genuinely helps
- Adjusted R² can decrease when you add a useless variable

**sklearn note:** No built-in function — always compute manually:
```python
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

---

## Why Not Just Use R²? — When You Need the Others

R² tells you **relative fit** — how good your model is compared to a baseline.
RMSE and MAE tell you **absolute error** — how wrong you are in real units.

**You need both:**

| Situation | Use |
|---|---|
| Explaining model to business stakeholders | RMSE or MAE — "off by ₹2L on average" is meaningful |
| Comparing models on same dataset | R² or Adjusted R² |
| Comparing models across different datasets | RMSE or MAE — R² is not comparable across datasets with different variance |
| Data has outliers, don't want them to influence evaluation | MAE |
| Large errors are especially costly | RMSE |
| Adding features, checking if they genuinely help | Adjusted R² |

---

## Code — Computing All Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

# MAE
mae = mean_absolute_error(y_test, y_pred)

# MSE
mse = mean_squared_error(y_test, y_pred)

# RMSE
rmse = np.sqrt(mse)
# OR in newer sklearn
rmse = mean_squared_error(y_test, y_pred, squared=False)

# R²
r2 = r2_score(y_test, y_pred)
# OR
r2 = model.score(X_test, y_test)

# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE:          {mae:.2f}")
print(f"MSE:          {mse:.2f}")
print(f"RMSE:         {rmse:.2f}")
print(f"R²:           {r2:.4f}")
print(f"Adjusted R²:  {adjusted_r2:.4f}")
```

---

## Train vs Test Evaluation — Always Do Both

```python
# Training metrics
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Test metrics
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
```

**Pattern interpretation:**

| Pattern | Meaning | Action |
|---|---|---|
| Train R² high, Test R² high | Good model | Deploy |
| Train R² high, Test R² low | Overfitting | Regularise, get more data |
| Train R² low, Test R² low | Underfitting | More complex model, more features |
| Train ≈ Test but both low | Model inadequate | Try different algorithm |

---

## Complete Summary Table

| Metric | Formula | Units | Outlier Sensitive | Best For |
|---|---|---|---|---|
| MAE | mean\|yᵢ - ŷᵢ\| | Same as y | No | Interpretable error, outliers present |
| MSE | mean(yᵢ - ŷᵢ)² | Squared | Yes | Optimisation objective |
| RMSE | √MSE | Same as y | Yes | General purpose reporting |
| R² | 1 - RSS/TSS | Unitless [(-∞,1]] | No | Model comparison, same dataset |
| Adjusted R² | Penalised R² | Unitless | No | Comparing models with different features |

---

## Interview One-Liners

**MAE:** "Average absolute error — robust to outliers, directly interpretable in the target's units."

**MSE:** "Average squared error — penalises large mistakes disproportionately, mathematically convenient for optimisation."

**RMSE:** "Root of MSE — brings MSE back to original units, most commonly reported in practice."

**R²:** "Proportion of variance explained — how much better your model is compared to just predicting the mean. Always between -∞ and 1."

**Adjusted R²:** "R² penalised for number of features — only increases when new features genuinely help, not just by adding noise. Essential when comparing models with different numbers of predictors."

**Why not just use R²?**
"R² is relative — tells you how good the fit is but not how wrong predictions are in real units. A model can have R² = 0.95 but still be off by ₹10 lakhs on average. You need RMSE or MAE to know the actual error magnitude."
