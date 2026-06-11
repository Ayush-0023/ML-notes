# Linear Regression — Complete Notes

---

## What is Linear Regression?

Linear regression is a statistical technique used to model the relationship between one or more independent variables (predictors) and a dependent variable (target). The goal is to find a linear equation that best predicts the target variable.

**Key distinction:** Linear regression is linear in the **parameters** — not necessarily in the input variables. So polynomial regression (with x², x³) is still linear regression because the coefficients remain linear.

### Three Types

| Type | Input | Output | Equation |
|---|---|---|---|
| Simple LR | Single column | Numerical | y = mx + c |
| Multiple LR | Multiple columns | Numerical | y = β₀ + β₁x₁ + β₂x₂ + ... |
| Polynomial LR | Single/multiple columns with polynomial terms | Numerical | y = β₀ + β₁x + β₂x² + ... |

---

## Intuition

**Example:** CGPA as input, Package as output.

We need to find m and c in: **Package = m(CGPA) + c**

- **m (slope/weight)** — tells how much the output depends on the input. If m is large, a small change in CGPA causes a large change in Package. If m is small, the effect is minimal.
- **c (intercept/offset)** — the value of output when input = 0. Example: if Experience = 0 (fresher), the Package still has some base value = c.

**The entire job of linear regression is to find the best m and c for your data.**

---

## The Real World Has Noise

No line perfectly captures reality. Two students who both studied the same hours won't score exactly the same — sleep, luck, mood all affect the outcome. So the true model is:

> **y = mx + c + ε**

Where ε (epsilon) is the noise term — everything the line can't capture.

**Where does ε ~ N(0, σ²) come from?**
Noise is the sum of many small independent effects (sleep, luck, mood...). The Central Limit Theorem says: when you add many small independent random effects, the result follows a Gaussian distribution regardless of what each individual effect follows.

So ε ~ N(0, σ²) is not an arbitrary assumption — it's a mathematical consequence of how noise works in the real world.

---

## How to Find m and c — Two Methods

### Method 1 — OLS (Ordinary Least Squares)

A closed-form solution — a direct formula.

#### Why Squared Errors?

We measure error as the distance between actual and predicted values. Three options:

**Option 1 — Sum of raw errors:** Σ(yᵢ - ŷᵢ)
Problem: positive and negative errors cancel out. Useless.

**Option 2 — Sum of absolute errors:** Σ|yᵢ - ŷᵢ|
Problem: |x| has a sharp corner at zero — not differentiable at that point. Can't solve analytically.

**Option 3 — Sum of squared errors:** Σ(yᵢ - ŷᵢ)²
Three reasons this wins:
1. Eliminates cancellation
2. Penalises large errors more heavily (error of 10 → contributes 100; error of 1 → contributes 1)
3. Smooth, differentiable everywhere — can minimise analytically

**The deeper reason:** Squared errors correspond to MLE (Maximum Likelihood Estimation) under the assumption that noise is Gaussian. They are the same thing viewed from different angles.

#### Loss Function vs Cost Function

| Term | What it measures | Formula |
|---|---|---|
| **Loss function** | Error for a single data point | L = (yᵢ - ŷᵢ)² |
| **Cost function** | Aggregated error across entire dataset | J = (1/n)Σ(yᵢ - ŷᵢ)² |

#### Deriving m and c

The cost function is:
> J(m, c) = Σ(yᵢ - mxᵢ - c)²

To minimise J, take partial derivatives and set to zero:

**Partial derivative with respect to c:**
> ∂J/∂c = Σ 2(yᵢ - mxᵢ - c)(-1) = 0
> → Σ(yᵢ - mxᵢ - c) = 0

This says: sum of all errors = 0. The line passes through the centre of the data.

**Partial derivative with respect to m:**
> ∂J/∂m = Σ 2(yᵢ - mxᵢ - c)(-xᵢ) = 0
> → Σxᵢ(yᵢ - mxᵢ - c) = 0

Solving these two equations simultaneously gives:

> **m = [Σxᵢyᵢ - n·x̄·ȳ] / [Σxᵢ² - n·x̄²]**
> **c = ȳ - m·x̄**

Where x̄ and ȳ are the means of x and y.

#### OLS in Code

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Slope (m): {model.coef_}")
print(f"Intercept (c): {model.intercept_}")

y_pred = model.predict(X_test)
```

---

## Multiple Linear Regression

### What is it?

MLR models the relationship between one dependent variable and **two or more** independent variables.

**Example:** 2 inputs (CGPA, IQ) and 1 output (LPA) → 3D case → best fit **plane** (not line).

- 2D → best fit line
- 3D → best fit plane
- 4D+ → best fit hyperplane

### The Equation

**For 3D (2 inputs):**
> ŷ = β₀ + β₁x₁ + β₂x₂

**General form (n inputs):**
> ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

The coefficients β₁, β₂, ... βₙ tell us how much the output depends on each respective input.

### Matrix Formulation

For n rows and m input columns, we can write the entire system as:

> **ŷ = Xβ**

Where:
- **X** is the (n × m+1) input matrix with a column of 1s prepended for the intercept
- **β** is the (m+1 × 1) coefficient vector [β₀, β₁, β₂, ..., βₘ]ᵀ
- **ŷ** is the (n × 1) predicted output vector

The residuals (errors) are:
> **e = y - ŷ = y - Xβ**

The cost function (Sum of Squared Errors):
> **J = eᵀe = (y - Xβ)ᵀ(y - Xβ)**

### Normal Equation — Closed Form Solution

Minimising J with respect to β:
> ∂J/∂β = -2Xᵀ(y - Xβ) = 0
> → XᵀXβ = Xᵀy
> → **β = (XᵀX)⁻¹Xᵀy**

This is the **Normal Equation** — the single most important formula in linear regression.

**Unpacking each piece:**
- **Xᵀy** — how each predictor correlates with the target
- **XᵀX** — how predictors relate to each other
- **(XᵀX)⁻¹** — exists only if XᵀX is invertible

**When does (XᵀX)⁻¹ fail?**
When two or more input features are perfectly correlated (multicollinearity). Example: X₁ = "Years of Experience", X₂ = 2×X₁. XᵀX becomes singular (determinant = 0), inverse doesn't exist.

---

## The Three Lenses of Linear Regression

The same problem can be understood three ways — all equivalent:

| Perspective | Statement |
|---|---|
| **Algebraic** | Minimise Σ(yᵢ - ŷᵢ)² by setting partial derivatives to zero |
| **Probabilistic** | Maximise likelihood under Gaussian noise assumption (MLE) |
| **Geometric** | Project y onto the column space of X — residuals are perpendicular to that space |

---

## OLS Properties — Gauss-Markov Theorem

OLS is **BLUE** — Best Linear Unbiased Estimator — under five assumptions:

| # | Assumption | What it protects |
|---|---|---|
| 1 | Linearity | Model shape is correct |
| 2 | No perfect multicollinearity | (XᵀX)⁻¹ exists and is stable |
| 3 | E[ε\|X] = 0 | β estimates are unbiased |
| 4 | Homoskedasticity — Var(ε) = σ² constant | Standard errors are correct |
| 5 | No autocorrelation | Standard errors are correct |

**Critical pattern:**
- Violations 1, 2, 3 → β itself is wrong
- Violations 4, 5 → β is fine but inference (confidence intervals, p-values) is wrong

### Why OLS is Unbiased — Proof

From the Normal Equation:
> β̂ = (XᵀX)⁻¹Xᵀy = (XᵀX)⁻¹Xᵀ(Xβ + ε) = β + (XᵀX)⁻¹Xᵀε

Taking expectation (since E[ε] = 0):
> **E[β̂] = β** ✓

The moment E[ε] ≠ 0 (omitted variable bias), this proof collapses and β̂ becomes biased.

---

## Assumptions — What Breaks When They Fail

| Assumption violated | Problem | Fix |
|---|---|---|
| Non-linearity | Model is wrong shape | Polynomial/non-linear models |
| Multicollinearity | β unstable or undefined | Ridge regression, drop correlated features |
| E[ε\|X] ≠ 0 (omitted variable) | β is biased | Add the missing variable |
| Heteroskedasticity | Standard errors wrong | Weighted Least Squares, robust SE |
| Autocorrelation | Standard errors wrong | Time series models (ARIMA) |

---

## OLS vs Gradient Descent — When to Use Which

| Situation | Use OLS | Use Gradient Descent |
|---|---|---|
| Small/medium dataset | ✓ | ✓ |
| Large dataset (millions of rows) | ✗ (O(n³) matrix inversion) | ✓ |
| Multicollinearity present | ✗ (XᵀX not invertible) | ✓ |
| Need exact solution | ✓ | ✗ (approximation) |

**sklearn mapping:**
- `LinearRegression()` → uses OLS
- `SGDRegressor()` → uses Gradient Descent

---

## Regularisation — When OLS Overfits

When the model memorises training data (large coefficients), add a penalty:

**Ridge (L2):**
> J = Σ(yᵢ - ŷᵢ)² + λΣβⱼ²
- Shrinks all coefficients toward zero
- Fixes multicollinearity by adding λI to XᵀX → always invertible
- Closed form: β = (XᵀX + λI)⁻¹Xᵀy

**Lasso (L1):**
> J = Σ(yᵢ - ŷᵢ)² + λΣ|βⱼ|
- Shrinks coefficients to exactly zero → automatic feature selection
- No closed form → solved iteratively

**Elastic Net:**
> J = Σ(yᵢ - ŷᵢ)² + λ₁Σ|βⱼ| + λ₂Σβⱼ²
- Combines both penalties

---

## The Three Things That Sound the Same But Aren't

| | What it is | In LR |
|---|---|---|
| **Model** | Assumed relationship shape | ŷ = mx + c |
| **Loss function** | How we measure error | J = Σ(yᵢ - ŷᵢ)² |
| **Estimation method** | How we minimise the loss | OLS or Gradient Descent |

These are independent — you can mix and match them.

---

## Interview One-Liners

**What is linear regression?**
"Finding the linear relationship in data that minimises prediction error — formally, minimising the sum of squared residuals to find the best-fit line or hyperplane."

**Why squared errors?**
"Three reasons: eliminates cancellation, penalises large errors more heavily, and gives a smooth differentiable surface. More deeply — squared errors correspond to MLE under Gaussian noise."

**What is the Normal Equation?**
"β = (XᵀX)⁻¹Xᵀy — the closed-form solution that directly computes optimal coefficients in one shot. Fails when XᵀX is not invertible due to multicollinearity."

**What is Gauss-Markov?**
"Under five assumptions, OLS is BLUE — lowest variance among all linear unbiased estimators. It doesn't say OLS is universally best — Ridge can outperform it by trading bias for lower variance."

**Linear vs Multiple Linear Regression?**
"Same concept — find the best-fit hyperplane. Simple LR finds a line in 2D. Multiple LR finds a plane or hyperplane in higher dimensions. The matrix formulation handles both identically."
