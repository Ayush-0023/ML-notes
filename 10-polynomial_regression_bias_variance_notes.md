# Polynomial Regression & Bias-Variance Tradeoff — Complete Notes

---

## Polynomial Regression

### Why Do We Need It?

Linear regression assumes the relationship between input and output is linear. But real world data is often non-linear — a straight line simply can't capture the pattern.

**Example:** House prices don't grow linearly with size — the relationship curves. Age vs medical cost curves upward. Salary vs experience plateaus after a certain point.

In these cases, linear regression will have high bias — the model is fundamentally the wrong shape regardless of how much data you give it.

**Key insight:** Polynomial regression is still a form of linear regression — the only difference is that we add polynomial terms as new features during preprocessing.

---

### How It Works

**Step 1 — Feature Transformation (Preprocessing)**

For 1 input column X, degree 2 polynomial:
> X → [X⁰, X¹, X²] = [1, X, X²]

Example: X = 35 → [1, 35, 1225]

Shape before: (n, 1) → Shape after: (n, 3)

For degree 3:
> X → [1, X, X², X³]

**Step 2 — Apply Linear Regression on Transformed Features**

The model equation becomes:
> ŷ = β₀ + β₁X + β₂X²  (degree 2)
> ŷ = β₀ + β₁X + β₂X² + β₃X³  (degree 3)

**Step 3 — Find Coefficients Using OLS or Gradient Descent**

Same as standard linear regression — the math is identical. We're just finding coefficients for more features.

---

### Why Is It Called "Linear" Regression?

The equation ŷ = β₀ + β₁X + β₂X² looks non-linear but:
- We already know the values of X, X², X³
- We only need to find the coefficients β₀, β₁, β₂, β₃
- The relationship between ŷ and the **coefficients** is linear

Linear in the **parameters** = linear regression. The input features can be any transformation.

---

### With Multiple Input Columns

For 2 inputs (X₁, X₂), degree 2:
> Features become: [1, X₁, X₂, X₁², X₁X₂, X₂²]

Cross terms (X₁X₂) are automatically included — they capture interaction effects between features.

**Number of features grows rapidly with degree and number of inputs:**

| Inputs | Degree | Features |
|---|---|---|
| 1 | 2 | 3 |
| 1 | 3 | 4 |
| 2 | 2 | 6 |
| 2 | 3 | 10 |
| 10 | 2 | 66 |
| 10 | 3 | 286 |

This rapid growth is why polynomial regression can quickly lead to overfitting with high degrees.

---

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Generate non-linear data
X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1 — Manual transformation
poly = PolynomialFeatures(degree=2, include_bias=True)
# include_bias=True adds the X⁰=1 column (intercept)
# interaction_only=False includes X² terms (default)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

# Method 2 — Using Pipeline (recommended)
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
```

---

### Choosing the Optimal Degree — The Hyperparameter Problem

Degree is a **hyperparameter** — set before training, not learned from data.

**Too low degree → Underfitting:**
- Model too simple to capture the pattern
- High bias, low variance
- Poor performance on both train and test

**Too high degree → Overfitting:**
- Model memorises every quirk in training data
- Low bias, high variance
- Excellent train performance, terrible test performance

**How to find the optimal degree:**

```python
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

train_scores = []
test_scores = []
degrees = range(1, 11)

for degree in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('model', LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    train_scores.append(r2_score(y_train, pipe.predict(X_train)))
    test_scores.append(r2_score(y_test, pipe.predict(X_test)))

plt.plot(degrees, train_scores, label='Train R²')
plt.plot(degrees, test_scores, label='Test R²')
plt.xlabel('Degree')
plt.ylabel('R² Score')
plt.legend()
plt.title('Finding Optimal Degree')
plt.show()

# Optimal degree = where test R² is highest
# After that point, test R² drops (overfitting begins)
```

For single input columns — also check the best fit line visually.
For multiple input columns — rely on R² scores and learning curves.

---

## Bias-Variance Tradeoff

### Definitions

**Bias:**
The inability of a model to truly capture the relationship between input and output. If the model can't find the pattern even in training data, it has high bias.

More precisely: Bias is the error from wrong assumptions in the model. A high-bias model is systematically wrong — it misses the true pattern regardless of how much data you give it.

*Analogy: Studying without effort and going to the exam — underperforms everywhere.*

**Variance:**
The difference in performance between training and testing datasets. If training error is 10 and test error is 100, variance = 90 (high).

More precisely: How much your model's predictions change when trained on different samples. A high-variance model is too sensitive to the specific data it was trained on — it memorises noise.

*Analogy: Only memorising past papers — performs great on familiar questions, fails on new ones.*

**We need:** Low bias AND low variance. But there's a fundamental tradeoff.

---

### The Tradeoff

The things that reduce bias tend to increase variance, and vice versa:

| Model complexity | Bias | Variance | Result |
|---|---|---|---|
| Too simple (low degree) | High | Low | Underfitting |
| Just right | Low | Low | Good generalisation |
| Too complex (high degree) | Low | High | Overfitting |

**Mathematically:**
> Total Error = Bias² + Variance + Irreducible Noise

Irreducible noise (ε) is the noise in the data itself — no model can eliminate it.

---

### Underfitting

Model is too simple to capture the underlying pattern.

**Signs:**
- Poor performance on training data AND test data
- High training error
- Low model complexity

**Causes:**
- Degree too low in polynomial regression
- λ too high in regularisation
- Wrong model choice (using linear model for non-linear data)
- Too few features

**Fixes:**
- Increase model complexity (higher degree, more features)
- Reduce regularisation strength (lower λ)
- Try a more powerful algorithm

---

### Overfitting

Model memorises training data including its noise — fails to generalise.

**Signs:**
- Excellent performance on training data, poor on test data
- Very large coefficients
- Large gap between train and test error

**Causes:**
- Degree too high in polynomial regression
- Too many features relative to data points
- Model too complex for the amount of data

**Three techniques to prevent overfitting:**
1. **Regularisation** — Ridge, Lasso, ElasticNet (add penalty for large coefficients)
2. **Bagging** — Random Forest (train multiple models on random subsets, average predictions)
3. **Boosting** — XGBoost, AdaBoost (train models sequentially, each fixing previous errors)

---

### Visualising the Tradeoff

```python
# Learning curve — shows bias-variance tradeoff
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='r2'
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label='Train Score')
plt.plot(train_sizes, test_mean, label='Test Score')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.legend()
plt.title('Learning Curve')
plt.show()

# High bias: both curves flat and low
# High variance: large gap between train and test curves
# Good fit: both curves high and close together
```

---

### Bias-Variance in Context of Ridge/Lasso

| λ value | Effect on bias | Effect on variance | Result |
|---|---|---|---|
| λ = 0 | Low (pure OLS) | High | Overfitting possible |
| λ optimal | Low-medium | Low-medium | Best generalisation |
| λ very high | High | Very low | Underfitting |

Finding optimal λ: plot bias and variance vs λ. Choose λ where both are acceptably low — slightly before their intersection point.

---

### Interview One-Liners

**What is bias?**
"Error from wrong assumptions — the model is systematically wrong regardless of data size. A straight line fit to curved data will always underfit no matter how much data you add."

**What is variance?**
"Sensitivity to training data — how much predictions change across different samples. High variance means the model memorised training noise rather than learning the true pattern."

**Why is there a tradeoff?**
"Reducing bias requires more complex models that fit training data more closely — but complex models are more sensitive to training data and have higher variance. They trade off because the same mechanisms that fix one tend to worsen the other."

**How does regularisation fit in?**
"Regularisation deliberately introduces a small amount of bias (by penalising large coefficients) to get a much larger reduction in variance. The total error (bias² + variance) often decreases even though bias increased."

**What is polynomial regression?**
"A form of linear regression where we add polynomial feature transformations as preprocessing. The model is still linear in its parameters — we're just fitting a curve instead of a line by expanding the feature space."
