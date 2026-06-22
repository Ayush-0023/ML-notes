# Softmax Regression & Logistic Regression Extensions — Complete Notes

---

## 1. Softmax Regression (Multinomial Logistic Regression)

### What is it?

Logistic regression handles binary classification (2 classes). Softmax regression generalises this to **K classes** — any number of output categories.

Also called: Multinomial Logistic Regression.

**Example:** Predicting placement status → Not placed (0), Placed (1), Didn't sit (2) — 3 classes.

---

### Approach 1 — One vs Rest (OvR)

Convert the multi-class problem into multiple binary problems using One-Hot Encoding.

For K classes, train K separate binary logistic regression models:
- Model 1: Is it class 0? (vs all others)
- Model 2: Is it class 1? (vs all others)
- Model 3: Is it class 2? (vs all others)

**Parameters:** If you have n input features, each model has n weights + 1 bias = n+1 parameters. Total = K × (n+1) parameters.

**Example:** 2 inputs (CGPA, IQ), 3 classes → 3 × 3 = 9 parameters total.

**Prediction:** Run all K models, pick the class with the highest probability.

**Problem:** Probabilities from separate models don't sum to 1. You can get P(class 0)=0.7, P(class 1)=0.65, P(class 2)=0.6 — meaningless to compare. Also inefficient for many classes — need to train K separate models.

---

### Approach 2 — Softmax Regression (Proper Multiclass)

Train one single model that predicts all K classes simultaneously. Probabilities are forced to sum to 1.

**Two changes from binary logistic regression:**

| | Binary Logistic Regression | Softmax Regression |
|---|---|---|
| **Activation** | Sigmoid | Softmax |
| **Loss function** | Binary Cross Entropy | Categorical Cross Entropy |

---

### The Softmax Function

Converts K raw scores (logits) into K probabilities that sum to 1.

> **P(class k) = e^(zₖ) / Σⱼ e^(zⱼ)**

**Properties:**
- All outputs between 0 and 1
- All outputs sum to exactly 1
- Larger logit → larger probability (exponential amplifies differences)
- The largest logit gets a disproportionately large probability — confident predictions

**Example:**

```
z₀ = 2.1  → e^2.1 = 8.16
z₁ = 0.8  → e^0.8 = 2.23
z₂ = 0.1  → e^0.1 = 1.11
Sum = 11.50

P(class 0) = 8.16/11.50 = 0.71
P(class 1) = 2.23/11.50 = 0.19
P(class 2) = 1.11/11.50 = 0.10
Sum = 1.00 ✓
```

**Relationship to sigmoid:** When K=2, softmax reduces to exactly sigmoid. Sigmoid is a special case of softmax for binary classification.

---

### Step-by-Step Working of Softmax Regression

**Step 1 — Input Features**
> x = [x₁, x₂, ..., xₙ]  e.g., [CGPA=8.0, IQ=120]

**Step 2 — Compute Logits for Each Class**
Each class k has its own weight vector wₖ and bias bₖ:
> zₖ = wₖ · x + bₖ

For 3 classes and 2 inputs → 3 weight vectors × 2 weights + 3 biases = 9 parameters total.

In matrix form:
> z = Wx + b

Where W is (K × n) and b is (K × 1).

**Step 3 — Apply Softmax**
> P(class k) = e^(zₖ) / Σⱼ e^(zⱼ)

Get K probabilities summing to 1.

**Step 4 — Compute Loss (Categorical Cross Entropy)**
> L = -Σₖ yₖ · log(P(class k))

Where yₖ is 1 for the true class and 0 for others (one-hot encoded).

Since only the true class has yₖ=1, this simplifies to:
> L = -log(P(true class))

Intuition: penalise the model for assigning low probability to the correct class.

**Step 5 — Backpropagation + Gradient Descent**
Compute gradients of loss with respect to all weights and biases. Update using gradient descent.

**Step 6 — Repeat for All Samples**
After many epochs, the model learns to assign high probability to the correct class for each input.

---

### Categorical Cross Entropy — The Loss Function

> **J = -(1/n) Σᵢ Σₖ yᵢₖ · log(P(class k | xᵢ))**

Where:
- i = data point index
- k = class index
- yᵢₖ = 1 if point i belongs to class k, else 0

Since only one yᵢₖ = 1 per row (one true class):
> J = -(1/n) Σᵢ log(P(true class | xᵢ))

The model is penalised whenever it assigns low probability to the correct class. Confident wrong predictions get enormous penalties.

---

### Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Softmax regression — multi_class='multinomial' + softmax solver
model = LogisticRegression(
    multi_class='multinomial',  # use softmax
    solver='lbfgs',             # supports multinomial
    C=1.0,
    max_iter=1000
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)  # shape: (n_samples, n_classes)

print(classification_report(y_test, y_pred))
```

---

## 2. Polynomial Features in Logistic Regression (Non-Linear Decision Boundaries)

### The Problem

Standard logistic regression draws a **linear decision boundary** — a straight line in 2D, a hyperplane in higher dimensions. If the true boundary is curved, logistic regression will fail.

**Example:** Concentric circles — one class is a ring around another. No straight line can separate them.

### The Fix — Same as Polynomial Regression

Add polynomial features as preprocessing. The decision boundary in the **original** feature space becomes curved, even though the model is still linear in the transformed feature space.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
```

**What this does:**
- Original features: [x₁, x₂]
- After degree 2: [x₁, x₂, x₁², x₁x₂, x₂²]
- The linear boundary in this 5D space becomes a curve in the original 2D space

**Trade-off:** Higher degree → more flexible boundary → risk of overfitting. Combine with regularisation (C parameter) to control.

---

## 3. Logistic Regression Hyperparameters

### C (Inverse Regularisation Strength)

> C = 1/λ

**High C (e.g., C=100):** Very low regularisation → model fits training data closely → risk of overfitting
**Low C (e.g., C=0.01):** Strong regularisation → coefficients shrunk heavily → risk of underfitting
**Default C=1.0:** Moderate regularisation

```python
# Finding optimal C using cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)
print(f"Optimal C: {grid.best_params_['C']}")
```

### penalty

Controls the type of regularisation:

| Value | Regularisation | Notes |
|---|---|---|
| `'l2'` | Ridge (default) | Works with all solvers |
| `'l1'` | Lasso | Feature selection, sparse solution |
| `'elasticnet'` | ElasticNet | Requires solver='saga' |
| `None` | No regularisation | Risk of overfitting |

### solver

The optimisation algorithm used:

| Solver | Best For | Supports |
|---|---|---|
| `'lbfgs'` | Default, small-medium datasets | L2, multinomial |
| `'liblinear'` | Small datasets, binary | L1, L2 |
| `'saga'` | Large datasets | L1, L2, ElasticNet |
| `'newton-cg'` | Multiclass | L2 |
| `'sag'` | Large datasets | L2 |

### max_iter

Maximum number of iterations for the solver to converge. Increase if you get convergence warnings.

```python
LogisticRegression(max_iter=1000)  # default is 100 — often too low
```

### class_weight

Handles class imbalance by giving more weight to minority class during training:

```python
LogisticRegression(class_weight='balanced')
# 'balanced' automatically weights classes inversely proportional to frequency
# Alternative: class_weight={0: 1, 1: 10}  # manually specify weights
```

### multi_class

| Value | Behaviour |
|---|---|
| `'auto'` | Chooses based on data and solver (default) |
| `'ovr'` | One-vs-Rest — trains K binary classifiers |
| `'multinomial'` | Softmax — trains one model for all classes |

---

## Complete Logistic Regression — sklearn Parameters Summary

```python
LogisticRegression(
    C=1.0,                    # inverse regularisation strength
    penalty='l2',             # 'l1', 'l2', 'elasticnet', None
    solver='lbfgs',           # optimisation algorithm
    max_iter=1000,            # convergence iterations
    class_weight='balanced',  # handle imbalance
    multi_class='multinomial',# 'ovr' or 'multinomial'
    random_state=42,          # reproducibility
    tol=1e-4,                 # tolerance for convergence
    fit_intercept=True        # whether to fit bias term
)
```

---

## Binary vs Multiclass Logistic Regression — Full Picture

```
Binary Classification (2 classes)
        ↓
Sigmoid activation → probability between 0 and 1
        ↓
Binary Cross Entropy loss
        ↓
Threshold (default 0.5) → predict 0 or 1

Multiclass Classification (K classes)
        ↓
Two approaches:
    ├── One-vs-Rest (OvR) → K separate binary classifiers
    │       Simple but probabilities don't sum to 1
    └── Softmax → one unified model
            Softmax activation → K probabilities summing to 1
            Categorical Cross Entropy loss
            Predict class with highest probability
```

---

## Interview One-Liners

**What is softmax regression?**
"The generalisation of logistic regression to K classes. It replaces sigmoid with the softmax function — which converts K raw scores into K probabilities summing to 1. The loss function changes from binary cross-entropy to categorical cross-entropy."

**Softmax function?**
"P(class k) = e^(zₖ) / Σⱼ e^(zⱼ). Exponentiates each logit and normalises. Forces all probabilities to sum to 1. Larger logit → disproportionately larger probability."

**OvR vs Softmax?**
"OvR trains K separate binary classifiers — simple but probabilities don't sum to 1 and becomes expensive with many classes. Softmax trains one unified model — probabilities always sum to 1 and is more efficient. Softmax is the standard approach."

**How to handle non-linear boundaries in logistic regression?**
"Add polynomial features as preprocessing — same concept as polynomial regression. The decision boundary in the original feature space becomes curved while the model remains linear in the transformed space. Use with regularisation to prevent overfitting."

**What does the C parameter do?**
"C is the inverse of regularisation strength (C = 1/λ). High C → weak regularisation → more flexible model → risk of overfitting. Low C → strong regularisation → simpler model → risk of underfitting. Default C=1.0 gives moderate regularisation."
