# Gradient Boosting — Complete Notes

---

## Bagging vs Boosting — The Fundamental Difference

| | Bagging | Boosting |
|---|---|---|
| **Training** | Parallel — all models trained independently | Sequential — each model depends on the previous |
| **Goal** | Reduce variance | Reduce bias |
| **Base learner ideal** | High variance, low bias (deep trees) | High bias, low variance (shallow trees/stumps) |
| **Data weighting** | Uniform (bootstrap sampling) | Errors of previous model inform the next |
| **Combination** | Simple average/vote | Weighted sequential addition |
| **Example** | Random Forest | AdaBoost, Gradient Boosting, XGBoost |

**Boosting's core idea:** Connect smaller (weak) models sequentially to build one bigger, stronger model. Each new model is told the mistakes of all previous models combined, and focuses specifically on fixing those mistakes.

---

## Gradient Boosting — The Core Idea

Unlike AdaBoost (which reweights misclassified data points), Gradient Boosting works by having each new model predict the **residual errors** (mistakes) of the combined previous models.

**The name "Gradient" Boosting:** Each new model is trained to approximate the **negative gradient** of the loss function with respect to the predictions — which, for squared error loss, turns out to simply be the residual (actual - predicted). This connects gradient boosting directly to gradient descent — but descending in "function space" rather than "parameter space."

---

## Gradient Boosting for Regression — Step by Step

### Setup

Suppose we use 3 small models (m1, m2, m3) for a regression problem.

### Step 1 — Initial Prediction (m1)

The first "model" is not an ML model at all — it's simply the **mean** of the target column.

> pred1 = mean(y)

This is the simplest possible prediction — if you had to guess y with no information at all, the mean minimises squared error.

### Step 2 — Calculate Residuals

> **res1 = y - pred1**

In Gradient Boosting terminology, this is called the **pseudo-residual** — "pseudo" because in the general case (any loss function), this isn't literally y - ŷ, but rather the negative gradient of the loss function. For squared error loss specifically, it happens to equal y - ŷ exactly.

### Step 3 — Train Second Model on Residuals (m2)

Model 2 (typically a decision tree — works best in practice) takes the **original input features (X)** but is trained to predict **res1** (the residuals from model 1), not the original y.

> m2 learns to predict: res1 = y - pred1

This means m2 is essentially learning to predict "how wrong was model 1, and in which direction."

### Step 4 — Combine Predictions with Learning Rate

> **y_pred = pred1 + η × m2(X)**

Where η (eta) is the **learning rate** (e.g., 0.1). The learning rate shrinks each new model's contribution — preventing the ensemble from overcorrecting too aggressively on any single round.

### Step 5 — Calculate New Residuals

> **res2 = y - y_pred = y - (pred1 + η×m2(X))**

### Step 6 — Train Third Model on New Residuals (m3)

Model 3 takes the original X but predicts **res2** — the residual that remains after models 1 and 2 combined.

### Step 7 — Final Combined Prediction

> **y_pred_final = pred1 + η×m2(X) + η×m3(X)**

This continues for as many models (`n_estimators`) as specified. Each new tree chips away at whatever error remains from all previous trees combined.

---

## Worked Code Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05*np.random.randn(100)  # non-linear relationship + noise

df = pd.DataFrame()
df['X'] = X.reshape(100)
df['y'] = y

plt.scatter(df['X'], df['y'])
plt.title('Non-linear data')
plt.show()

# Step 1 — Initial prediction = mean
df['pred1'] = df['y'].mean()
df['res1'] = df['y'] - df['pred1']

# Step 2 — Train tree on residuals
tree1 = DecisionTreeRegressor(max_leaf_nodes=8)  # 8-32 leaf nodes works well in practice
tree1.fit(df['X'].values.reshape(100, 1), df['res1'].values)

# Generate predictions for visualisation
X_test = np.linspace(-0.5, 0.5, 500)
learning_rate = 0.1
y_pred = df['pred1'].iloc[0] + learning_rate * tree1.predict(X_test.reshape(500, 1))

# Step 3 — Combine and calculate next residual
df['pred2'] = df['pred1'] + learning_rate * tree1.predict(df['X'].values.reshape(100, 1))
df['res2'] = df['y'] - df['pred2']

# Step 4 — Train second tree on new residuals
tree2 = DecisionTreeRegressor(max_leaf_nodes=8)
tree2.fit(df['X'].values.reshape(100, 1), df['res2'].values)

# Combined prediction from both trees
y_pred_combined = (
    df['pred1'].iloc[0] +
    learning_rate * sum(
        tree.predict(X_test.reshape(-1, 1)) for tree in [tree1, tree2]
    )
)
```

**This loop continues** — calculate residual, train new tree on residual, add scaled prediction to ensemble — for as many rounds as desired.

---

## sklearn Implementation (Production Use)

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

# Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(
    n_estimators=100,      # number of sequential trees
    learning_rate=0.1,     # shrinkage factor
    max_depth=3,           # depth of each tree
    subsample=1.0,         # fraction of samples used per tree (< 1.0 = stochastic GB)
    random_state=42
)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Classification
gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## AdaBoost vs Gradient Boosting

| | AdaBoost | Gradient Boosting |
|---|---|---|
| **Error correction mechanism** | Reweight misclassified samples | Fit new model to residuals (pseudo-gradient) |
| **Base learner** | Decision stumps (depth=1) typically | Shallow trees (depth 3-8) typically |
| **Combination weight** | α calculated per model based on error rate | Fixed learning rate shrinkage for all models |
| **Loss function flexibility** | Limited — exponential loss originally | Highly flexible — any differentiable loss function |
| **Sensitivity to outliers** | High (weights blow up on hard points) | Lower (gradient-based, more graceful) |
| **Generalises to** | Primarily classification | Both regression and classification naturally |

**The key conceptual leap from AdaBoost to Gradient Boosting:** AdaBoost manipulates sample weights based on a specific error rate formula. Gradient Boosting instead reframes boosting as **gradient descent in function space** — generalising to ANY differentiable loss function, not just exponential loss. This is why Gradient Boosting is more flexible and became the foundation for XGBoost, LightGBM, and CatBoost.

---

## Mathematics of Gradient Boosting

### Additive Modelling

In standard ML, we create a single function f(X) to map X to Y. But sometimes the relationship is too complex for one function — even polynomial regression can fail (Runge's Phenomenon: high-degree polynomials oscillate wildly near the edges of the data range, especially with evenly spaced points).

**Additive Modelling solves this** by approximating the big, complex function as a SUM of many smaller, simpler functions:

> **F(x) = f₀(x) + f₁(x) + f₂(x) + ... + fₘ(x)**

Each fᵢ(x) is simple (like a shallow decision tree), but their sum can approximate arbitrarily complex relationships. **Boosting algorithms are a direct application of additive modelling** — each new tree is one more term added to the sum.

### Formal Gradient Boosting Derivation

**Goal:** Minimise a loss function L(y, F(x)) by building F(x) additively.

**Step 1 — Initialise:**
> F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)

For squared error loss, this minimiser is the mean of y (exactly what we did in Step 1 of the worked example).

**Step 2 — For each boosting round m = 1 to M:**

**(a) Compute the pseudo-residuals** — the negative gradient of the loss function with respect to the current prediction:

> **rᵢₘ = -[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]** evaluated at F(x) = F_{m-1}(x)

For squared error loss L = (1/2)(y-F(x))²:
> ∂L/∂F(x) = -(y - F(x))
> Negative gradient = (y - F(x)) = the actual residual

**This is exactly why, for squared error loss, "pseudo-residual" = ordinary residual.** For other loss functions (e.g., absolute error, log-loss for classification), the pseudo-residual is a different but analogous quantity — always representing "the direction the prediction needs to move to reduce loss."

**(b) Fit a weak learner (tree) hₘ(x) to predict the pseudo-residuals rᵢₘ.**

**(c) Compute the optimal step size (often simplified to a fixed learning rate η in practice):**

> γₘ = argmin_γ Σᵢ L(yᵢ, F_{m-1}(xᵢ) + γ·hₘ(xᵢ))

**(d) Update the model:**

> **Fₘ(x) = F_{m-1}(x) + η × γₘ × hₘ(x)**

**Step 3 — Final model after M rounds:**

> **F(x) = F₀(x) + η Σₘ γₘhₘ(x)**

This is gradient descent — but instead of updating numerical parameters, we're updating an entire FUNCTION at each step, by adding a new tree that points in the direction of steepest loss decrease.

---

## Gradient Boosting for Classification — Geometric Intuition

For classification, we can't directly use "residual = actual - predicted" since predictions are probabilities, not raw values. Instead, Gradient Boosting for classification works in **log-odds space** (similar to logistic regression).

### The Process

**Step 1 — Initial prediction:** Start with the log-odds of the base rate:
> F₀(x) = log(p / (1-p))  where p = proportion of positive class

**Step 2 — Convert to probability via sigmoid:**
> P(x) = 1 / (1 + e^(-F(x)))

**Step 3 — Calculate pseudo-residuals:**
For log-loss (the standard classification loss), the negative gradient simplifies elegantly to:
> rᵢ = yᵢ - P(xᵢ)

Same intuitive form as regression — actual minus predicted probability — but now derived from the log-loss gradient, not squared error.

**Step 4 — Fit a tree to these residuals**, just like in regression.

**Step 5 — Update F(x) in log-odds space**, then convert back to probability via sigmoid for predictions.

**Geometric interpretation:** Each new tree shifts the decision boundary's log-odds surface slightly, in the direction that reduces classification error most — analogous to how each tree shifted the regression curve toward the true values.

---

## Key Hyperparameters

| Parameter | Meaning | Effect |
|---|---|---|
| `n_estimators` | Number of sequential trees | More → more complex, risk of overfitting |
| `learning_rate` | Shrinkage applied to each tree's contribution | Lower → need more trees, but better generalisation |
| `max_depth` | Depth of each individual tree | Typically shallow (3-8) — deep trees overfit fast in boosting |
| `subsample` | Fraction of training data used per tree | < 1.0 introduces randomness — "Stochastic Gradient Boosting" |
| `min_samples_split` / `min_samples_leaf` | Standard tree regularisation | Prevents overly specific splits |

### The n_estimators / learning_rate Tradeoff

This is the most important tuning relationship in Gradient Boosting:

> Lower learning_rate + more n_estimators ≈ Higher learning_rate + fewer n_estimators (similar final performance, but former generalises better)

**Common practice:** Set learning_rate low (0.01-0.1) and use early stopping to find the optimal n_estimators automatically:

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=1000,       # set high
    learning_rate=0.05,      # set low
    max_depth=3,
    validation_fraction=0.1,
    n_iter_no_change=10,     # stop if no improvement for 10 rounds
    random_state=42
)
gbr.fit(X_train, y_train)
print(f"Optimal number of trees used: {gbr.n_estimators_}")
```

---

## Why Shallow Trees in Gradient Boosting?

Unlike Random Forest (which benefits from deep trees to maximise individual tree variance, later reduced by averaging), Gradient Boosting works best with **shallow trees** (depth 3-8, sometimes even depth 1 stumps).

**Reasoning:** Boosting already builds complexity through the sequential addition of many trees. If each individual tree is also complex (deep), the ensemble overfits extremely fast — you're compounding complexity twice. Shallow trees keep each step "weak," letting the additive process build complexity gradually and controllably.

---

## Complete Flow Diagram

```
Initialise F₀(x) = mean(y)  [or log-odds for classification]
        ↓
For m = 1 to n_estimators:
        ↓
    Calculate pseudo-residuals = negative gradient of loss
        ↓
    Train a shallow tree hₘ(x) to predict these residuals
        ↓
    Update: Fₘ(x) = F_{m-1}(x) + learning_rate × hₘ(x)
        ↓
    (repeat)
        ↓
Final prediction: F(x) = sum of all weighted trees
[Apply sigmoid if classification, to convert log-odds → probability]
```

---

## Interview One-Liners

**What is Gradient Boosting?**
"A sequential boosting technique where each new model is trained to predict the residual errors (negative gradient of the loss function) of the combined previous models. It's essentially gradient descent performed in function space rather than parameter space."

**How does it differ from AdaBoost?**
"AdaBoost reweights misclassified samples and combines models using a calculated 'say' (alpha) based on error rate. Gradient Boosting instead fits each new model directly to the residuals/pseudo-gradients of the loss function — making it generalisable to any differentiable loss function, not just exponential loss."

**Why are pseudo-residuals called 'pseudo'?**
"For squared error loss, the negative gradient happens to exactly equal (actual - predicted) — the ordinary residual. But for other loss functions (e.g., log-loss for classification), the negative gradient is a different, analogous quantity — hence 'pseudo' residual, since it's not always the literal arithmetic residual."

**Why shallow trees in Gradient Boosting but deep trees in Random Forest?**
"Random Forest reduces variance by averaging many high-variance (deep) trees. Gradient Boosting builds complexity additively and sequentially — using deep trees too would compound complexity and overfit rapidly. Shallow trees keep each boosting step weak and controlled."

**What is additive modelling and why does boosting use it?**
"Additive modelling approximates a complex function as the sum of many simple functions. Boosting is a direct application — each new weak learner is one additional term in the sum, gradually building up to approximate complex, non-linear relationships that a single model (even polynomial regression, due to Runge's phenomenon) might fail to capture."

**What does learning_rate control?**
"It shrinks each new tree's contribution to the overall prediction. Lower learning rate requires more estimators but generally leads to better generalisation — it's the primary regularisation lever in Gradient Boosting, working jointly with n_estimators."
