# Gradient Descent — Complete Notes

---

## Why Gradient Descent?

OLS gives us the Normal Equation: **β = (XᵀX)⁻¹Xᵀy**

So why do we need Gradient Descent at all?

**Problem 1 — Computational cost:**
Matrix inversion using Gauss-Jordan Elimination has complexity **O(n³)**. For large datasets with many features, this becomes prohibitively slow.

**Problem 2 — Multicollinearity:**
If two features are perfectly correlated (e.g., X₂ = 2×X₁), XᵀX becomes singular — determinant is zero, inverse doesn't exist. OLS completely breaks down.

**Problem 3 — Universality:**
OLS only works for linear regression with squared error loss. Gradient Descent works for **any** algorithm with a differentiable loss function — logistic regression, neural networks, anything.

---

## What is Gradient Descent?

Gradient descent is a first-order iterative optimisation algorithm for finding a local minimum of a differentiable function.

The idea: take repeated small steps in the **opposite direction of the gradient** (steepest descent) until you reach the minimum.

**Why "first-order"?**
- First-order methods use only the **first derivative** (gradient) to determine direction
- Second-order methods (e.g., Newton's Method) use both first and second derivative (Hessian matrix) — more accurate but requires matrix inversion, which is not practical for high-dimensional problems

---

## Intuition — Step by Step

Consider 1 input (CGPA) and 1 output (LPA). We already know m and want to find b.

The loss function L depends on b. If we plot L vs b, we get a bowl-shaped curve with one minimum.

**Step 1 — Start at a random point**
Select a random value for b (e.g., b = -10). Mark the corresponding L value.

**Step 2 — Calculate the slope (gradient)**
- If slope is **negative** → move right (increase b) → towards minimum
- If slope is **positive** → move left (decrease b) → towards minimum
- The algorithm doesn't "know" which way is right — it just follows the slope

**Step 3 — Update b**
Without learning rate: b_new = b_old - slope
Problem: if slope = -50, then b_new = -10 - (-50) = 40. Value jumps drastically → zigzag behaviour → many iterations.

**Step 4 — Introduce Learning Rate (α)**
> **b_new = b_old - α × slope**

With α = 0.01: b_new = -10 - (0.01 × -50) = -9.5 → small controlled step toward minimum.

**Step 5 — Repeat until convergence**
Two stopping criteria:
1. |b_new - b_old| < threshold (e.g., 0.001) — stop when change is negligible
2. Fixed number of iterations called **epochs** (e.g., 1000) — use loss curve to find when it flattens

---

## Mathematical Formulation

### Simple Case (finding b only, m known)

Loss function:
> L = (1/n) Σ(yᵢ - (mxᵢ + b))²

Derivative with respect to b:
> ∂L/∂b = -(2/n) Σ(yᵢ - (mxᵢ + b))
> = (2/n) Σ(ŷᵢ - yᵢ)

Update rule:
> **b_new = b_old - α × ∂L/∂b**

### Full Case (finding both m and b)

Since both m and b are unknown, the loss function depends on both. In 3D, this looks like a valley — we need to travel along the curved surface to reach the minimum.

At every step, there are two components — one in m's direction and one in b's direction. We use **partial derivatives** for both:

Partial derivative with respect to b:
> ∂L/∂b = -(2/n) Σ(yᵢ - ŷᵢ)

Partial derivative with respect to m:
> ∂L/∂m = -(2/n) Σxᵢ(yᵢ - ŷᵢ)

Update rules:
> **b_new = b_old - α × ∂L/∂b**
> **m_new = m_old - α × ∂L/∂m**

Both updates happen simultaneously at every iteration.

---

## Three Types of Gradient Descent

### 1. Batch Gradient Descent

Computes gradient using **all n training examples** in every update step.

> ∂J/∂w = (1/n) Σᵢ (ŷᵢ - yᵢ) xᵢ

```python
for epoch in range(epochs):
    y_pred = X.dot(w) + b
    error = y_pred - y
    dw = (1/n) * X.T.dot(error)   # ALL samples
    db = (1/n) * np.sum(error)
    w = w - lr * dw
    b = b - lr * db
```

**Convergence path:** Smooth and stable — no noise.

```
Cost
  |●
  | ●
  |  ●●●●____________  ← converges smoothly
  +─────────────────→ iterations
```

**Pros:** Stable convergence, guaranteed to find minimum for convex functions.
**Cons:** Very slow for large datasets — one update requires computing gradients across all n points.

**sklearn:** `LinearRegression()` uses OLS not GD. Use `SGDRegressor(learning_rate='constant', eta0=0.01)` for GD.

---

### 2. Stochastic Gradient Descent (SGD)

Updates weights using **one random sample** at a time.

```python
for epoch in range(epochs):
    for i in range(n):
        random_idx = np.random.randint(0, n)
        xi = X[random_idx:random_idx+1]
        yi = y[random_idx:random_idx+1]

        y_pred = xi.dot(w) + b
        error = y_pred - yi
        dw = xi.T.dot(error)
        db = np.sum(error)
        w = w - lr * dw
        b = b - lr * db
```

**Convergence path:** Noisy and erratic — bounces around the minimum.

```
Cost
  |● ●
  |  ● ●  ●
  |    ●●● ●●●_______  ← noisy but converges
  +─────────────────→ iterations
```

**Pros:** 
- Much faster per epoch — one sample per update
- Noise helps escape local minima (useful for non-convex loss functions)
- Can do online learning — update model as new data arrives

**Cons:**
- Never fully converges — keeps bouncing near minimum
- Noisy updates — high variance

**sklearn:** `SGDRegressor()`

---

### 3. Mini-Batch Gradient Descent

Updates weights using a **small batch** of samples (typically 32, 64, or 128) — the middle ground.

```python
batch_size = 32

for epoch in range(epochs):
    for i in range(0, n, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        y_pred = X_batch.dot(w) + b
        error = y_pred - y_batch
        dw = (1/batch_size) * X_batch.T.dot(error)
        db = (1/batch_size) * np.sum(error)
        w = w - lr * dw
        b = b - lr * db
```

**Convergence path:** Slightly noisy but generally smooth — best of both worlds.

**Pros:**
- Faster than Batch GD — doesn't need all data per update
- More stable than SGD — batch averaging reduces noise
- Takes advantage of vectorised operations and GPU parallelism
- **Standard choice in deep learning**

**Cons:** Adds batch_size as a new hyperparameter to tune.

---

### Comparison Table

| | Batch GD | SGD | Mini-Batch GD |
|---|---|---|---|
| **Samples per update** | All n | 1 | batch_size (32-256) |
| **Convergence** | Smooth | Noisy | Slightly noisy |
| **Speed** | Slowest | Fastest per update | Best overall |
| **Memory** | High | Low | Medium |
| **Escapes local minima** | No | Yes | Sometimes |
| **Used in** | Small datasets | Online learning | Deep learning (standard) |

---

## Effect of Learning Rate

**Too small (e.g., α = 0.0001):**
- Converges very slowly
- Needs many epochs
- May not reach minimum within fixed iterations

**Too large (e.g., α = 1.0):**
- Overshoots the minimum
- Keeps bouncing past the solution
- May never converge — diverges

**Just right (e.g., α = 0.01):**
- Reaches minimum efficiently
- Stable convergence

```
Too large:          Just right:         Too small:
Cost                Cost                Cost
|  ↗↙↗↙            |●                  |●
|  ↗↙↗↙            | ●                 | ●
|  (diverges)       |  ●●●___           |  ●
+──────→ iters      +────────→ iters    |   ●●●●●●
                                        +────────────→ iters
```

**How to choose learning rate:**
- Start with 0.01 (common default)
- Plot loss curve — if loss goes up → too high. If barely decreasing → too low
- Use learning rate schedules — start high, decay over time
- Use adaptive optimisers (Adam, RMSprop) that adjust learning rate automatically

---

## Effect of Loss Function on Gradient Descent

### Convex vs Non-Convex

**Convex function:** Any line connecting two points on the function lies above or on the function. Has exactly **one global minimum**. No local minima problem.
- Example: MSE loss in linear regression → always convex → GD always finds optimal solution

**Non-convex function:** Has multiple local minima and maxima.
- Example: Loss functions in neural networks → non-convex → GD may get stuck in local minima

**The local minima problem:**
If the random starting point is near a local minimum, GD converges there and gets stuck. It has no way to know if there's a better minimum elsewhere.

*Solutions (covered in deep learning):* Random restarts, momentum, Adam optimiser, learning rate schedules.

### Saddle Point Problem

In some loss functions, there are flat regions (plateaux). Since the gradient is nearly zero on a flat surface, each step is tiny — the algorithm takes many iterations to cross it.

In 3D, this plateau is called a **saddle point** — flat in one direction, curved in another. GD slows dramatically here.

*Solution:* Momentum-based optimisers (covered in deep learning).

---

## Effect of Feature Scaling on Gradient Descent

This is one of the most important practical insights.

**With properly scaled features:**
The contour plot of the cost function is roughly **circular**. GD takes a relatively straight path to the minimum — fast convergence.

**Without scaling (features at very different scales):**
The contour plot becomes **elongated/elliptical**. GD takes a zigzag path — inefficient, slow convergence, may need many more epochs.

```
Without scaling:          With scaling:
    ↗↙↗↙                    ↘
      ↗↙↗↙                    ↘
        ★ (minimum)              ★ (minimum)
Zigzag path               Straight path
```

**This is why feature scaling is mandatory before running gradient descent.**

---

## Universality of Gradient Descent

Gradient descent is not specific to linear regression. For **any** algorithm with a differentiable loss function:

1. Define the loss function
2. Compute its derivative (gradient) with respect to parameters
3. Apply update rule: **θ_new = θ_old - α × ∂L/∂θ**

This is why gradient descent is the backbone of all modern ML and deep learning. Logistic regression, neural networks, SVMs — all use it.

The only requirement: the loss function must be **differentiable** (or at least subdifferentiable, as in Lasso).

---

## Complete Algorithm

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    n, p = X.shape
    w = np.zeros(p)    # initialise weights to zero
    b = 0              # initialise bias
    history = []       # track loss

    for epoch in range(epochs):
        # Forward pass
        y_pred = X.dot(w) + b

        # Compute gradients
        error = y_pred - y
        dw = (1/n) * X.T.dot(error)
        db = (1/n) * np.sum(error)

        # Update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Track loss
        loss = (1/(2*n)) * np.sum(error**2)
        history.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return w, b, history
```

---

## Interview One-Liners

**Why Gradient Descent over OLS?**
"OLS requires matrix inversion which is O(n³) — impractical for large datasets. Also fails when XᵀX is singular due to multicollinearity. Gradient Descent is an iterative approximation that scales to any dataset size and works for any differentiable loss function."

**What is the learning rate?**
"Controls the step size at each iteration. Too small → slow convergence. Too large → overshoots, never converges. Typically start with 0.01 and adjust based on the loss curve."

**Batch vs SGD vs Mini-Batch?**
"Batch uses all data per update — smooth but slow. SGD uses one sample — fast but noisy, can escape local minima. Mini-Batch uses a small batch — best of both worlds and the standard choice in deep learning."

**Why must features be scaled before GD?**
"Without scaling, the cost function contour is elongated — GD takes a zigzag path and converges slowly. With scaling, the contour is circular and GD takes a direct path to the minimum."

**What is a saddle point?**
"A flat region in the loss surface where the gradient is nearly zero. GD takes tiny steps here and can seem stuck. Momentum-based optimisers help cross saddle points faster."
