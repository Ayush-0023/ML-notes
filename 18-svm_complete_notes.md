# Support Vector Machines (SVM) — Complete Notes

---

## Intuition — SVM as an Extension of Logistic Regression

In logistic regression, we calculate the perpendicular distance between the decision boundary (line/hyperplane) and a data point:

> z = wᵀx + b

This z is passed through sigmoid to get P(y=1).

- Point close to the line → z near 0 → probability near 0.5 (uncertain)
- Point far from the line → z large → probability near 0 or 1 (confident)

**The key insight SVM builds on:** Distance from the boundary correlates with confidence.

**What SVM does differently:** Instead of just finding *any* separating line, SVM finds the line that maximises the distance to the nearest points on both sides. It directly optimises for maximum confidence/separation, not just correct classification.

---

## Geometric Intuition

**Step 1 — Start with a random hyperplane π** that separates the two classes.

**Step 2 — Push the hyperplane in the positive direction** (parallel shift) until it touches the nearest positive (green) point. This becomes **π+**.

**Step 3 — Push the hyperplane in the negative direction** until it touches the nearest negative (red) point. This becomes **π-**.

**Step 4 — Calculate the margin (d)** — the distance between π+ and π-.

**Step 5 — Repeat for different random hyperplanes**, and select the one that gives the **maximum margin d**.

```
        ●  ●  (π+)
      ●    ●
- - - - - - - - - -  (π, the actual decision boundary)
      ○    ○
        ○  ○  (π-)

        ←——d——→  (margin)
```

**The optimisation goal:** Find w and b in (wᵀx + b) such that the margin d is maximised.

---

## Support Vectors

The points where π+ and π- touch the data — i.e., the closest points to the decision boundary on each side — are called **Support Vectors**.

**Why they matter:**
- Only these points determine the position of the decision boundary
- All other points (further from the boundary) could be removed without changing the model at all
- This is why SVM is memory-efficient at prediction time — it only needs to remember the support vectors, not the entire dataset

---

## Hard Margin SVM — Mathematical Formulation

### Setup

Assume the data is **perfectly linearly separable** (no overlap, no noise). We want to find the hyperplane:

> wᵀx + b = 0

Such that:
- For positive class (y=1): wᵀx + b ≥ 1
- For negative class (y=-1): wᵀx + b ≤ -1

(Note: SVM conventionally uses y ∈ {-1, +1}, not {0, 1})

Combined into a single constraint:
> **yᵢ(wᵀxᵢ + b) ≥ 1** for all training points i

### The Margin

The two boundary hyperplanes are:
> π+ : wᵀx + b = 1
> π- : wᵀx + b = -1

The distance between them (the margin) is:
> **margin = 2 / ||w||**

**Derivation intuition:** The perpendicular distance from origin to a hyperplane wᵀx+b=c is |c|/||w||. So distance from π+ to π is 1/||w||, and from π- to π is also 1/||w||. Total margin = 2/||w||.

### The Optimisation Problem

We want to **maximise** the margin:
> maximise 2/||w||

Which is equivalent to:
> **minimise (1/2)||w||²**

Subject to:
> yᵢ(wᵀxᵢ + b) ≥ 1 for all i

**Why minimise ||w||² instead of maximising 2/||w||?**
- Maximising 2/||w|| is the same as minimising ||w||
- We use ||w||² instead of ||w|| because it's differentiable everywhere (no square root) — easier optimisation
- The 1/2 is just for convenience when taking derivatives (cancels the 2 from the square)

### Solving with Lagrange Multipliers

This is a constrained optimisation problem. We introduce Lagrange multipliers αᵢ ≥ 0 for each constraint:

> L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1]

Taking partial derivatives and setting to zero:

> ∂L/∂w = w - Σᵢ αᵢyᵢxᵢ = 0  →  **w = Σᵢ αᵢyᵢxᵢ**
> ∂L/∂b = -Σᵢ αᵢyᵢ = 0  →  **Σᵢ αᵢyᵢ = 0**

**Key insight:** w is expressed as a weighted combination of training points, where the weights are αᵢyᵢ. Points with αᵢ = 0 don't contribute to w at all — only points with αᵢ > 0 matter. **These are exactly the support vectors.**

This transforms into the **dual problem** — maximise with respect to α:
> maximise Σᵢαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)
> subject to αᵢ ≥ 0 and Σᵢαᵢyᵢ = 0

**Why solve the dual instead of the primal?** The dual problem only involves dot products xᵢᵀxⱼ between data points — this is exactly what makes the Kernel Trick possible (explained below).

---

## Soft Margin SVM — Handling Non-Separable Data

### The Problem

Hard Margin SVM requires perfect linear separability. Real data almost always has noise, outliers, or overlapping classes. Hard margin SVM either fails to converge or produces a terrible boundary trying to accommodate one noisy point.

### The Fix — Slack Variables

Introduce a slack variable ξᵢ ≥ 0 for each point, allowing some points to violate the margin (or even be misclassified):

> yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ

- ξᵢ = 0 → point is correctly classified and outside the margin (no violation)
- 0 < ξᵢ < 1 → point is inside the margin but still correctly classified
- ξᵢ ≥ 1 → point is misclassified

### The New Optimisation Problem

> minimise (1/2)||w||² + C·Σᵢξᵢ

Subject to:
> yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ and ξᵢ ≥ 0

**C is the regularisation hyperparameter** — controls the tradeoff between maximising margin and minimising classification errors.

| C value | Effect | Result |
|---|---|---|
| **Large C** | Heavily penalises misclassification | Narrow margin, fits training data closely → risk of overfitting |
| **Small C** | Tolerates more misclassification | Wide margin, more tolerant of errors → risk of underfitting |

This is the bias-variance tradeoff applied to SVM. C is analogous to 1/λ in regularised regression.

```python
from sklearn.svm import SVC

svm_large_c = SVC(kernel='linear', C=100)   # narrow margin, overfitting risk
svm_small_c = SVC(kernel='linear', C=0.01)  # wide margin, underfitting risk
```

---

## The Kernel Trick — Handling Non-Linear Data

### The Problem

Hard/Soft Margin SVM finds a **linear** decision boundary. But many real datasets are not linearly separable — like concentric circles where one class surrounds another.

```
        ○ ○ ○ ○ ○
      ○  ● ● ● ●  ○
     ○  ●       ●  ○
      ○  ● ● ● ●  ○
        ○ ○ ○ ○ ○
```

No straight line can separate the inner class (●) from the outer ring (○).

### The Fix — Map to Higher Dimensions

If data isn't linearly separable in its current dimension, **transform it into a higher dimension** where it becomes linearly separable.

**Example:** For the concentric circles problem, adding a third dimension z = x² + y² transforms the data. In 3D, the inner circle (small radius) sits lower, the outer circle (large radius) sits higher — now a flat plane CAN separate them.

### Why We Don't Actually Compute the Transformation

Explicitly transforming every point into higher dimensions is computationally expensive — especially since the dual SVM problem only needs **dot products** xᵢᵀxⱼ, not the raw transformed vectors.

**The Kernel Trick:** Define a kernel function K(xᵢ, xⱼ) that computes the dot product of the transformed vectors **without actually performing the transformation.**

> K(xᵢ, xⱼ) = φ(xᵢ)ᵀφ(xⱼ)

Where φ is the (implicit) transformation to higher dimensions. We compute the result of the dot product in higher dimensional space directly from the original lower-dimensional vectors — without ever explicitly constructing φ(x).

This is why it's called a "trick" — we get the benefit of higher dimensions without the computational cost of actually going there.

### Common Kernel Functions

**1. Linear Kernel (no transformation)**
> K(xᵢ, xⱼ) = xᵢᵀxⱼ

Use when data is already linearly separable.

**2. Polynomial Kernel**
> K(xᵢ, xⱼ) = (xᵢᵀxⱼ + c)^d

Maps to polynomial feature space of degree d. Captures interactions between features.

**3. RBF Kernel (Radial Basis Function / Gaussian Kernel)**
> K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)

Most commonly used. Maps to an infinite-dimensional space. Measures similarity based on distance — closer points are more similar.

- γ (gamma) controls how far the influence of a single point reaches
- High γ → narrow influence → complex, wiggly boundary → overfitting risk
- Low γ → wide influence → smooth boundary → underfitting risk

**4. Sigmoid Kernel**
> K(xᵢ, xⱼ) = tanh(αxᵢᵀxⱼ + c)

Behaves similarly to a neural network activation. Less commonly used.

---

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

# Generate non-linear data (concentric circles)
X, y = make_circles(100, factor=0.1, noise=0.1)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='bwr')
plt.title('Non-linearly separable data')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Linear kernel — will perform poorly on circular data
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
y_pred = linear_svm.predict(X_test)
print(f"Linear kernel accuracy: {accuracy_score(y_test, y_pred):.4f}")

# RBF kernel — should perform well
rbf_svm = SVC(kernel='rbf', gamma='scale')
rbf_svm.fit(X_train, y_train)
y_pred = rbf_svm.predict(X_test)
print(f"RBF kernel accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Polynomial kernel
poly_svm = SVC(kernel='poly', degree=2)
poly_svm.fit(X_train, y_train)
y_pred = poly_svm.predict(X_test)
print(f"Polynomial kernel accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

**Expected result:** Linear kernel performs poorly (~50%, random) on circular data. RBF and Polynomial kernels perform much better since they can capture the circular boundary.

---

## Choosing C and Gamma — Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
```

**The C-gamma interaction for RBF kernel:**

| C | gamma | Result |
|---|---|---|
| Low | Low | Very smooth, simple boundary → underfitting |
| Low | High | Margin tries to be wide but boundary still wiggly → unstable |
| High | Low | Smooth boundary but fits closely → balanced |
| High | High | Complex, wiggly boundary → overfitting |

---

## Always Scale Before SVM

Like KNN, SVM relies on distances (especially with RBF kernel). Features at different scales distort the margin calculation.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## SVM for Regression — SVR

SVM's concepts extend to regression: instead of finding a margin that separates classes, SVR finds a boundary (tube) that contains most points within a margin of error ε.

```python
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)
```

---

## Advantages and Disadvantages

**Advantages:**
- Effective in high-dimensional spaces
- Works well even when number of features > number of samples
- Memory efficient — only stores support vectors, not entire dataset
- Versatile — different kernels handle different data shapes
- Robust to overfitting in high dimensions (with proper C)

**Disadvantages:**
- Slow training on large datasets — O(n²) to O(n³) depending on implementation
- Choosing the right kernel and hyperparameters requires experimentation
- Doesn't directly provide probability estimates (requires extra calibration — `probability=True`)
- Less interpretable than logistic regression — no simple coefficient interpretation
- Sensitive to feature scaling

---

## Complete Comparison — SVM vs Logistic Regression vs KNN

| | SVM | Logistic Regression | KNN |
|---|---|---|---|
| **Decision boundary** | Maximum margin hyperplane | Any separating hyperplane (MLE-optimal) | Implicit, based on neighbours |
| **Non-linear data** | Yes (kernel trick) | Only with polynomial features | Yes (naturally) |
| **Training speed** | Slow on large data | Fast | Instant (lazy) |
| **Prediction speed** | Fast (uses support vectors only) | Fast | Slow on large data |
| **Probability output** | Not by default | Yes (natural) | Yes (vote proportion) |
| **Interpretability** | Low | High | Low |
| **Outlier sensitivity** | Moderate (soft margin helps) | Moderate | High |

---

## Interview One-Liners

**What is SVM?**
"A classification algorithm that finds the hyperplane maximising the margin between classes. Unlike logistic regression which finds any separating boundary, SVM specifically optimises for maximum separation — using only the closest points (support vectors) to define the boundary."

**What are support vectors?**
"The training points closest to the decision boundary that determine its position. Removing any other point doesn't change the model at all — only support vectors matter."

**Hard margin vs Soft margin?**
"Hard margin requires perfect linear separability with no tolerance for misclassification — fails on noisy data. Soft margin introduces slack variables and a C parameter that allows some misclassification, trading margin width for tolerance of errors — practical for real-world data."

**What is the kernel trick?**
"A technique that computes dot products in a higher-dimensional space without explicitly transforming the data into that space. This lets SVM find non-linear boundaries efficiently — the dual optimisation problem only needs dot products, which kernels compute directly."

**RBF vs Linear vs Polynomial kernel?**
"Linear for already separable data. Polynomial captures feature interactions up to a chosen degree. RBF (most common) measures similarity based on distance and can model arbitrarily complex boundaries — controlled by the gamma parameter."

**What does C control?**
"The tradeoff between margin width and misclassification tolerance. High C → narrow margin, fits training data closely, overfitting risk. Low C → wide margin, more tolerant of errors, underfitting risk. Equivalent to inverse regularisation strength."
