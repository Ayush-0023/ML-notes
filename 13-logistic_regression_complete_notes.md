# Logistic Regression — Complete Notes

---

## Why Logistic Regression?

Linear regression predicts continuous values — salary, house price, marks. But what if you need to predict a category — placed or not placed, spam or not spam, disease or no disease?

That's a **classification problem.** Linear regression breaks down for two reasons:

**Problem 1 — Output goes out of bounds:**
Linear regression can predict -0.3 or 1.7. Probabilities below 0 or above 1 are mathematically meaningless.

**Problem 2 — The relationship isn't linear:**
Going from 0 to 1 hour of study barely changes your chance of passing. Going from 4 to 5 hours at the borderline changes everything. Going from 9 to 10 hours barely matters again. The effect is S-shaped, not linear.

---

## Requirement — Linearly Separable Data

Logistic regression requires that the data is linearly separable — or at least approximately so. This means a line (in 2D) or hyperplane (in higher dimensions) can reasonably separate the two classes.

**Example:** CGPA and IQ as inputs, placement as output. Blue points = not placed, green points = placed. If these can be separated by a line, logistic regression applies.

---

## Step 1 — The Perceptron Trick (Foundation)

Before logistic regression, we need to understand the perceptron algorithm — the simpler predecessor.

### The Line Equation

In logistic regression, the decision boundary is written as:
> **AX₁ + BX₂ + C = 0**

We need to find A, B, and C — the parameters of the line.

### How the Perceptron Algorithm Works

**Setup:**
- 2 input columns (X₁, X₂) and 1 output column (Y)
- Calculate **activation z** for each data point:

> z = w₀ + w₁x₁ + w₂x₂

Or in general form:
> z = w₀x₀ + w₁x₁ + w₂x₂ (where x₀ = 1 always)

**Why w₀ (bias)?**
Without bias, the decision boundary is forced to pass through the origin. If all inputs are zero, output must be zero — which is often wrong. Bias allows the boundary to shift freely. Example: even with zero hours studied, someone might pass by luck. Bias captures this non-zero baseline.

**Prediction rule:**
- If z ≥ 0 → predict 1 (positive class)
- If z < 0 → predict 0 (negative class)

**Algorithm:**
1. Initialise weights randomly
2. For each data point, calculate z and make prediction
3. If correctly classified → do nothing
4. If misclassified → apply perceptron update rule
5. Repeat for fixed epochs (e.g., 1000) or until convergence

### Labelling Regions (+ve and -ve)

Given line AX + BY + C = 0, how do we know which side is positive?

**Method 1 — Simple (put (0,0)):**
Substitute (0,0) into the equation. If result is positive → the region containing origin is positive.
Example: 2x + 3y + 5 = 0 → putting (0,0) gives +5 → right side is positive.

**Method 2 — Geometric:**
Draw a perpendicular from point (A, B) to the line. Stand at the intersection, face toward (A, B). The region you're facing has the same sign as the equation evaluated at (A, B).

### How Changing A, B, C Transforms the Line

| Change | Effect |
|---|---|
| Increase/decrease C | Moves line up or down (parallel shift) |
| Increase A | Rotates line clockwise about y-intercept |
| Decrease A | Rotates line anti-clockwise about y-intercept |
| Increase B | Rotates line anti-clockwise about x-intercept |
| Decrease B | Rotates line clockwise about x-intercept |

### The Perceptron Update Rule — Mathematical Intuition

**Example:** Line is 2x + 3y + 5 = 0. Coefficients: [A=2, B=3, C=5].

**For a misclassified negative point (4,5) in positive region:**
We need to move the line toward (4,5) to push it into the negative region.
Subtract coordinates:
> [A, B, C] - [x₁, x₂, 1] = [2-4, 3-5, 5-1] = [-2, -2, 4]
> New line: -2x - 2y + 4 = 0

**For a misclassified positive point (1,3) in negative region:**
We need to move the line toward (1,3).
Add coordinates:
> [A, B, C] + [x₁, x₂, 1] = [2+1, 3+3, 5+1] = [3, 6, 6]
> New line: 3x + 6y + 6 = 0

**Rule:**
- Negative point in positive region → **subtract**
- Positive point in negative region → **add**

**With learning rate (to prevent abrupt jumps):**
> New [A, B, C] = Old [A, B, C] ± α × [x₁, x₂, 1]

Example with α = 0.01 and misclassified negative point (4,5):
> [2 - 0.01×4, 3 - 0.01×5, 5 - 0.01×1] = [1.96, 2.95, 4.99]

**Generalised Perceptron Update Rule:**
> w_new = w_old + α(y - ŷ)x

Where (y - ŷ) gives +1 or -1 depending on misclassification direction.

---

## Step 2 — Problems with Perceptron

The perceptron algorithm has four critical problems:

1. **Only works on linearly separable data** — fails completely on overlapping/messy data
2. **Doesn't converge on non-separable data** — loops forever without finding a solution
3. **Doesn't find the best fit line** — finds any separating line, not the optimal one
4. **Stops after correct classification** — no optimisation for generalisation

**Root cause:** The perceptron uses a **step function** as its activation. The step function gives only discrete 0 or 1 outputs — no gradients, no smooth optimisation.

---

## Step 3 — Replacing the Step Function with Sigmoid

To fix the perceptron's problems, we replace the step function with the **sigmoid function**.

### The Sigmoid Function

> **σ(z) = 1 / (1 + e⁻ᶻ)**

**Properties:**
- Output always between 0 and 1 — valid probability
- When z → +∞, σ(z) → 1
- When z → -∞, σ(z) → 0
- When z = 0, σ(z) = 0.5

### Where Sigmoid Comes From

Sigmoid doesn't come from nowhere. It falls out naturally from assuming **log odds is linear in x**:

1. Probability p is bounded [0,1] — can't model directly with mx + c
2. Convert to odds: p/(1-p) — range [0, ∞]
3. Take log of odds: log(p/1-p) — range (-∞, +∞) — same as mx + c
4. Assume log odds = mx + c (natural linear assumption)
5. Solve back for p → get sigmoid

Sigmoid is the mathematical consequence of assuming log odds is linear.

### Probabilistic Interpretation

> σ(z) = P(positive class)

- σ(z) = 0.5 → 50% chance of placement (on the boundary)
- σ(z) = 0.7 → 70% chance of placement (clearly positive side)
- σ(z) = 0.3 → 30% chance (clearly negative side)

This creates a probability gradient across the feature space — not just hard 0/1 labels.

### Applying Sigmoid to the Update Rule

With sigmoid, (y - ŷ) no longer gives discrete values. Instead:
- Correctly classified points: small (y - ŷ) → small push away from line
- Misclassified points: large (y - ŷ) → large pull toward line
- The magnitude of push/pull depends on distance from the line

This solves the abrupt/discrete problem of the perceptron. But there's still a problem — this approach doesn't guarantee finding the **optimal** boundary. It just finds any boundary that reduces error, not the best one.

---

## Step 4 — The Loss Function (Binary Cross Entropy)

### Why We Need a Loss Function

Using sigmoid in the update rule is still "perceptron-style" learning — iteratively adjusting based on error, without guaranteeing optimality.

The ML approach: define a **loss function** that quantifies total error, then find coefficients that minimise it. This is principled optimisation, not just error correction.

### Maximum Likelihood Estimation (MLE)

MLE finds parameters that make the observed data most probable.

**For each data point:**
- If actual y = 1 (positive class): probability = σ(z) = P(green)
- If actual y = 0 (negative class): probability = 1 - σ(z) = P(red)

**Likelihood of entire dataset** = product of all individual probabilities:
> L = Π P(yᵢ)

**Example — Two models:**

| Point | Actual | Model 1 P | Model 2 P |
|---|---|---|---|
| P1 | Green | 0.7 | 0.7 |
| P2 | Red | 0.4 | 0.6 |
| P3 | Red | 0.4 | 0.6 |
| P4 | Green | 0.8 | 0.7 |

Model 1 Likelihood = 0.7 × 0.4 × 0.4 × 0.8 = **0.089**
Model 2 Likelihood = 0.7 × 0.6 × 0.6 × 0.7 = **0.176**

Model 2 is better — higher likelihood means observed data is more probable under that model.

### The Log Likelihood Trick

**Problem:** Multiplying many small probabilities creates astronomically tiny numbers — prone to numerical underflow (too small for computer to represent accurately).

**Solution:** Take the log of likelihood. Log turns multiplication into addition:
> log L = Σ log P(yᵢ)

**Problem:** Log of numbers between 0 and 1 is negative. We want to minimise, not maximise a negative number.

**Solution:** Take negative log likelihood:
> -log L = -Σ log P(yᵢ)

This is called **cross-entropy** — the summation of negative logs of maximum likelihood.

Now instead of maximising likelihood, we **minimise cross-entropy.**

### Binary Cross Entropy — The Final Loss Function

Combining both cases (y=1 and y=0) into one elegant formula:

> **L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]**

Check:
- When y=1: L = -log(ŷ) → penalises heavily if ŷ is close to 0
- When y=0: L = -log(1-ŷ) → penalises heavily if ŷ is close to 1

**Average over all data points (Cost Function):**

> **J = -(1/n) Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]**

This is also called **log-loss** or **binary cross-entropy loss**.

### Why Not Squared Errors for Logistic Regression?

When you plug sigmoid into squared errors, the cost function becomes **non-convex** — bumpy with multiple local minima. Gradient descent gets stuck in a fake minimum and never finds the true optimal weights.

Binary cross-entropy with sigmoid is **convex** — one clean global minimum. Gradient descent always finds the optimal solution.

**The deeper reason:** Binary cross-entropy comes from MLE under Bernoulli distribution (binary outputs). Squared errors come from MLE under Gaussian distribution (continuous outputs). Different output types → different noise assumptions → different loss functions.

### Intuition — Why Cross Entropy Punishes Confident Wrong Predictions

| Actual | Predicted | Penalty |
|---|---|---|
| 1 | 0.99 (confident correct) | -log(0.99) ≈ 0.01 (tiny) |
| 1 | 0.01 (confident wrong) | -log(0.01) ≈ 4.6 (huge) |
| 0 | 0.01 (confident correct) | -log(0.99) ≈ 0.01 (tiny) |
| 0 | 0.99 (confident wrong) | -log(0.01) ≈ 4.6 (huge) |

The further your confidence is from the truth, the more you pay.

---

## Step 5 — Gradient Descent

Binary cross-entropy has no closed form solution — unlike OLS for linear regression. The absolute value and log functions make analytical solving impossible.

**Solution:** Gradient Descent.

The gradient of binary cross-entropy with respect to weights turns out surprisingly clean:

> ∂J/∂w = (1/n) Σ (ŷᵢ - yᵢ) xᵢ
> ∂J/∂b = (1/n) Σ (ŷᵢ - yᵢ)

**Update rules:**
> w_new = w_old - α × ∂J/∂w
> b_new = b_old - α × ∂J/∂b

This is the same form as linear regression's gradient — because both are MLE under exponential family distributions.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))

def logistic_regression_gd(X, y, lr=0.01, epochs=1000):
    n, p = X.shape
    w = np.zeros(p)
    b = 0
    history = []

    for epoch in range(epochs):
        # Forward pass
        z = X.dot(w) + b
        y_pred = sigmoid(z)

        # Compute gradients
        error = y_pred - y
        dw = (1/n) * X.T.dot(error)
        db = (1/n) * np.sum(error)

        # Update weights
        w = w - lr * dw
        b = b - lr * db

        # Track loss
        loss = binary_cross_entropy(y, y_pred)
        history.append(loss)

    return w, b, history
```

---

## Step 6 — Making Predictions (Threshold)

After training, the model outputs probabilities. You need to convert them to class labels.

**Default threshold = 0.5:**
- If σ(z) ≥ 0.5 → predict 1
- If σ(z) < 0.5 → predict 0

**But 0.5 isn't always right.** The threshold is a **business decision:**

| Application | Strategy | Threshold |
|---|---|---|
| Disease detection | Minimise false negatives (missing sick people is dangerous) | Lower (e.g., 0.3) |
| Spam detection | Minimise false positives (blocking real email is worse) | Higher (e.g., 0.7) |
| Fraud detection | Minimise false negatives (missing fraud is costly) | Lower |

---

## sklearn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Always scale before logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    C=1.0,              # C = 1/λ — inverse of regularisation strength
    penalty='l2',       # 'l1', 'l2', 'elasticnet', None
    solver='lbfgs',     # optimisation algorithm
    max_iter=1000       # maximum iterations for convergence
)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)           # class labels (0 or 1)
y_prob = model.predict_proba(X_test_scaled)     # probabilities [[P(0), P(1)], ...]

# Custom threshold
threshold = 0.3
y_pred_custom = (y_prob[:, 1] >= threshold).astype(int)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Note on C parameter:** sklearn uses C = 1/λ. Higher C = less regularisation. Lower C = more regularisation. This is the inverse of the λ convention in the loss function.

---

## Perceptron vs Logistic Regression — Full Comparison

| | Perceptron | Logistic Regression |
|---|---|---|
| **Activation** | Step function (hard 0/1) | Sigmoid (smooth probability) |
| **Output** | 0 or 1 directly | Probability between 0 and 1 |
| **Loss function** | None — just error correction | Binary cross-entropy |
| **Optimisation** | Update rule (perceptron trick) | Gradient descent on loss |
| **Works on non-separable data** | No — loops forever | Yes — finds best possible boundary |
| **Finds optimal boundary** | No — any separating line | Yes — minimises log-loss |
| **Probabilities** | No | Yes |
| **Convergence guarantee** | Only if linearly separable | Always converges (convex loss) |

---

## The Complete Flow

```
Binary classification problem
        ↓
Data should be linearly separable
        ↓
Perceptron finds any separating line (not optimal)
        ↓
Replace step function with sigmoid → smooth probabilities
        ↓
Still not optimal → need loss function
        ↓
MLE under Bernoulli → Binary Cross Entropy
        ↓
Minimise with Gradient Descent (no closed form)
        ↓
Apply threshold (default 0.5) → class prediction
        ↓
Evaluate with precision, recall, F1, AUC-ROC
```

---

## Interview One-Liners

**What is logistic regression?**
"A binary classification algorithm that models the probability of a class using the sigmoid function. It finds the decision boundary by minimising binary cross-entropy loss via gradient descent."

**Why sigmoid and not something else?**
"Sigmoid falls naturally from assuming log odds is linear in x. It's the only function that maps the linear combination mx+c to a valid probability between 0 and 1 while maintaining that linearity in log odds space."

**Why binary cross-entropy and not squared errors?**
"Squared errors + sigmoid creates a non-convex loss function with multiple local minima — gradient descent gets stuck. Binary cross-entropy is convex. Also, cross-entropy comes from MLE under Bernoulli distribution — the correct statistical assumption for binary outputs."

**What is the relationship between perceptron and logistic regression?**
"Both use the same linear structure w₀ + w₁x₁ + w₂x₂. Perceptron applies a hard threshold and updates weights only on mistakes — no probability, no loss function, only works on separable data. Logistic regression applies sigmoid for probabilities and minimises cross-entropy — works on any data and finds the optimal boundary."

**What is MLE in logistic regression?**
"We find weights that maximise the probability of observing the actual labels given the model. For binary outcomes, this means multiplying P(yᵢ) across all points — which after taking log and negating — gives binary cross-entropy. Minimising cross-entropy is identical to maximising likelihood."
