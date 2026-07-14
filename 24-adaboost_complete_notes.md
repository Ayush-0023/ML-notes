# AdaBoost (Adaptive Boosting) — Complete Notes

---

## What is AdaBoost?

AdaBoost (Adaptive Boosting) is the first practical boosting algorithm. Unlike Bagging/Random Forest where models are trained independently and in parallel, AdaBoost trains models **sequentially** — each new model focuses specifically on correcting the mistakes of the previous models.

**Core idea:** Combine many "weak learners" (models slightly better than random guessing) into one "strong learner" by having each subsequent model pay more attention to the data points that previous models got wrong.

**Where it fits in the ensemble landscape:**
- Bagging reduces **variance** (works on high-variance, low-bias base models like deep trees)
- Boosting reduces **bias** (works on high-bias, low-variance base models like shallow trees/stumps)

---

## The Weak Learner — Decision Stumps

AdaBoost typically uses **decision stumps** as base learners — a decision tree with only ONE split (depth=1). A stump is barely better than random guessing — it's deliberately weak.

```
        Root
       /    \
   Leaf 1   Leaf 2
```

**Why deliberately weak learners?** Boosting's power comes from combining many weak learners intelligently, not from using strong learners. Using strong learners (deep trees) would cause severe overfitting since boosting already reduces bias aggressively.

---

## Geometric Intuition

Imagine a dataset with two classes that aren't perfectly separable by a single line.

**Round 1:** Train the first stump. It draws a simple boundary, getting most points right but misclassifying a few.

**Round 2:** The misclassified points from Round 1 get **higher weight** (more importance). The next stump is forced to pay more attention to these difficult points — it will draw a different boundary that better handles the previously misclassified points (possibly at the cost of others).

**Round 3 onwards:** This process repeats — each stump focuses on whatever the ensemble so far has gotten wrong.

**Final prediction:** Weighted combination of all stumps — stumps that performed better get more say in the final vote.

```
Round 1: ●  ●  ✗(wrong)  ●  ●        ← stump 1 boundary
Round 2: ●  ●  ✓(now correct, was weighted higher)  ✗(new mistake)  ●  ← stump 2 boundary
Round 3: focuses on stump 2's mistake
...
Final: weighted vote of all stumps
```

---

## Step-by-Step Algorithm

### Step 1 — Initialise Sample Weights

Every training point starts with equal weight:
> wᵢ = 1/n  (for all n samples)

### Step 2 — Train a Weak Learner

Train a decision stump on the (weighted) data. It finds the single best split given current weights.

### Step 3 — Calculate the Stump's Error Rate

> **Error (ε) = Σ (wᵢ for misclassified points) / Σ (all wᵢ)**

This is the weighted error rate — not just a simple count, but weighted by how much each point currently matters.

### Step 4 — Calculate the Stump's Say (Alpha / Performance Weight)

Each stump gets a "say" in the final vote based on how well it performed:

> **α = (1/2) × ln((1-ε)/ε)**

**Key properties of this formula:**
- If ε = 0.5 (random guessing) → α = 0 → this stump gets zero influence in the final vote
- If ε < 0.5 (better than random) → α > 0 → positive influence, more accurate stump = larger α
- If ε > 0.5 (worse than random) → α < 0 → negative influence (effectively flips its vote — being consistently wrong is itself informative!)
- If ε → 0 (perfect stump) → α → ∞ (extremely high influence)

```
ε = 0.1  → α = 0.5×ln(0.9/0.1) = 0.5×ln(9) ≈ 1.10  (high influence)
ε = 0.3  → α = 0.5×ln(0.7/0.3) = 0.5×ln(2.33) ≈ 0.42  (moderate influence)
ε = 0.5  → α = 0.5×ln(1) = 0  (zero influence)
ε = 0.7  → α = 0.5×ln(0.3/0.7) ≈ -0.42  (negative influence)
```

### Step 5 — Update Sample Weights

Increase weights of misclassified points, decrease weights of correctly classified points:

**For misclassified points:**
> w_new = w_old × e^(α)

**For correctly classified points:**
> w_new = w_old × e^(-α)

**Combined formula (often written as):**
> w_new = w_old × e^(-α × yᵢ × h(xᵢ))

Where yᵢ is the true label (+1/-1) and h(xᵢ) is the stump's prediction (+1/-1). If they match (correct), the exponent is negative → weight decreases. If they don't match (incorrect), the exponent is positive → weight increases.

### Step 6 — Normalise Weights

After updating, normalise so all weights sum to 1 again:
> wᵢ = wᵢ / Σwᵢ

### Step 7 — Repeat

Go back to Step 2. Train the next stump on the data with UPDATED weights (misclassified points now have higher weight, so the next stump is incentivised to get them right).

### Step 8 — Final Prediction

Combine all stumps using their α values as weights:

> **H(x) = sign(Σₜ αₜ × hₜ(x))**

Each stump votes ±1 (for binary classification), weighted by its α. The sign of the weighted sum gives the final class prediction.

---

## Worked Numerical Example

**Setup:** 5 data points, binary classification (+1/-1). Initial weight for each = 1/5 = 0.2.

**Round 1:**
- Train stump 1. Suppose it misclassifies 1 out of 5 points.
- Error ε₁ = 0.2 (weight of that 1 misclassified point) / 1.0 (total weight) = 0.2
- α₁ = 0.5 × ln(0.8/0.2) = 0.5 × ln(4) ≈ 0.693

**Update weights:**
- Misclassified point: w_new = 0.2 × e^(0.693) ≈ 0.2 × 2.0 = 0.4
- Correctly classified points (4 of them): w_new = 0.2 × e^(-0.693) ≈ 0.2 × 0.5 = 0.1 each

**Normalise:** Total = 0.4 + 4×0.1 = 0.8. Divide each by 0.8:
- Misclassified point: 0.4/0.8 = 0.5
- Each correct point: 0.1/0.8 = 0.125

Now the previously misclassified point has 50% of the total weight — the next stump MUST pay attention to it or risk a huge error penalty.

**Round 2:** Train stump 2 on this re-weighted data. It will likely get the previously-misclassified point right (since it now dominates the weighted error calculation), possibly at the cost of misclassifying a different point.

**This continues** for a set number of rounds (n_estimators), with each stump specialising in fixing the ensemble's current weak spots.

---

## Code Implementation

### From Scratch (Simplified)

```python
import numpy as np

class SimpleAdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, 1/n_samples)  # initial equal weights

        for t in range(self.n_estimators):
            # Train weak learner (decision stump) with sample weights
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)
            pred = stump.predict(X)

            # Calculate weighted error
            incorrect = (pred != y)
            error = np.sum(w[incorrect]) / np.sum(w)
            error = np.clip(error, 1e-10, 1 - 1e-10)  # avoid log(0)

            # Calculate alpha (stump's say)
            alpha = 0.5 * np.log((1 - error) / error)

            # Update weights
            w = w * np.exp(-alpha * y * pred)
            w = w / np.sum(w)  # normalise

            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        weighted_sum = np.dot(self.alphas, stump_preds)
        return np.sign(weighted_sum)
```

### Using sklearn

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # decision stump
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME',  # or 'SAMME.R' (deprecated in newer versions)
    random_state=42
)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Access individual stump weights (alphas)
print(f"Estimator weights: {ada.estimator_weights_}")
print(f"Estimator errors: {ada.estimator_errors_}")
```

### AdaBoost Regressor

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada_reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=50,
    learning_rate=1.0,
    loss='linear',  # 'linear', 'square', 'exponential'
    random_state=42
)
ada_reg.fit(X_train, y_train)
y_pred = ada_reg.predict(X_test)
```

---

## AdaBoost Hyperparameters

| Parameter | Meaning | Effect |
|---|---|---|
| `estimator` | Base weak learner | Default is decision stump (max_depth=1) |
| `n_estimators` | Number of boosting rounds | More rounds → more complex model, risk of overfitting if too high |
| `learning_rate` | Shrinks each stump's contribution (α × learning_rate) | Lower → need more estimators but better generalisation; higher → faster but risk of overfitting |
| `algorithm` | 'SAMME' (discrete) or 'SAMME.R' (real, uses probabilities) | SAMME.R generally converges faster |

### The learning_rate Parameter — Important Detail

The final prediction formula becomes:
> H(x) = sign(Σₜ (learning_rate × αₜ) × hₜ(x))

**Lower learning_rate (e.g., 0.1):**
- Each stump's influence is dampened
- Need MORE estimators to achieve the same overall effect
- Generally improves generalisation — classic bias-variance tradeoff via shrinkage

**Higher learning_rate (e.g., 1.0, the default):**
- Each stump has full influence immediately
- Converges faster but can overfit with too many estimators

### Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3]  # tuning the base stump's depth too
}

grid = GridSearchCV(
    AdaBoostClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

**Key tuning insight:** `n_estimators` and `learning_rate` work together — there's a tradeoff. Lower learning_rate typically needs higher n_estimators to compensate. Tune them jointly, not independently.

---

## AdaBoost's Sensitivity to Outliers and Noise

Because AdaBoost progressively increases the weight of misclassified points, **outliers and mislabeled data get disproportionately high weight** over successive rounds. The algorithm essentially obsesses over points it can't classify correctly — which is often a sign of noise, not signal.

**This is AdaBoost's biggest weakness** — it's notably more sensitive to noisy data and outliers compared to Random Forest or Gradient Boosting (which uses a more robust gradient-based weighting mechanism).

**Mitigation:** Clean your data well before AdaBoost, or use Gradient Boosting / XGBoost which handle this more gracefully.

---

## AdaBoost vs Random Forest

| | AdaBoost | Random Forest |
|---|---|---|
| **Training** | Sequential | Parallel |
| **Base learner** | Weak (decision stump) | Strong (deep tree) |
| **Targets** | Bias reduction | Variance reduction |
| **Sample weighting** | Yes — misclassified points weighted up | No — uniform bootstrap sampling |
| **Final combination** | Weighted vote (by α) | Simple majority vote/average |
| **Outlier sensitivity** | High | Low |
| **Overfitting risk** | Can overfit with too many estimators | Generally robust to overfitting |
| **Parallelisable** | No (sequential dependency) | Yes |

---

## The Complete AdaBoost Flow

```
Initialise equal weights for all samples
        ↓
Train weak learner (stump) on weighted data
        ↓
Calculate weighted error (ε) of this stump
        ↓
Calculate stump's say (α) — higher accuracy = higher α
        ↓
Increase weight of misclassified points
Decrease weight of correctly classified points
        ↓
Normalise weights
        ↓
Repeat for n_estimators rounds
        ↓
Final prediction = weighted vote of all stumps (weighted by α)
```

---

## Interview One-Liners

**What is AdaBoost?**
"A sequential boosting algorithm that combines weak learners (typically decision stumps) into a strong learner. Each new learner focuses on the mistakes of the previous ensemble by increasing the weight of misclassified points."

**How is a stump's influence (alpha) determined?**
"α = 0.5 × ln((1-ε)/ε) where ε is the weighted error rate. Better stumps (lower error) get exponentially higher influence in the final vote. A stump with 50% error (random guessing) gets zero influence."

**Why does AdaBoost use weak learners instead of strong ones?**
"Boosting already aggressively reduces bias by focusing on hard examples across rounds. Using strong learners (deep trees) on top of this sequential error-correction would cause severe overfitting. Weak learners combined intelligently is the whole point of boosting."

**AdaBoost vs Random Forest?**
"Random Forest trains trees in parallel on bootstrap samples to reduce variance — works well with high-variance deep trees. AdaBoost trains stumps sequentially, reweighting misclassified points each round, to reduce bias — works well with high-bias shallow stumps."

**Why is AdaBoost sensitive to outliers?**
"Misclassified points get exponentially increasing weight each round. If a point is an outlier or mislabeled, it can never be classified correctly — so its weight keeps growing across rounds, eventually distorting the model's focus entirely toward that bad point."

**What does learning_rate control in AdaBoost?**
"It scales down each stump's contribution to the final vote. Lower learning rate requires more estimators but generally improves generalisation — a shrinkage technique to control overfitting, similar in spirit to regularisation."
