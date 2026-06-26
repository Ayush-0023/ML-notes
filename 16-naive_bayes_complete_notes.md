# Naive Bayes — Complete Notes

---

## Probability Foundations

### Conditional Probability

The probability of event A occurring **given** that event B has already occurred:

> **P(A|B) = P(A ∩ B) / P(B)** ; where P(B) ≠ 0

**Intuition:** You're restricting your universe to only cases where B happened, then asking how often A also happens within that restricted universe.

---

### Independent Events

Two events A and B are independent when:

> **P(A ∩ B) = P(A) × P(B)**

This means the occurrence of B has no effect on A and vice versa.

**Example:** Tossing a coin and rolling a die — the result of one doesn't affect the other.

**Implication for conditional probability:**
If A and B are independent:
> P(A|B) = P(A ∩ B) / P(B) = P(A)×P(B) / P(B) = P(A)

Knowing B happened tells you nothing about A.

---

### Mutually Exclusive Events

Two events A and B are mutually exclusive when:

> **P(A ∩ B) = 0**

Both events cannot occur simultaneously.

**Example:** Taking a left turn and taking a right turn at the same intersection — impossible to do both.

**Key distinction from independence:**
- Independent events CAN both occur (they just don't influence each other)
- Mutually exclusive events CANNOT both occur

If A and B are mutually exclusive: P(A|B) = 0 (if B happened, A definitely didn't)

---

### Bayes Theorem

For two events A and B:

> **P(A|B) = P(B|A) × P(A) / P(B)** ; where P(B) ≠ 0

**The four components:**

| Term | Name | Meaning |
|---|---|---|
| P(A\|B) | **Posterior** | What we want — probability of A given we observed B |
| P(B\|A) | **Likelihood** | How probable is B given A is true |
| P(A) | **Prior** | Our initial belief about A before seeing B |
| P(B) | **Evidence** | Total probability of observing B |

**Intuition:** Bayes theorem lets you update your belief about A (prior) after observing new evidence B, to get a refined belief (posterior).

---

### Solved Problem — Bayes Theorem

**Question:** A factory has 3 machines (M1, M2, M3) making markers.
- M1 makes 20%, M2 makes 30%, M3 makes 50%
- Defect rates: M1=5%, M2=3%, M3=1%

If a randomly picked marker is defective, what's the probability it came from M3?

**Given:**
- P(M1) = 0.20, P(M2) = 0.30, P(M3) = 0.50
- P(D|M1) = 0.05, P(D|M2) = 0.03, P(D|M3) = 0.01

**Find:** P(M3|D)

**Step 1 — Apply Bayes Theorem:**
> P(M3|D) = P(D|M3) × P(M3) / P(D)

**Step 2 — Find P(D) using Total Probability:**
> P(D) = P(D|M1)×P(M1) + P(D|M2)×P(M2) + P(D|M3)×P(M3)
> P(D) = (0.05×0.20) + (0.03×0.30) + (0.01×0.50)
> P(D) = 0.010 + 0.009 + 0.005 = **0.024**

**Step 3 — Calculate P(M3|D):**
> P(M3|D) = (0.01 × 0.50) / 0.024 = 0.005 / 0.024 = **0.208 ≈ 20.8%**

**Interpretation:** Even though M3 makes the most markers (50%), its very low defect rate (1%) means only 20.8% of defective markers come from it. M1 produces far fewer markers but has a high defect rate — contributing disproportionately to defectives.

---

## Naive Bayes Algorithm

### The Core Idea

Naive Bayes is a classification algorithm based on Bayes Theorem. Given a set of features, it predicts the most probable class.

**The "Naive" assumption — Conditional Independence:**
Naive Bayes assumes all features are **conditionally independent** given the class label.

This means: given that we know the class, knowing the value of one feature tells us nothing about any other feature.

**Example:** Given we know it's a spam email, knowing it contains the word "money" tells us nothing about whether it also contains "free" — they're treated as independent given the class.

This assumption is almost never true in reality — that's why it's called "naive." Yet despite this, Naive Bayes works surprisingly well in practice.

---

### Mathematical Foundation

**Goal:** Given features X = (x₁, x₂, ..., xₙ), find the class C that maximises P(C|X).

**Using Bayes Theorem:**
> P(C|X) = P(X|C) × P(C) / P(X)

**Key insight:** P(X) is the same for all classes — we can ignore it for comparison purposes. We only need to maximise the numerator:

> P(C|X) ∝ P(X|C) × P(C)

**Applying the Naive (Conditional Independence) Assumption:**
> P(X|C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)

So:
> **P(C|X) ∝ P(C) × Π P(xᵢ|C)**

This is the key equation. For each class, multiply the prior by the likelihood of each feature given that class.

---

### Maximum A Posteriori (MAP) Rule

**MAP:** Choose the class with the highest posterior probability.

> **ŷ = argmax_C [P(C) × Π P(xᵢ|C)]**

For binary classification (Yes/No):
- Compute P(Yes) × Π P(xᵢ|Yes)
- Compute P(No) × Π P(xᵢ|No)
- Predict whichever is larger

---

### Worked Example — Play Tennis

**Dataset features:** Outlook, Temperature, Humidity, Wind → Play (Yes/No)

**Training phase:** Build probability lookup table from training data.

```python
import pandas as pd

data = pd.read_csv('play_tennis.csv')
data.drop(columns=['day'], inplace=True)

# Build lookup table

# Prior probabilities
p_yes = (data['play'] == 'Yes').mean()  # P(Yes)
p_no  = (data['play'] == 'No').mean()   # P(No)

# Conditional probabilities P(feature=value | class)
# Example: P(Sunny | Yes)
p_sunny_yes = ((data['outlook']=='Sunny') & (data['play']=='Yes')).sum() / (data['play']=='Yes').sum()
```

**Problem 1:** Outlook=Sunny, Temp=Hot, Humidity=High, Wind=Weak → Play?

> P(Yes|features) ∝ P(Yes) × P(Sunny|Yes) × P(Hot|Yes) × P(High|Yes) × P(Weak|Yes)
> P(No|features)  ∝ P(No)  × P(Sunny|No)  × P(Hot|No)  × P(High|No)  × P(Weak|No)

Compare → predict whichever is higher (MAP rule).

---

### How Naive Bayes Actually Works — Training vs Testing

**Training Phase:**
Naive Bayes doesn't "train" in the traditional sense (no gradient descent, no iteration). It simply:
1. Counts occurrences of each feature value for each class
2. Builds a probability lookup table (dictionary)
3. Stores all conditional probabilities P(xᵢ|C) and priors P(C)

**Testing Phase:**
For a new data point:
1. Look up P(C) and all P(xᵢ|C) from the lookup table
2. Multiply them together for each class
3. Predict the class with the highest product

This makes Naive Bayes extremely fast — both training and prediction are just table lookups and multiplications.

---

### The Zero Probability Problem — Laplace Smoothing

**Problem:** If a feature value never appeared with a certain class in training data, its probability = 0. Multiplying by 0 makes the entire product 0 — wiping out all other evidence.

**Example:** "Overcast" never appeared with "No" in training data → P(Overcast|No) = 0 → P(No|Overcast, ...) = 0 regardless of other features.

**Fix — Laplace Smoothing (Add-1 smoothing):**
Add 1 to every count before computing probabilities:

> P(xᵢ = v | C) = (Count(xᵢ=v, C) + 1) / (Count(C) + K)

Where K = number of unique values for feature xᵢ.

```python
# Without smoothing: count/total
# With Laplace smoothing: (count + 1) / (total + num_unique_values)
```

This ensures no probability is ever exactly zero.

---

### Types of Naive Bayes

Different versions handle different types of features:

| Type | Feature Type | Distribution Assumed | Use Case |
|---|---|---|---|
| **Gaussian NB** | Continuous numerical | Gaussian (Normal) | Continuous features (age, salary) |
| **Multinomial NB** | Discrete counts | Multinomial | Text classification (word counts) |
| **Bernoulli NB** | Binary (0/1) | Bernoulli | Binary features (word present/absent) |
| **Categorical NB** | Categorical | Categorical | Categorical features like play tennis |

---

### Gaussian Naive Bayes — Handling Numerical Data

When features are continuous, we can't build a simple frequency table. Instead, we assume each feature follows a Gaussian distribution within each class.

For each class C and feature xᵢ, we estimate:
- μᵢ,c = mean of xᵢ for class C
- σᵢ,c = standard deviation of xᵢ for class C

Then use the Gaussian PDF to compute likelihood:

> P(xᵢ = v | C) = (1/√(2πσ²)) × exp(-(v - μ)² / 2σ²)

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Access learned parameters
print(model.theta_)   # means for each class-feature combination
print(model.var_)     # variances for each class-feature combination
```

---

### sklearn Implementation

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

# Gaussian — for continuous features
gnb = GaussianNB(var_smoothing=1e-9)  # var_smoothing prevents zero variance
gnb.fit(X_train, y_train)

# Multinomial — for word counts (text)
mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing parameter
mnb.fit(X_train_counts, y_train)

# Bernoulli — for binary features
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_binary, y_train)

# Categorical — for categorical features
cnb = CategoricalNB(alpha=1.0)
cnb.fit(X_train_categorical, y_train)

y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)

print(classification_report(y_test, y_pred))
```

---

### Advantages and Disadvantages

**Advantages:**
- **Extremely fast** — training is just counting, prediction is just lookup
- **Works well with small data** — doesn't need millions of samples
- **Handles high dimensions well** — text classification with thousands of features
- **No gradient descent** — no learning rate, no convergence issues
- **Probabilistic output** — naturally gives probabilities, not just labels
- **Robust to irrelevant features** — irrelevant features contribute nearly equally to all classes

**Disadvantages:**
- **Naive assumption rarely holds** — features are almost never truly independent
- **Zero probability problem** — fixed by Laplace smoothing but still a concern
- **Poor probability estimates** — actual probability values are often miscalibrated (but class ranking is usually correct)
- **Doesn't capture feature interactions** — misses important patterns when features are correlated

---

### When to Use Naive Bayes

| Situation | Suitable? |
|---|---|
| Text classification (spam, sentiment) | ✓ Excellent — MultinomialNB |
| Small dataset, quick baseline | ✓ Good |
| Real-time prediction needed | ✓ Excellent — very fast |
| Features are highly correlated | ✗ Poor — naive assumption violated |
| Need accurate probability estimates | ✗ Poor — probabilities often miscalibrated |
| Complex feature interactions matter | ✗ Poor — can't capture them |

---

## Complete Flow

```
Training:
Raw data
    ↓
Count occurrences of each feature value per class
    ↓
Compute P(C) for each class
    ↓
Compute P(xᵢ=v | C) for every feature value and class
    ↓
Store in lookup table (dictionary)

Testing:
New data point (x₁, x₂, ..., xₙ)
    ↓
For each class C:
    Score = P(C) × P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)
    ↓
Predict class with highest score (MAP rule)
```

---

## Interview One-Liners

**What is Naive Bayes?**
"A probabilistic classifier based on Bayes theorem that assumes conditional independence between features given the class. Despite this naive assumption being rarely true, it works surprisingly well in practice — especially for text classification."

**What is the naive assumption?**
"Conditional independence — given the class label, all features are assumed to be independent of each other. P(X|C) = Π P(xᵢ|C). This makes the computation tractable but is almost never true in reality."

**What is MAP rule?**
"Maximum A Posteriori — predict the class that maximises P(C) × Π P(xᵢ|C). We ignore P(X) because it's the same for all classes and doesn't affect which class has the highest posterior."

**What is Laplace smoothing?**
"Adding 1 to every count before computing probabilities. Prevents zero probabilities when a feature value never appeared with a certain class in training data. Without it, one zero probability wipes out all other evidence."

**Why is Naive Bayes fast?**
"Training is just counting — no gradient descent, no iteration. Testing is just multiplying probabilities looked up from a table. Both are O(n×k) where n is features and k is classes."

**Gaussian vs Multinomial vs Bernoulli NB?**
"Gaussian for continuous features — assumes normal distribution within each class. Multinomial for word counts — text classification. Bernoulli for binary features — word present or absent. Choice depends on the nature of your features."
