# Voting Ensemble — Complete Notes

---

## What is a Voting Ensemble?

A Voting Ensemble combines predictions from multiple models (typically using different algorithms) to make a final prediction. All models are trained independently on the same data, then their predictions are combined.

---

## Voting Classifier

### Hard Voting

Each model votes for a class. The class with the majority of votes wins.

**Example:** 3 models predicting spam/not spam:
- Model 1 → Spam
- Model 2 → Not Spam
- Model 3 → Spam

**Result:** Spam (2 out of 3 votes) — majority rules.

### Soft Voting

Each model outputs class **probabilities**. Probabilities are averaged, and the class with the highest average probability wins.

**Example:** Same 3 models, now outputting P(spam):
- Model 1 → 0.9
- Model 2 → 0.4
- Model 3 → 0.8

Average = (0.9 + 0.4 + 0.8) / 3 = **0.7** → Predict Spam (0.7 > 0.5)

### Hard vs Soft Voting — Which is Better?

Soft voting is generally preferred because:
- It uses more information (probabilities, not just final labels)
- It can correctly handle close calls — e.g., if 2 models are barely confident in one class and 1 model is very confident in the other, soft voting can flip the result, while hard voting cannot
- Requires that all models support `predict_proba()`

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

model1 = LogisticRegression()
model2 = SVC(probability=True)  # probability=True needed for soft voting
model3 = DecisionTreeClassifier()

# Hard voting
voting_hard = VotingClassifier(
    estimators=[('lr', model1), ('svc', model2), ('dt', model3)],
    voting='hard'
)

# Soft voting (generally better)
voting_soft = VotingClassifier(
    estimators=[('lr', model1), ('svc', model2), ('dt', model3)],
    voting='soft',
    weights=[2, 1, 1]  # optional — give more weight to better models
)

voting_soft.fit(X_train, y_train)
y_pred = voting_soft.predict(X_test)
y_prob = voting_soft.predict_proba(X_test)
```

---

## Voting Regressor

For regression, there's no "hard vs soft" distinction — the final prediction is always the **average** of all base models' outputs.

**Example:** 3 models predicting house price:
- Model 1 → $300,000
- Model 2 → $320,000
- Model 3 → $310,000

**Final prediction:** (300,000 + 320,000 + 310,000) / 3 = **$310,000**

```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

model1 = LinearRegression()
model2 = SVR()
model3 = DecisionTreeRegressor()

voting_reg = VotingRegressor(
    estimators=[('lr', model1), ('svr', model2), ('dt', model3)],
    weights=[2, 1, 1]  # optional weighting
)

voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)
```

---

## Why Voting Ensemble Works

### Two Critical Assumptions

**Assumption 1 — Base models should be independent**
Models should make different kinds of errors. If all models are highly correlated (make the same mistakes), voting provides no benefit — you're just averaging the same wrong answer multiple times.

**Assumption 2 — All base models should have accuracy > 50%**
If even one model performs worse than random guessing, it can drag the ensemble's performance below that of the best individual model. Voting amplifies good models when most models are decent — but it amplifies bad decisions when models are mostly bad.

### The Mathematical Intuition (Condorcet's Jury Theorem)

This is the formal mathematical basis for why voting works — based on a result from 18th century mathematics.

**Setup:** Suppose you have N independent classifiers, each with accuracy p > 0.5. As N increases, the probability that the **majority vote** is correct approaches 1.

**Simple example:** 3 independent models, each with 70% accuracy.

Probability majority vote is correct:
> P(majority correct) = P(all 3 correct) + P(exactly 2 correct)

> P(all 3 correct) = 0.7³ = 0.343
> P(exactly 2 correct) = C(3,2) × 0.7² × 0.3 = 3 × 0.49 × 0.3 = 0.441

> P(majority correct) = 0.343 + 0.441 = **0.784**

The ensemble (78.4%) outperforms any individual model (70%)!

**As N increases further** (say 100 independent 70%-accurate models), the majority vote accuracy approaches 100% — this is the formal proof behind "wisdom of the crowd."

**Critical caveat:** This only holds if models are truly **independent**. In practice, models trained on the same data are never perfectly independent — which is why ensemble gains are real but smaller than the theoretical maximum.

### What Happens if Models Have < 50% Accuracy

If individual model accuracy p < 0.5, the same math works in reverse — the ensemble's accuracy gets **worse** than individual models as N increases. This is why Assumption 2 is critical — bad models actively hurt the ensemble, they don't just fail to help.

---

## Complete Implementation Example

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define diverse base models
lr = LogisticRegression()
svc = SVC(probability=True)
knn = KNeighborsClassifier()

# Check individual performance first
for name, model in [('LR', lr), ('SVC', svc), ('KNN', knn)]:
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"{name}: {acc:.4f}")

# Voting ensemble
voting = VotingClassifier(
    estimators=[('lr', lr), ('svc', svc), ('knn', knn)],
    voting='soft'
)
voting.fit(X_train_scaled, y_train)
print(f"Voting Ensemble: {accuracy_score(y_test, voting.predict(X_test_scaled)):.4f}")

# Cross-validation to confirm improvement is consistent
scores = cross_val_score(voting, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## When to Use Voting Ensemble

| Situation | Suitable? |
|---|---|
| You have several decent (>50% accuracy) diverse models | ✓ Yes |
| Models are highly correlated (similar errors) | ✗ Limited benefit |
| One model is much worse than others | ✗ Risk of dragging down performance |
| Need interpretability | ✗ Harder to explain than single model |
| Want a quick performance boost with minimal engineering | ✓ Yes — easiest ensemble method to implement |

---

## Interview One-Liners

**What is a voting ensemble?**
"Combines predictions from multiple different algorithms using majority vote (hard voting) or averaged probabilities (soft voting) for classification, and simple averaging for regression."

**Hard vs soft voting?**
"Hard voting counts final class predictions and takes the majority. Soft voting averages predicted probabilities before deciding — generally more accurate since it uses confidence information, not just the final label."

**Why does voting need independent, decent models?**
"Condorcet's Jury Theorem proves that majority vote accuracy approaches 100% as the number of independent classifiers increases — but only if each classifier has >50% accuracy. If models are correlated or individually worse than random, the ensemble can perform worse than the best single model."
