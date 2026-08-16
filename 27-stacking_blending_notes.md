# Stacking and Blending Ensembles — Complete Notes

---

## What is Stacking?

Stacking (Stacked Generalisation) is an ensemble technique where multiple different base models' predictions are used as **input features** to train a final "meta-model" that learns how to optimally combine them.

**Key difference from Voting:** Voting combines predictions using a fixed rule (majority vote or simple average). Stacking instead **learns** the best way to combine predictions — giving more weight to models that tend to be more reliable, potentially even learning non-linear combination rules.

---

## How Stacking Works — Step by Step

### Step 1 — Train Base Models (Level 0)

Train several different algorithms (e.g., Logistic Regression, SVM, Decision Tree, KNN) on the training data — same as Voting Ensemble's setup.

### Step 2 — Generate Predictions for Meta-Model Training

Here's the critical part that makes stacking different from naive combination: **the base models' predictions used to train the meta-model must come from data the base models did NOT see during their own training** — otherwise we get severe data leakage and overfitting.

**The proper approach — K-Fold Predictions:**

1. Split training data into K folds (e.g., K=5)
2. For each fold k:
   - Train each base model on the other K-1 folds
   - Predict on fold k (which that model hasn't seen)
3. After all folds are processed, every training row has an "out-of-fold" prediction from each base model
4. These out-of-fold predictions become the new feature set for the meta-model

```
Original Training Data (5 folds)
        ↓
For each base model (LR, SVM, DT):
    Train on folds [2,3,4,5] → predict fold 1 (out-of-fold)
    Train on folds [1,3,4,5] → predict fold 2 (out-of-fold)
    Train on folds [1,2,4,5] → predict fold 3 (out-of-fold)
    ... and so on
        ↓
Combine all out-of-fold predictions → new feature matrix
[LR_pred, SVM_pred, DT_pred] for every training row
        ↓
Train meta-model on this new feature matrix → original y
```

### Step 3 — Train the Meta-Model (Level 1)

The meta-model (often a simple model like Logistic Regression or Linear Regression) is trained using the base models' out-of-fold predictions as input features, and the original target as output.

### Step 4 — Final Prediction Pipeline

For new/test data:
1. Each base model (now trained on the FULL training data) makes a prediction
2. These predictions become input features
3. The meta-model combines them into the final prediction

---

## Why K-Fold Out-of-Fold Predictions Matter

**Without this step (naive approach):** If you train base models on the full training data and then use their predictions on that SAME training data to train the meta-model, the meta-model would essentially be learning from predictions that are "too good" (since base models have already seen and memorised parts of that data). This causes severe overfitting — performance looks great on training data, collapses on test data.

**With out-of-fold predictions:** The meta-model learns from realistic, unseen-data-quality predictions — exactly what it will face at test time.

---

## Implementation

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_models = [
    ('lr', LogisticRegression()),
    ('svc', SVC(probability=True)),
    ('dt', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier())
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,              # K-fold for generating out-of-fold predictions (handled automatically!)
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)
print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Compare to individual models
for name, model in base_models:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc:.4f}")
```

**sklearn's `StackingClassifier`/`StackingRegressor` handles the K-fold out-of-fold prediction generation automatically** via the `cv` parameter — you don't need to implement it manually.

---

## What is Blending?

Blending is a simplified variant of stacking. Instead of using K-fold cross-validation to generate out-of-fold predictions, blending uses a **single holdout validation set**.

### How Blending Works

1. Split training data into Train and Validation (Holdout) sets — e.g., 80/20
2. Train base models on the Train set only
3. Generate predictions from base models on the Validation set
4. Train the meta-model using these validation predictions as features, validation labels as target
5. For test time: base models (trained on Train set) predict on test data → meta-model combines these into final prediction

```
Training Data
    ↓
Split into Train (80%) + Validation (20%)
    ↓
Train base models on Train (80%) only
    ↓
Base models predict on Validation (20%) — these are "clean" unseen predictions
    ↓
Train meta-model on [base model predictions on Validation] → [Validation labels]
    ↓
At test time: base models (trained on Train 80%) predict on test → meta-model combines
```

### Stacking vs Blending

| | Stacking | Blending |
|---|---|---|
| **Validation method** | K-fold cross-validation | Single holdout set |
| **Data usage** | More efficient — every row gets an out-of-fold prediction | Less efficient — holdout portion isn't used for base model training |
| **Computational cost** | Higher (K times more base model training) | Lower (base models trained once) |
| **Robustness** | More robust (averages over K folds) | Less robust (depends on the single split) |
| **Used by** | Academic/research settings, when compute allows | Kaggle competitions (faster iteration) |

**Practical takeaway:** Blending is essentially a computationally cheaper, less rigorous approximation of stacking. Stacking is more robust but requires K times more base-model training.

---

## Why Stacking/Blending Works

The meta-model can learn patterns like:
- "Model A is more reliable when the input has characteristic X"
- "Model B and Model C tend to agree, but when they disagree, trust Model C more"
- Non-linear combinations that simple voting/averaging could never capture

This is strictly more expressive than Voting — Voting is actually a special case of Stacking where the meta-model is forced to be "simple majority/average" rather than learned.

---

## Multi-Level Stacking

Stacking can have multiple levels — Level 0 (base models) → Level 1 (meta-model on base predictions) → Level 2 (another meta-model on Level 1 outputs) → and so on. In practice, 2 levels are most common; going deeper rarely helps and risks overfitting further.

---

## Interview One-Liners

**What is stacking?**
"An ensemble method where predictions from multiple base models become input features for a meta-model, which learns the optimal way to combine them — rather than using a fixed rule like majority voting."

**Why use K-fold out-of-fold predictions in stacking?**
"To avoid data leakage. If the meta-model trains on predictions made by base models on data those models already saw, the meta-model learns from artificially good predictions — leading to severe overfitting. Out-of-fold predictions simulate genuine unseen-data performance."

**Stacking vs Blending?**
"Stacking uses K-fold cross-validation to generate out-of-fold predictions for the meta-model — more robust but more computationally expensive. Blending uses a single holdout validation set instead — faster but less robust, commonly used in Kaggle for quick iteration."

**Stacking vs Voting?**
"Voting combines predictions with a fixed rule (majority vote/average). Stacking learns the combination rule via a meta-model — strictly more expressive, since voting is essentially a stacking special case with a hardcoded, unlearned combiner."
