# Ensemble Learning — Complete Notes

---

## What is Ensemble Learning?

Ensemble learning is a technique where multiple models (called "weak learners" or "base models") are combined to produce a stronger, more accurate predictive model. By aggregating multiple models, we reduce errors, improve generalisation, and enhance robustness.

**Core principle:** Individual models may be "weak" or inconsistent, but when combined correctly, their errors tend to cancel out while their correct predictions reinforce each other.

---

## Wisdom of the Crowd

The foundational idea behind ensemble learning: a large group's collective judgment is often more accurate than a single expert's.

**Four conditions required for it to work effectively:**

1. **Diversity of Opinion** — Individuals (models) should have unique knowledge/perspectives
2. **Independence** — Opinions shouldn't be influenced by each other
3. **Decentralisation** — Decisions come from individuals, not a central authority
4. **Aggregation** — A method exists to combine individual opinions (averaging, voting)

**In ML terms:** Base models must be diverse and largely independent. If all models make the same mistakes, combining them doesn't help at all.

---

## How Diversity is Created

For ensemble learning to work, base models must differ from each other. Two ways to achieve this:

**Method 1 — Different Algorithms**
m1 = Logistic Regression, m2 = SVM, m3 = Decision Tree, m4 = KNN...
*(Used in Voting and Stacking)*

**Method 2 — Same Algorithm, Different Data**
m1 trained on dataset D1, m2 on D2, m3 on D3...
*(Used in Bagging and Boosting)*

---

## How Predictions Are Combined

**Classification:**
Each base model predicts a class (0 or 1). Final prediction = majority vote.

**Regression:**
Each base model predicts a continuous value. Final prediction = mean (average) of all predictions.

---

## The Four Types of Ensemble Learning

| Type | Base Models | Training | Combination |
|---|---|---|---|
| **Voting** | Different algorithms | Same data, parallel | Majority vote / averaging |
| **Stacking** | Different algorithms | Same data, parallel | Meta-model learns optimal combination |
| **Bagging** | Same algorithm | Different bootstrap samples, parallel | Majority vote / averaging |
| **Boosting** | Same algorithm | Sequential, each fixes previous errors | Weighted combination |

### Voting (Democracy)
All models vote equally on the same input. No learning of how to combine — simple majority/average.

### Stacking (Partial Democracy)
A meta-model learns how much weight to give each base model's prediction — smarter combination than simple voting.

### Bagging (Bootstrap Aggregating)
Same algorithm trained on different random subsets (with replacement) of data. Reduces **variance**.

### Boosting
Models trained sequentially — each new model focuses on correcting the previous model's mistakes. Reduces **bias**.

---

## Why Ensemble Learning Works

**1. Reduces Variance**
Multiple models trained on different data/features average out overfitting from any single model. (Bagging's mechanism)

**2. Reduces Bias**
Sequential methods combine weak learners into a strong learner capturing more complex patterns. (Boosting's mechanism)

**3. Reduces Noise Sensitivity**
Random fluctuations in data are less likely to dominate when many models are combined — errors average out.

**4. Improves Generalisation**
Diversity among models ensures better performance on unseen data — no single model's blind spot dominates.

---

## Advantages and Disadvantages

**Advantages:**
- Improved performance — higher accuracy than individual models
- Addresses bias-variance tradeoff directly (Bagging↓variance, Boosting↓bias)
- Robustness — less sensitive to noise and anomalies

**Disadvantages:**
- Computational complexity — training/storing multiple models needs more resources
- Longer training time — especially with large datasets or complex base models
- Reduced interpretability — harder to explain than a single decision tree or linear model

---

## Quick Decision Guide

```
Need a quick performance boost with existing diverse models?
    → Voting

Want models to learn how to weight each other optimally?
    → Stacking

Base model has high variance (e.g., deep decision tree)?
    → Bagging (Random Forest)

Base model has high bias (e.g., shallow decision tree)?
    → Boosting (AdaBoost, Gradient Boosting, XGBoost)
```

---

## Interview One-Liners

**What is ensemble learning?**
"Combining multiple models to produce a stronger model than any individual one. Works because diverse models make different errors — when aggregated, errors cancel out while correct predictions reinforce each other."

**Why does diversity matter?**
"If all models make the same mistakes, combining them doesn't help — you just get the same wrong answer multiple times. Diversity ensures different models' errors are uncorrelated, so averaging genuinely reduces error."

**Bagging vs Boosting?**
"Bagging trains models in parallel on different data samples to reduce variance — good for high-variance base models like deep trees. Boosting trains models sequentially, each correcting the previous one's errors, to reduce bias — good for high-bias base models like shallow trees."
