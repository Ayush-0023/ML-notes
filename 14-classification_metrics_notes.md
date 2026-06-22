# Classification Metrics — Complete Notes

---

## Why Not Just Use Accuracy?

Accuracy = (Correct Predictions) / (Total Predictions)

Sounds perfect. But it lies when classes are imbalanced.

**Example:** 95% of emails are not spam. A model that predicts "not spam" for everything gets 95% accuracy — without learning anything useful. It completely misses every spam email.

Accuracy is misleading when classes are unbalanced. You need metrics that actually measure performance on each class.

---

## The Confusion Matrix

The foundation of all classification metrics. A table showing what your model actually predicted vs what the truth was.

For binary classification:

```
                    Predicted
                  Positive  Negative
Actual  Positive |   TP   |   FN   |
        Negative |   FP   |   TN   |
```

| Term | Meaning | Example (disease detection) |
|---|---|---|
| **TP** (True Positive) | Predicted positive, actually positive | Sick person correctly identified as sick |
| **TN** (True Negative) | Predicted negative, actually negative | Healthy person correctly identified as healthy |
| **FP** (False Positive) | Predicted positive, actually negative | Healthy person wrongly flagged as sick |
| **FN** (False Negative) | Predicted negative, actually positive | Sick person wrongly cleared as healthy |

### Type 1 and Type 2 Errors

| Error Type | What it is | Also called |
|---|---|---|
| **Type 1 Error** | Predicted positive but actually negative | False Positive (FP) |
| **Type 2 Error** | Predicted negative but actually positive | False Negative (FN) |

**Which is worse depends on context:**
- Disease detection: FN is worse (missing a sick person is dangerous)
- Spam detection: FP is worse (blocking a real email is more annoying than missing spam)
- Fraud detection: FN is worse (missing fraud is costly)

---

## Accuracy

> **Accuracy = (TP + TN) / (TP + TN + FP + FN)**

**When to use:** Only when classes are balanced AND both types of errors are equally important.

**When NOT to use:** Imbalanced datasets — it gives a false sense of performance.

---

## Precision

> **Precision = TP / (TP + FP)**

"Of all the times I predicted positive, how often was I actually right?"

**Intuition:** Quality of positive predictions. High precision = when you say positive, you're usually correct.

**When it matters:** When False Positives are costly.
- Spam filter: if you mark something as spam (positive), you better be right — otherwise real emails get lost
- Cancer diagnosis for surgery: if you recommend surgery (positive), you better be sure

**Tradeoff:** To increase precision, be more conservative with positive predictions → miss more actual positives (lower recall)

---

## Recall (Sensitivity)

> **Recall = TP / (TP + FN)**

"Of all the actual positives, how many did I catch?"

**Intuition:** Coverage of actual positives. High recall = you found most of the real positives.

**When it matters:** When False Negatives are costly.
- Disease detection: you want to catch every sick person, even if some healthy people get flagged
- Fraud detection: you want to flag every fraud, even at the cost of some false alarms

**Tradeoff:** To increase recall, be more aggressive with positive predictions → more false alarms (lower precision)

---

## The Precision-Recall Tradeoff

Precision and Recall are inversely related — improving one usually hurts the other.

**Controlled by the classification threshold:**
- **Lower threshold** (e.g., 0.3) → predict positive more often → higher recall, lower precision
- **Higher threshold** (e.g., 0.7) → predict positive less often → higher precision, lower recall

```
Threshold = 0.3:  High recall, low precision (catches most positives, many false alarms)
Threshold = 0.5:  Balanced (default)
Threshold = 0.7:  High precision, low recall (confident predictions, misses some positives)
```

The right threshold is a **business decision**, not a mathematical one.

---

## F1 Score

> **F1 = 2 × (Precision × Recall) / (Precision + Recall)**

The harmonic mean of precision and recall. Balances both.

**Why harmonic mean instead of arithmetic mean?**
Arithmetic mean of Precision=1.0 and Recall=0.0 = 0.5 — looks decent but the model is useless.
Harmonic mean = 0 — correctly penalises extreme imbalance between precision and recall.

**When to use:** When both precision and recall matter equally — neither FP nor FN is clearly more costly.

**F1 range:** 0 to 1. Higher is better.

### Weighted F1 for Imbalanced Classes

```python
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')
# Weights each class by its support (number of actual instances)
```

---

## ROC Curve and AUC

### What is ROC?

ROC = Receiver Operating Characteristic curve.

Plots **True Positive Rate (Recall)** vs **False Positive Rate** across all possible thresholds.

> **TPR (True Positive Rate) = TP / (TP + FN)** — same as Recall
> **FPR (False Positive Rate) = FP / (FP + TN)** — rate of false alarms among actual negatives

Each threshold gives one (FPR, TPR) point. Connecting all points gives the ROC curve.

**Ideal model:** Curve hugs the top-left corner (high TPR, low FPR at every threshold).
**Random model:** Diagonal line from (0,0) to (1,1) — no better than chance.
**Terrible model:** Below the diagonal — worse than random.

### AUC — Area Under the Curve

> **AUC = Area under the ROC curve**

- AUC = 1.0 → perfect classifier
- AUC = 0.5 → random classifier (no skill)
- AUC < 0.5 → worse than random (predictions are inverted)

**Why AUC is powerful:**
- **Threshold-independent** — evaluates model performance across all thresholds, not just 0.5
- **Scale-independent** — doesn't depend on absolute probabilities, only rank order
- **Works well for imbalanced classes** — unlike accuracy

**Interpretation:** AUC = 0.85 means the model correctly ranks a random positive example above a random negative example 85% of the time.

---

## Complete Code

```python
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability of positive class

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.show()

# Precision, Recall, F1
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# Full Report
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Custom threshold
threshold = 0.3
y_pred_custom = (y_prob >= threshold).astype(int)
print(f"\nWith threshold={threshold}:")
print(f"Precision: {precision_score(y_test, y_pred_custom):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_custom):.4f}")
```

---

## Multiclass Classification Metrics

For more than 2 classes, metrics extend using averaging strategies:

| Strategy | What it does | Use when |
|---|---|---|
| **Macro** | Average across all classes equally | All classes equally important |
| **Weighted** | Average weighted by class frequency | Imbalanced classes, larger classes matter more |
| **Micro** | Pool all predictions then compute | Every prediction equally important |

```python
# Multiclass confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# Multiclass F1
f1_macro    = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_micro    = f1_score(y_test, y_pred, average='micro')
```

---

## Quick Reference — When to Use Which Metric

| Situation | Best Metric |
|---|---|
| Balanced classes, equal error costs | Accuracy or F1 |
| Imbalanced classes | F1, Precision, Recall, AUC |
| FP is very costly (spam, surgery) | Precision |
| FN is very costly (disease, fraud) | Recall |
| Want threshold-independent evaluation | AUC-ROC |
| Comparing models across datasets | AUC-ROC |
| Both FP and FN matter equally | F1 Score |
| Multiclass, all classes matter equally | Macro F1 |
| Multiclass, imbalanced | Weighted F1 |

---

## Interview One-Liners

**Why not just use accuracy?**
"Accuracy is misleading on imbalanced datasets. A model predicting the majority class always gets high accuracy without learning anything useful. Precision, recall, and F1 reveal actual per-class performance."

**What is precision?**
"Of all positive predictions, how many were actually positive. High precision = low false alarm rate. Use when false positives are costly."

**What is recall?**
"Of all actual positives, how many did the model catch. High recall = low miss rate. Use when false negatives are costly."

**What is F1?**
"Harmonic mean of precision and recall. Balances both. Better than arithmetic mean because it penalises extreme imbalance between the two."

**What is AUC-ROC?**
"AUC measures the probability that the model ranks a random positive example higher than a random negative one. It's threshold-independent and works well for imbalanced classes. AUC=0.5 is random, AUC=1.0 is perfect."

**Type 1 vs Type 2 error?**
"Type 1 = False Positive — predicted positive when actually negative. Type 2 = False Negative — predicted negative when actually positive. Which is worse depends on the application."
