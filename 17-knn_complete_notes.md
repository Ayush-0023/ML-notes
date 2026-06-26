# K-Nearest Neighbours (KNN) — Complete Notes

---

## What is KNN?

KNN is a simple, non-parametric, instance-based learning algorithm used for both classification and regression. It makes predictions based on the K most similar training examples to the query point.

**Non-parametric:** Makes no assumptions about the underlying data distribution.
**Instance-based (Lazy learning):** Stores all training data and does computation only at prediction time. No explicit training phase.

---

## Intuition

**Example:** Dataset with CGPA, IQ, and Placement (0 or 1).

**During prediction for a new query point `a`:**
1. Calculate distance between `a` and ALL training points
2. Sort distances in ascending order
3. Take the K nearest neighbours
4. For classification → majority vote → predict that class
5. For regression → average of K neighbours' values

**Example with K=3:**
- 3 nearest neighbours: [1, 1, 0]
- Majority = 1 → predict 1 (Placed)

**In higher dimensions:**
Every training point is a vector. The query point is also a vector. Distance is computed between vectors — same principle, just more dimensions.

---

## Distance Metrics

KNN is entirely dependent on distance. Different metrics measure "closeness" differently.

### Euclidean Distance (default, most common)
> d = √(Σ(xᵢ - yᵢ)²)

Straight-line distance. Works well for continuous features with similar scales.

### Manhattan Distance (L1)
> d = Σ|xᵢ - yᵢ|

Sum of absolute differences. More robust to outliers than Euclidean. Good for high-dimensional data.

### Minkowski Distance (generalisation)
> d = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)

- p=1 → Manhattan
- p=2 → Euclidean
- p→∞ → Chebyshev (max difference across dimensions)

### Cosine Similarity
> similarity = (x·y) / (||x|| × ||y||)

Measures angle between vectors, not magnitude. Good for text data where direction matters more than length.

```python
from sklearn.neighbors import KNeighborsClassifier

# Different distance metrics
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_minkowski = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)
```

---

## Why Feature Scaling is Mandatory

KNN uses distances. Without scaling, features with larger ranges dominate the distance calculation.

**Example:** CGPA (6-10) and Salary (30,000-100,000). The salary difference between two points is thousands of times larger than the CGPA difference — salary completely dominates the distance. CGPA becomes irrelevant.

**Rule:** Always StandardScaler (or MinMaxScaler) before KNN.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('data.csv')
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:], df.iloc[:, 0],
    test_size=0.2, random_state=2
)

# Always scale before KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only, don't fit

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## Choosing K — The Critical Hyperparameter

K controls the bias-variance tradeoff:

| K value | Effect | Result |
|---|---|---|
| K = 1 | Only 1 neighbour | Overfitting — decision surface has many tiny regions, very sensitive to noise |
| K = n | All points are neighbours | Underfitting — always predicts the majority class of entire training set |
| K = optimal | Balanced | Good generalisation |

### Method 1 — √n Rule (not recommended)
> K ≈ √n (n = number of training samples)

Quick estimate but not reliable. Ignores the actual data structure.

### Method 2 — Cross-Validation (recommended)

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

# Method A — Simple train/test comparison
scores = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    scores.append(accuracy_score(y_test, y_pred))

plt.plot(range(1, 31), scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('KNN — Choosing Optimal K')
plt.show()
# Select K where accuracy is highest

# Method B — Proper cross-validation (more reliable)
cv_scores = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = np.argmax(cv_scores) + 1
print(f"Optimal K: {optimal_k}")
```

**General rules of thumb:**
- Choose odd K for binary classification to avoid ties
- K between 3 and 20 is typical for most problems
- Larger K → smoother decision boundary → more bias, less variance

---

## KNN for Regression

KNN works for regression too — instead of majority vote, take the average of K neighbours:

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance'  # closer neighbours weighted more
)
knn_reg.fit(X_train_scaled, y_train)
y_pred = knn_reg.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

---

## Weighted KNN

Standard KNN treats all K neighbours equally. Weighted KNN gives more influence to closer neighbours.

> Prediction = Σ (wᵢ × yᵢ) / Σ wᵢ

Where wᵢ = 1/distance (closer = higher weight)

```python
knn_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'  # 'uniform' (default) or 'distance'
)
```

**When to use:** When you believe closer neighbours should have stronger influence — generally improves performance on noisy data.

---

## Decision Surface

A visual tool to understand how the model divides the feature space into class regions.

**How it works:**
1. Plot training data
2. Find the range of X and Y axes
3. Generate a dense numpy meshgrid covering this range
4. Send all meshgrid points through the trained model
5. Colour each point by its predicted class (e.g., 0=blue, 1=orange)
6. Display as a background image

**What it reveals:**
- K=1: Many small irregular regions → overfitting clearly visible
- K=large: Smooth, simple regions → underfitting visible
- K=optimal: Clean, sensible boundaries

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple decision surface implementation
def plot_decision_surface(model, X, y, title='Decision Surface'):
    h = 0.02  # step size in mesh

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.show()

# Using mlxtend (easier)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train_scaled, y_train.values, clf=knn)
plt.title('KNN Decision Surface')
plt.show()
```

---

## Overfitting and Underfitting in KNN

**K=1 (Overfitting):**
- Each training point becomes its own region
- Decision surface is extremely jagged and irregular
- Perfect training accuracy, poor test accuracy
- High variance — very sensitive to noise

**K=n (Underfitting):**
- The entire dataset is every point's neighbourhood
- Always predicts the majority class
- Low training AND test accuracy (unless one class dominates)
- High bias — completely ignores local structure

**Optimal K:**
- Smooth, generalised decision boundary
- Good balance of training and test accuracy
- Found through cross-validation

---

## KNN vs Other Algorithms — Key Differences

| | KNN | Logistic Regression | Naive Bayes |
|---|---|---|---|
| **Training** | None (lazy) | Gradient descent | Counting |
| **Prediction** | Distance computation | Matrix multiply + sigmoid | Lookup table |
| **Assumptions** | None | Linear boundary | Feature independence |
| **Interpretability** | Low | High (coefficients) | Medium |
| **Speed (train)** | Fast | Slow | Very fast |
| **Speed (predict)** | Slow (large data) | Fast | Very fast |
| **Works on non-linear** | Yes | No (without poly) | No |

---

## Limitations of KNN

**1. Slow Prediction on Large Datasets (Lazy Learning)**
All computation happens during prediction. For each query point, distances to ALL n training points must be computed. O(n×d) per prediction where d is dimensions.

No work is done during training — hence "lazy learner." But at prediction time, the laziness catches up.

**Fix:** Approximate nearest neighbour algorithms (KD-Tree, Ball-Tree, LSH) for large datasets.

```python
knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
# algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'
```

**2. Curse of Dimensionality**
In high dimensions, all points become roughly equidistant from each other. "Nearest neighbour" loses its meaning — there's no meaningful notion of closeness anymore.

Rule of thumb: KNN struggles beyond ~20 features without dimensionality reduction first.

**3. Sensitive to Outliers**
Outliers can create isolated regions in the decision surface, pulling nearby query points to wrong predictions. Use K>1 and weighted KNN to mitigate.

**4. Non-Homogeneous Feature Scales**
Features at different scales distort distances. Mandatory: always scale before KNN.

**5. Imbalanced Datasets**
With majority class heavily dominating, K nearest neighbours are likely all from the majority class → biased predictions.

**Fix:** Use weighted KNN, oversample minority class (SMOTE), or adjust class weights.

**6. Good for Prediction, Not Inference**
KNN gives you a prediction but can't tell you WHY — no coefficients, no feature importance, no interpretation. It's a black box.

If you need to explain which features drive predictions → use logistic regression or decision trees instead.

---

## Complete sklearn Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale — mandatory
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Find optimal K using GridSearchCV
param_grid = {
    'n_neighbors': range(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")

# 4. Train with best parameters
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test_scaled)

# 5. Evaluate
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

---

## Interview One-Liners

**What is KNN?**
"A lazy, non-parametric algorithm that classifies a query point by majority vote (or average for regression) of its K nearest training neighbours based on distance. No model is built during training — all computation happens at prediction time."

**Why is scaling mandatory for KNN?**
"KNN uses distances. Without scaling, features with larger ranges dominate the distance metric — other features become irrelevant regardless of their actual importance."

**How do you choose K?**
"Cross-validation — plot accuracy vs K and pick the K that maximises validation performance. K=1 overfits, K=n underfits. Generally prefer odd K for binary classification to avoid ties."

**What is the curse of dimensionality in KNN?**
"In high dimensions, all points become roughly equidistant. The concept of nearest neighbour breaks down — there's no meaningful local neighbourhood. KNN typically fails beyond ~20 features without prior dimensionality reduction."

**Why is KNN called lazy learning?**
"It does no work during training — just stores the data. All computation (distance calculations) happens at prediction time. Fast training, slow prediction on large datasets."

**KNN for regression?**
"Instead of majority vote, predict the average (or weighted average) of the K nearest neighbours' target values."
