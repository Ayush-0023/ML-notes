# Decision Trees — Complete Notes

---

## What is a Decision Tree?

A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It splits data into smaller and smaller subsets based on feature values, forming a tree-like structure of decisions.

**Programmatic intuition:** Decision trees are nothing but a giant structure of nested if-else conditions.

**Geometric intuition:** Decision trees use hyperplanes that run **parallel to the axes** to cut your coordinate system into hyper-cuboids (rectangles in 2D, boxes in 3D). This is fundamentally different from SVM or logistic regression which can draw diagonal boundaries.

```
Feature 2
    |          |
    |  Class A | Class B
    |          |
    |——————————|
    |          |
    |  Class C | Class D
    |          |
    +——————————————→ Feature 1
```

Each split creates axis-aligned boundaries — hence "parallel to axes."

**Famous use case:** The game Akinator uses decision trees to identify who you're thinking of by asking binary yes/no questions.

---

## Terminology

| Term | Definition |
|---|---|
| **Root Node** | The topmost node — the first decision split, applied to the entire dataset |
| **Decision Node** | An internal node that is neither the root nor a leaf — splits data further |
| **Splitting Point** | The condition at a node where the tree branches (e.g., "Age > 30?") |
| **Branch / Edge** | The connection between nodes — represents an outcome of a decision |
| **Leaf Node** | The terminal node — makes the final prediction (class label for classification, value for regression) |
| **Depth** | The length of the longest path from root to leaf |
| **Pruning** | Removing branches that provide little predictive power to reduce overfitting |

---

## Advantages and Disadvantages

**Advantages:**
- Easy to understand and visualise — mirrors human decision-making
- No feature scaling required (splits are based on rank, not distance)
- Handles both numerical and categorical data
- Captures non-linear relationships naturally
- Automatically handles feature interactions
- Fast prediction — O(log n) once trained
- No assumptions about data distribution (non-parametric)

**Disadvantages:**
- Prone to overfitting — a fully grown tree memorises training data
- Unstable — small changes in data can produce very different trees
- Biased toward features with more unique values (information gain is affected by cardinality)
- Poor extrapolation — can't predict beyond the range seen in training data
- More suitable for categorical than numerical features (explained below)

---

## How Decision Trees Work — The Core Algorithm

At each node, the algorithm asks: **"Which feature and which split value creates the purest possible child nodes?"**

**The process (ID3/CART algorithm):**
1. Start with all training data at the root node
2. For each feature, calculate the impurity of all possible splits
3. Choose the feature and split that maximises Information Gain (or minimises Gini impurity)
4. Split the data into child nodes
5. Repeat recursively for each child node
6. Stop when a stopping criterion is met (all leaf nodes are pure, max depth reached, etc.)

---

## Entropy — Measuring Disorder

### Intuition

Entropy is the measure of disorder/uncertainty in a dataset.

**Physical analogy:** Ice < Water < Vapour in terms of disorder — vapour molecules have maximum freedom to move randomly, hence maximum entropy.

**ML analogy:** A dataset with all "Yes" labels has zero uncertainty — you know exactly what to predict. A 50/50 split has maximum uncertainty. More knowledge = less entropy.

### Mathematical Formula

For a dataset with C classes:

> **H = -Σ pᵢ × log₂(pᵢ)**

Where pᵢ = proportion of class i in the dataset.

**Example — Binary classification:**
- All Yes (100%, 0%): H = -(1×log₂1 + 0×log₂0) = 0 → **perfectly pure**
- 50/50 split: H = -(0.5×log₂0.5 + 0.5×log₂0.5) = 1 → **maximum impurity**
- 70/30 split: H = -(0.7×log₂0.7 + 0.3×log₂0.3) ≈ 0.881

### Key Observations

- Minimum entropy = 0 (all samples belong to one class → perfectly pure)
- Maximum entropy = 1 for binary, can be > 1 for more classes
- Both log₂ and logₑ can be used (log₂ gives entropy in "bits")
- More uncertainty → more entropy; more knowledge → less entropy

### Entropy for Continuous Variables

For continuous target variables, we plot the KDE (distribution). A narrower distribution (e.g., constrained between -1 and 1) has less uncertainty and therefore less entropy. A wider distribution has more entropy.

---

## Information Gain

### Definition

Information Gain (IG) is the reduction in entropy after splitting on a feature. It tells us how much "information" a feature gives us about the class label.

> **IG(parent, feature) = H(parent) - Σ [wᵢ × H(childᵢ)]**

Where wᵢ = proportion of samples going to child node i (weighted average).

### Example

**Parent node:** 10 samples (5 Yes, 5 No) → H(parent) = 1.0

**Split on Outlook:**
- Sunny: 3 samples (2 Yes, 1 No) → H = 0.918
- Overcast: 4 samples (4 Yes, 0 No) → H = 0.0 (pure!)
- Rainy: 3 samples (3 No, 0 Yes) → H = 0.0 (pure!)

Weighted entropy after split:
> H(after) = (3/10)×0.918 + (4/10)×0 + (3/10)×0 = 0.275

Information Gain:
> IG = 1.0 - 0.275 = **0.725**

**The algorithm chooses the feature with the highest IG at each node.**

### Why Maximise Information Gain?

High IG means the split creates child nodes that are much purer than the parent. The more impurity we remove, the better the split — the tree needs fewer questions to reach confident predictions.

---

## Gini Impurity

### Definition

Gini Impurity measures the probability of misclassifying a randomly chosen sample from a node, if labelled according to the class distribution in that node.

> **Gini = 1 - Σ pᵢ²**

**Key values:**
- Gini = 0 → all samples belong to one class → perfectly pure
- Gini = 0.5 → equal split between 2 classes → maximum impurity (for binary)
- Range: [0, 0.5] for binary, [0, 1-1/K] for K classes

### Example

- 100% one class: Gini = 1 - (1²) = 0
- 50/50 split: Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
- 70/30 split: Gini = 1 - (0.7² + 0.3²) = 1 - 0.58 = 0.42

### Entropy vs Gini Impurity

| | Entropy | Gini Impurity |
|---|---|---|
| **Formula** | -Σ p×log₂(p) | 1 - Σ p² |
| **Computation** | Uses logarithm — slower | Uses squares — faster |
| **Range (binary)** | [0, 1] | [0, 0.5] |
| **Split tendency** | More balanced splits | Tends to isolate the largest class |
| **sklearn default** | Not default | Default (criterion='gini') |
| **Use when** | Want balanced, interpretable splits | Large datasets, speed matters |

**In sklearn:**
```python
# Gini (default)
DecisionTreeClassifier(criterion='gini')
# IG and entropy not calculated at all

# Entropy
DecisionTreeClassifier(criterion='entropy')
# Gini not calculated at all
```

Both give similar results in practice. Gini is computationally slightly faster.

---

## Handling Numerical Features

### The Problem

For categorical features (e.g., Outlook: Sunny/Cloudy/Rainy), finding splits is straightforward — try each category.

For numerical features (e.g., Age: 23, 45, 67, ...), there are infinitely many possible split points. If we use all unique values, we get too many potential splits, making the tree impractically deep.

### The Solution — Threshold-Based Splitting

For each numerical feature with n unique values:
1. Sort the unique values
2. Consider n-1 midpoints as candidate split thresholds
3. For each threshold t, create: "feature > t" (right) vs "feature ≤ t" (left)
4. Calculate Information Gain for each threshold
5. Select the threshold that gives maximum IG

**Example:** Feature values [10, 20, 30, 40]
- Midpoint 1: split at 15 → {10} vs {20, 30, 40}
- Midpoint 2: split at 25 → {10, 20} vs {30, 40}
- Midpoint 3: split at 35 → {10, 20, 30} vs {40}
- Pick the midpoint with highest IG

### Why This is Computationally Expensive

This entire process is done for **every numerical feature at every node** during training. For a dataset with many numerical columns and many rows, training becomes slow.

**BUT:** This computation happens **only once during training**. Once the tree is built, prediction for any new query point is just O(log n) — traverse the tree from root to leaf.

**This is why decision trees are better suited for categorical than numerical data** — fewer unique values = fewer candidate splits = faster training.

---

## Overfitting and Underfitting in Decision Trees

A fully grown decision tree (no stopping criteria) will perfectly memorise training data — every leaf has pure single samples. This is severe overfitting.

**Controlling tree complexity — Hyperparameters:**

### max_depth

Maximum depth of the tree. Shallow tree = underfitting. Deep tree = overfitting.

```python
DecisionTreeClassifier(max_depth=5)
# Deeper → more complex → more overfitting risk
# Shallower → simpler → more underfitting risk
```

### min_samples_split

Minimum number of samples required to split a node. Higher = less splitting = simpler tree.

```python
DecisionTreeClassifier(min_samples_split=10)
# Node with < 10 samples won't be split further
```

### min_samples_leaf

Minimum number of samples required at a leaf node. Prevents creating very small, specific leaf nodes.

```python
DecisionTreeClassifier(min_samples_leaf=5)
# Each leaf must have at least 5 samples
```

### max_features

Number of features to consider when looking for the best split. Reduces overfitting and training time.

```python
DecisionTreeClassifier(max_features='sqrt')
# Consider sqrt(n_features) at each split
# Same idea used in Random Forests
```

### max_leaf_nodes

Limits total number of leaf nodes. Directly controls model complexity.

```python
DecisionTreeClassifier(max_leaf_nodes=20)
```

### Finding Optimal Hyperparameters

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

### Pre-Pruning vs Post-Pruning

**Pre-pruning (early stopping):** Stop growing the tree early using hyperparameters above. Simple and fast.

**Post-pruning (Cost Complexity Pruning):** Grow the full tree first, then remove branches that add little predictive value. Controlled by `ccp_alpha` in sklearn.

```python
# Cost complexity pruning
clf = DecisionTreeClassifier(ccp_alpha=0.01)
# Higher alpha → more pruning → simpler tree
```

---

## Regression Trees

Decision trees can also predict continuous values instead of class labels.

**Key differences from classification trees:**

| | Classification Tree | Regression Tree |
|---|---|---|
| **Leaf prediction** | Majority class | Mean of samples in leaf |
| **Split criterion** | Gini / Information Gain | MSE / MAE reduction |
| **Impurity measure** | Entropy or Gini | Variance of y values |

**How splits work in regression:**
At each node, find the split that minimises the weighted sum of variances in the child nodes (equivalent to maximising variance reduction):

> Criterion = Var(parent) - Σ[wᵢ × Var(childᵢ)]

```python
from sklearn.tree import DecisionTreeRegressor

reg_tree = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=10,
    criterion='squared_error'  # 'squared_error' (MSE), 'friedman_mse', 'absolute_error', 'poisson'
)
reg_tree.fit(X_train, y_train)
y_pred = reg_tree.predict(X_test)
```

---

## Complete Implementation

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(
    criterion='gini',       # 'gini' or 'entropy'
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Visualise tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True)
plt.show()

# Text representation
print(export_text(clf, feature_names=feature_names))

# Feature importance
importances = pd.Series(clf.feature_importances_, index=feature_names)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()
```

### Using dtreeviz for Beautiful Visualisation

```python
# pip install dtreeviz
from dtreeviz.trees import dtreeviz

viz = dtreeviz(
    clf,
    X_train,
    y_train,
    target_name='target',
    feature_names=feature_names,
    class_names=class_names
)
viz.view()
```

---

## Feature Importance in Decision Trees

Decision trees naturally provide feature importance scores — how much each feature contributed to reducing impurity across all splits.

> Feature Importance = Σ [weighted impurity reduction at each split using this feature]

Normalised to sum to 1.

```python
# Get feature importances
importances = clf.feature_importances_
# Higher value = feature contributed more to splits = more important
```

**Important caveat:** Decision tree feature importance is biased toward high-cardinality features (features with many unique values) — they have more split points to try and thus more chances to look important.

---

## Decision Tree vs Other Algorithms

| | Decision Tree | Logistic Regression | KNN | SVM |
|---|---|---|---|---|
| **Feature scaling needed** | No | Yes | Yes | Yes |
| **Non-linear boundaries** | Yes (axis-aligned) | No (without poly) | Yes | Yes (kernel) |
| **Interpretability** | Very high | High | Low | Low |
| **Training speed** | Fast | Fast | Instant | Slow |
| **Prediction speed** | O(log n) | O(1) | O(n) | Fast |
| **Handles missing values** | Sometimes | No | No | No |
| **Overfitting risk** | High | Low | Depends on K | Depends on C |

---

## Interview One-Liners

**What is a decision tree?**
"A supervised algorithm that recursively splits data on features that maximise information gain (or minimise Gini impurity), creating a tree of if-else rules. Prediction traverses from root to leaf in O(log n)."

**What is entropy and information gain?**
"Entropy measures the disorder/uncertainty in a dataset. Information gain is the reduction in entropy after splitting on a feature — how much that split clarifies the class distribution. At each node we pick the feature that maximises information gain."

**Entropy vs Gini?**
"Both measure node impurity. Gini uses squares (1 - Σp²) — faster, no logarithm. Entropy uses logs (-Σp×log₂p) — slower but tends to give more balanced splits. In practice they give very similar results."

**How does a decision tree handle numerical features?**
"It tries all possible midpoints between consecutive sorted values as split thresholds. For each threshold it calculates the information gain. It selects the threshold with maximum gain. This makes numerical features more computationally expensive than categorical ones."

**Why do decision trees overfit?**
"A fully grown tree creates a leaf for every training sample — zero training error but poor generalisation. Controlled with max_depth, min_samples_split, min_samples_leaf, and pruning."

**What is geometric intuition behind a decision tree?**
"Decision trees create axis-aligned hyperplanes that cut the feature space into rectangular hyper-cuboids. Each region corresponds to a leaf node prediction. Unlike SVM or logistic regression, the boundaries are always parallel to the axes."
