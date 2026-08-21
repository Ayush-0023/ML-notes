# K-Means Clustering — Complete Notes

---

## What is Clustering?

Clustering is an **unsupervised** machine learning technique that groups similar data points together based on their characteristics, **without using predefined labels**.

**Core principle:** Points within the same cluster should be more similar to each other than to points in other clusters, according to some similarity/distance measure (Euclidean distance, cosine similarity, etc.)

**Real-world examples:**
- Grouping customers into segments based on buying patterns (customer segmentation)
- Grouping news articles by topic
- Grouping similar images together
- Anomaly detection (points that don't fit any cluster well)

**Popular clustering algorithms:** K-Means, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models.

**Key distinction from classification:** Classification has predefined labels (supervised). Clustering discovers groupings on its own (unsupervised) — there's no "correct answer" to check against during training.

---

## What is K-Means Clustering?

K-Means is an unsupervised algorithm that partitions data into **K distinct, non-overlapping clusters**, where K is specified in advance by the user.

---

## The K-Means Algorithm — Step by Step

### Step 1 — Decide the Number of Clusters (K)

Choose the desired number of clusters. (How to choose K is covered below — Elbow Method.)

### Step 2 — Initialise Centroids

Randomly select K points from the dataset as initial centroids (cluster centers).

### Step 3 — Assign Points to Nearest Centroid

For each data point, calculate distance (typically Euclidean) to all K centroids. Assign the point to the cluster whose centroid is closest.

> distance = √(Σ(xᵢ - centroidᵢ)²)

This creates K groups based on current centroid positions.

### Step 4 — Update Centroids

For each cluster, compute the mean position of ALL points currently assigned to it. This mean becomes the new centroid.

> new_centroid = (1/n) × Σ(points in cluster)

### Step 5 — Repeat Until Convergence

Compare new centroids (Step 4) to previous centroids. 
- If they're unchanged (or change is below a threshold) → **stop**, algorithm has converged
- Otherwise → go back to Step 3 with the updated centroids

### Why Do Centroids Change Between Step 3 and Step 4?

In Step 3, clusters are formed based on **current** centroid positions — which initially are just random points, not necessarily good representatives of any true cluster structure.

In Step 4, we calculate the actual mean of points that ended up in each cluster — this mean is generally different from the original random centroid, because it now reflects the actual data distribution, not a random guess.

**This iterative correction is exactly how K-Means "learns"** — each round, centroids move closer to the true center of their naturally forming cluster, and points get reassigned more accurately, converging toward a stable solution.

---

## Why Do We Need K-Means?

**In low dimensions (2D, 3D):** You could potentially identify clusters just by plotting and visually inspecting the data.

**In high dimensions (e.g., 100 features):** Visual inspection becomes impossible — humans can't perceive more than 3 dimensions intuitively.

**K-Means provides an algorithmic solution** that works regardless of dimensionality. Euclidean distance (its core metric) is mathematically well-defined in any number of dimensions, so the algorithm scales to high-dimensional data where visualisation completely fails.

---

## Choosing K — The Elbow Method

### What is WCSS (Within-Cluster Sum of Squares)?

Also called **inertia**. For a single cluster:
1. Take the centroid
2. Calculate the distance of each point in that cluster from the centroid: d₁, d₂, d₃, ...
3. Square each distance and sum them:

> **WCSS(one cluster) = Σ dᵢ²**

For the **entire dataset** with K clusters, sum the WCSS across all K clusters:

> **Total WCSS = Σₖ Σᵢ (distance of point i from its cluster k's centroid)²**

### How the Elbow Method Works

1. Run K-Means with K=1 (entire dataset as one cluster) → compute WCSS₁
2. Run K-Means with K=2 → compute WCSS₂
3. Continue increasing K, computing WCSSₖ each time
4. Plot K (x-axis) vs WCSS (y-axis)

### Key Property — WCSS Always Decreases as K Increases

As K increases, clusters become smaller and points get closer to their centroids — so WCSS monotonically decreases.

**Extreme case:** If K = number of data points (every point is its own cluster), WCSS = 0 (every point is its own centroid, distance = 0).

This means you can't just pick the K that minimises WCSS — that would always be the maximum possible K, which is useless (no actual grouping/insight).

### The Elbow Point

```
WCSS
    |●
    |  ●
    |     ●
    |          ●___●___●___●  ← curve flattens here
    +─────────────────────────→ K
         ↑
    "elbow" point — optimal K
```

The elbow point is where the WCSS curve **stops decreasing sharply and starts to level off** — adding more clusters beyond this point gives diminishing returns. This is the optimal K — enough clusters to capture real structure, without over-segmenting into meaningless tiny groups.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # sklearn calls WCSS "inertia"

plt.plot(K_range, wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method')
plt.show()
# Visually identify the elbow point
```

---

## Alternative Method — Silhouette Score

The Elbow Method requires visual judgment, which can be ambiguous. **Silhouette Score** gives a quantitative measure.

> **Silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))**

Where:
- a(i) = average distance from point i to other points in the SAME cluster (cohesion)
- b(i) = average distance from point i to points in the NEAREST different cluster (separation)

**Range:** -1 to +1
- Close to +1 → point is well-matched to its own cluster, far from others (good)
- Close to 0 → point is on the border between two clusters
- Close to -1 → point may be in the wrong cluster

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):  # silhouette needs at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()
# Choose K with the highest silhouette score
```

**Elbow vs Silhouette:** Elbow is faster to compute but requires subjective visual judgment. Silhouette gives an objective numeric score but is more computationally expensive (O(n²) pairwise distances).

---

## K-Means++ — Smarter Initialisation

**The problem with random initialisation (Step 2):** If initial centroids are randomly placed badly (e.g., all clustered close together), K-Means can converge to a poor local optimum, or take many more iterations to converge.

**K-Means++ fix:** Choose initial centroids more intelligently:
1. Pick the first centroid randomly from the data points
2. For each subsequent centroid, pick a point with probability proportional to its squared distance from the nearest existing centroid (points far from existing centroids are more likely to be chosen)
3. Repeat until K centroids are chosen

This spreads initial centroids out more evenly, leading to faster and more reliable convergence.

```python
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
# 'k-means++' is actually the sklearn default — much better than 'random'
```

---

## K-Means From Scratch (Simplified Implementation)

```python
import numpy as np

class SimpleKMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n_samples = X.shape[0]

        # Step 2 — Randomly initialise centroids
        random_idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for iteration in range(self.max_iters):
            # Step 3 — Assign points to nearest centroid
            distances = np.array([
                np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids
            ])
            self.labels = np.argmin(distances, axis=0)

            # Step 4 — Update centroids
            new_centroids = np.array([
                X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)
            ])

            # Step 5 — Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X):
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0)
```

---

## sklearn Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Always scale features — K-Means uses Euclidean distance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    init='k-means++',     # smart initialisation (default)
    n_init=10,             # run algorithm 10 times with different seeds, keep best
    max_iter=300,
    random_state=42
)
kmeans.fit(X_scaled)

labels = kmeans.labels_          # cluster assignment for each point
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_        # WCSS

# Predict cluster for new points
new_labels = kmeans.predict(X_new_scaled)
```

**Why scaling matters here too:** K-Means relies entirely on Euclidean distance, just like KNN. Unscaled features with larger ranges will dominate distance calculations and distort cluster formation.

---

## Limitations of K-Means

**1. Must specify K in advance**
Unlike some algorithms (DBSCAN), you need to know/guess the number of clusters beforehand. Mitigated by Elbow Method / Silhouette Score.

**2. Assumes spherical, equally-sized clusters**
K-Means works best when clusters are roughly circular/spherical and similar in size. It struggles with elongated, irregular, or very differently-sized clusters.

**3. Sensitive to initialisation**
Even with K-Means++, poor luck can lead to suboptimal results. `n_init` parameter (running multiple times) mitigates this.

**4. Sensitive to outliers**
Outliers can pull centroids away from the true cluster center, since centroids are means (sensitive to extreme values, just like the mean in statistics).

**5. Sensitive to feature scaling**
Same issue as KNN — must scale features before applying K-Means.

**6. Struggles with non-convex shapes**
Clusters that aren't roughly circular (e.g., crescents, rings) are poorly captured by K-Means. DBSCAN handles these much better.

---

## K-Means vs Other Clustering Methods (Preview)

| | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|
| **Need to specify K?** | Yes | No (cut dendrogram) | No |
| **Cluster shape assumption** | Spherical | Flexible | Arbitrary (density-based) |
| **Handles outliers** | Poorly | Moderately | Well (marks as noise) |
| **Scalability** | Good (fast) | Poor (O(n²) or worse) | Good |
| **Deterministic** | No (depends on init) | Yes | Yes |

---

## Interview One-Liners

**What is K-Means clustering?**
"An unsupervised algorithm that partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids to the mean of their assigned points, until convergence."

**How do you choose K?**
"Elbow Method — plot WCSS vs K, look for the point where the decrease in WCSS sharply levels off. Alternatively, Silhouette Score gives a quantitative measure of how well-separated clusters are, and you pick the K maximising it."

**Why does WCSS always decrease as K increases?**
"More clusters mean smaller clusters, so points are closer to their centroids on average. At the extreme (K = n), every point is its own cluster with WCSS = 0. This is why you can't simply minimise WCSS — you need the elbow point, not the global minimum."

**What is K-Means++?**
"A smarter centroid initialisation strategy that selects initial centroids spread far apart from each other (probabilistically, based on distance from existing centroids), rather than purely random selection — leading to faster, more reliable convergence."

**Limitations of K-Means?**
"Requires specifying K upfront, assumes roughly spherical and similarly-sized clusters, sensitive to outliers (since centroids are means), sensitive to feature scaling, and struggles with non-convex cluster shapes — DBSCAN handles these cases better."

**Why is feature scaling necessary for K-Means?**
"K-Means relies entirely on Euclidean distance to assign points to clusters. Without scaling, features with larger numeric ranges dominate the distance calculation, distorting which points end up grouped together — same issue as KNN."
