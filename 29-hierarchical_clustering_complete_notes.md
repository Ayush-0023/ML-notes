# Hierarchical Clustering — Complete Notes

---

## Problems with K-Means That Motivate This Topic

K-Means relies entirely on distance-to-centroid to form clusters. This works well when clusters have clean, roughly spherical, well-separated boundaries. It fails on:
- **Concentric circles** — one cluster surrounding another
- **Crescent/moon-shaped clusters** — non-convex shapes
- **Clusters of very different sizes or densities**

This motivates two alternative approaches:
1. **Hierarchical Clustering** (this topic)
2. **Density-Based Clustering — DBSCAN** (next topic)

---

## Two Types of Hierarchical Clustering

### 1. Agglomerative Clustering (Bottom-Up)

Start with the assumption that **every point is its own cluster** (n points = n clusters). Then iteratively merge the two closest clusters together, recording each merge. This continues until only **one** giant cluster remains.

The complete merge history forms a tree-like structure called a **dendrogram**.

### 2. Divisive Clustering (Top-Down)

Works in the exact opposite direction. Starts with **one big cluster** containing all points, then recursively splits it into smaller clusters until each point is its own cluster.

### Why Agglomerative is Used More

**Merging is computationally easier than splitting.** To merge, you just need to find the two closest clusters — a relatively simple distance comparison. To split optimally, you'd need to consider all possible ways to partition a cluster — combinatorially much harder. This is why Agglomerative is the dominant approach in practice (and what sklearn implements as `AgglomerativeClustering`).

---

## The Agglomerative Algorithm

```
Step 1 — Initialise the Proximity Matrix (distances between all pairs of points)
Step 2 — Treat each point as its own cluster
Step 3 — Loop:
    a. Merge the 2 closest clusters
    b. Update the Proximity Matrix (recalculate distances involving the new merged cluster)
Step 4 — Continue until only 1 cluster remains
```

**The tricky part:** How do you measure "distance between two clusters" once clusters contain multiple points (not just single points)? This is where **Linkage Criteria** come in.

---

## Linkage Criteria — Measuring Distance Between Clusters

### 1. Single Linkage (Min / Nearest Point)

> distance(A, B) = min(distance between any point in A and any point in B)

Takes the closest pair of points across the two clusters.

**Effect:** Tends to create long, "chain-like" clusters. Can merge clusters that are connected by a thin bridge of points, even if the clusters' overall shapes are very different. Sensitive to noise.

### 2. Complete Linkage (Max / Farthest Point)

> distance(A, B) = max(distance between any point in A and any point in B)

Takes the farthest pair of points across the two clusters.

**Effect:** Tends to create compact, evenly-sized, spherical clusters. Less sensitive to noise than single linkage, but can break up genuinely elongated clusters incorrectly.

### 3. Average Linkage

> distance(A, B) = average(distance between all pairs of points across A and B)

A compromise between single and complete linkage.

**Effect:** Less extreme than either single or complete — moderate cluster shapes, moderate sensitivity to outliers.

### 4. Ward Linkage (Default in sklearn)

Calculates the centroid of each cluster, then the distance from the centroid to all points. Merges the two clusters whose merge results in the **smallest increase in total within-cluster variance**.

> Minimise: Σ(within-cluster variance) at each merge step

**Effect:** Tends to produce balanced, compact clusters similar in spirit to K-Means' objective (minimising within-cluster variance — same idea as WCSS in K-Means). This is why Ward linkage is the **default in sklearn** — it generally gives the most intuitive, well-balanced clustering results for typical datasets.

### Comparison Table

| Linkage | Distance Definition | Cluster Shape Tendency | Outlier Sensitivity |
|---|---|---|---|
| Single | Min distance | Long, chain-like | High |
| Complete | Max distance | Compact, spherical | Moderate |
| Average | Mean distance | Balanced | Moderate |
| Ward | Variance minimisation | Compact, balanced (like K-Means) | Low-moderate |

---

## The Dendrogram

A dendrogram visually represents the entire merge history as a tree.

```
Distance
   |
4  |              ___________________
   |             |                   |
3  |        _____|____           ____|____
   |       |          |         |         |
2  |    ___|___        |      ___|___      |
   |   |       |       |     |       |     |
1  |   A       B        C     D       E     F
       └───────────────────────────────────┘
                  Data points
```

- **X-axis:** individual data points
- **Y-axis:** the distance at which clusters were merged
- **Height of horizontal line:** the distance/dissimilarity at which two clusters merged
- **Reading bottom-up:** points merge into small clusters first (low height), which then merge into bigger clusters (higher height), until everything merges into one (top)

---

## Finding the Ideal Number of Clusters — Using the Dendrogram

Unlike K-Means (Elbow Method using WCSS), Hierarchical Clustering uses the **dendrogram itself** to decide K.

**Method:** Draw a horizontal line across the dendrogram. The number of vertical lines it crosses = the number of clusters at that cut height.

**How to choose where to cut:** Look for the **longest vertical line that isn't crossed by any horizontal merge** — i.e., find the biggest "jump" in merge distance. Cutting just below this jump gives the most natural number of clusters, since it means the next merge would combine clusters that are unusually far apart (and thus probably shouldn't be merged).

```python
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.title("Customer Dendrograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))

# Draw a horizontal line to visually decide cut point
plt.axhline(y=150, color='r', linestyle='--')  # adjust y based on visual inspection
plt.show()
```

---

## Hyperparameters

| Parameter | Meaning |
|---|---|
| `n_clusters` | Number of clusters to form (final cut point) |
| `affinity` / `metric` | Distance metric — 'euclidean', 'manhattan', 'cosine', etc. |
| `linkage` | 'single', 'complete', 'average', 'ward' (default) |

**Important constraint:** Ward linkage only works with Euclidean distance — if you choose `linkage='ward'`, `affinity` must be `'euclidean'`.

---

## Implementation

```python
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Scale first — distance-based, like K-Means
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 1 — Visualise dendrogram to decide K
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()

# Step 2 — Fit with chosen K
cluster = AgglomerativeClustering(
    n_clusters=5,
    affinity='euclidean',  # in newer sklearn versions, use metric='euclidean'
    linkage='ward'
)
labels = cluster.fit_predict(data_scaled)

# Visualise resulting clusters
plt.figure(figsize=(10, 7))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='rainbow')
plt.title("Agglomerative Clustering Result")
plt.show()
```

---

## Benefits and Limitations

**Benefits:**
- No need to pre-specify K (can decide after seeing the dendrogram)
- Produces a rich hierarchical structure — useful when you want clusters at multiple granularities
- Deterministic — no randomness like K-Means' centroid initialisation
- Works with any distance metric, not just Euclidean (except Ward linkage)

**Limitations:**
- **Computationally expensive on large datasets** — O(n²) or O(n³) depending on implementation, since you need pairwise distances between all points. This is the **biggest limitation**.
- Once a merge happens, it can't be undone (greedy algorithm — no backtracking)
- Sensitive to noise/outliers depending on linkage method (especially single linkage)
- Doesn't scale well — generally impractical beyond tens of thousands of points without optimisations

---

## Interview One-Liners

**What is Agglomerative Clustering?**
"A bottom-up hierarchical clustering method that starts with every point as its own cluster, then iteratively merges the two closest clusters until only one remains. The merge history forms a dendrogram, which can be cut at any height to get a desired number of clusters."

**Why is Agglomerative preferred over Divisive?**
"Merging two clusters just requires comparing distances — computationally simple. Splitting a cluster optimally requires considering all possible partitions — combinatorially much harder. This is why agglomerative is the dominant practical approach."

**What is Ward linkage and why is it the default?**
"Ward linkage merges the two clusters that result in the smallest increase in total within-cluster variance — similar in spirit to minimising WCSS in K-Means. It tends to produce compact, balanced clusters, making it the most generally useful default."

**How do you decide K in hierarchical clustering?**
"Using the dendrogram — look for the longest vertical line not crossed by a horizontal merge, which represents the biggest 'jump' in merge distance. Cutting just below this jump gives the most natural number of clusters."

**Biggest limitation of hierarchical clustering?**
"Computational cost — O(n²) or worse due to needing pairwise distances between all points, making it impractical for large datasets compared to K-Means or DBSCAN."
