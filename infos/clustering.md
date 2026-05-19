# Clustering algorithms

Clustering algorithms group image embeddings into clusters after optional dimensionality
reduction. The `clustering` key in a configuration run selects the algorithm.

> **Note on parameters.** Unlike storage, embedding, and reduction, clustering parameters are
> **not** passed into the constructor. They are forwarded as `**kwargs` at cluster time
> (`ClusteringBase.cluster(images, **params)`). Specify them under `"params"` in the configuration
> run exactly as shown in each example.

---

## KMeans

**Type string:** `kmeans`

Partitions data into k spherical clusters by iteratively assigning each point to the nearest
centroid and recomputing centroids. When `n_clusters` is omitted the optimal k is chosen
automatically by finding the elbow in the inertia curve.

**Pros:**
- Fast and scalable; works well on large datasets.
- Deterministic result (fixed `random_state`).
- Intuitive and easy to interpret.

**Cons:**
- Assumes convex, roughly equal-sized, spherical clusters — poor fit for irregular shapes.
- Sensitive to outliers, which pull centroids away from the cluster core.
- Distance metric is Euclidean; cosine similarity between embeddings is not captured.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | `int` | *(auto)* | Number of clusters. When omitted, chosen automatically via inertia elbow. |
| `max_clusters` | `int` | `20` | Upper bound on k when auto-detecting. Only used when `n_clusters` is not set. |

**Example (fixed k):**
```
{
  "type": "kmeans",
  "params": { "n_clusters": 8 }
}
```

**Example (auto-detect up to 30):**
```
{
  "type": "kmeans",
  "params": { "max_clusters": 30 }
}
```

---

## DBSCAN

**Type string:** `dbscan`

Groups tightly packed points and marks low-density points as noise (label `-1`). The
neighborhood radius `eps` is auto-detected from the k-distance elbow when not provided.

**Pros:**
- Does not require specifying the number of clusters.
- Naturally identifies and isolates noise/outliers (label `-1`).
- Discovers clusters of arbitrary shape, not limited to convex regions.

**Cons:**
- Sensitive to `eps` and `min_samples`; wrong values produce one giant cluster or all noise.
- Struggles with clusters of varying density.
- High-dimensional embeddings cause the curse of dimensionality — dimensionality reduction
  beforehand is strongly recommended.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eps` | `float` | *(auto)* | Neighborhood radius. When omitted, chosen via k-distance elbow. |
| `min_samples` | `int` | `5` | Minimum points required to form a dense region. |

**Example (auto eps):**
```
{
  "type": "dbscan",
  "params": { "min_samples": 5 }
}
```

**Example (fixed eps):**
```
{
  "type": "dbscan",
  "params": { "eps": 0.4, "min_samples": 3 }
}
```

---

## HDBSCAN

**Type string:** `hdbscan`

Extends DBSCAN by building a full cluster hierarchy across all density levels and extracting
the most stable clusters. Handles clusters of varying density and requires only
`min_cluster_size`.

**Pros:**
- Handles clusters of varying density — a major limitation of plain DBSCAN.
- Only one intuitive parameter (`min_cluster_size`) is usually needed.
- Robust to noise; assigns outliers label `-1`.
- Generally outperforms DBSCAN in practice on embedding spaces.

**Cons:**
- More memory-intensive than DBSCAN due to hierarchy construction.
- Still susceptible to the curse of dimensionality in very high-dimensional spaces.
- Non-deterministic cluster boundaries can shift between runs when data changes slightly.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_cluster_size` | `int` | `5` | Minimum number of points to form a cluster. |
| `max_cluster_size` | `int` | *(none)* | Upper bound on cluster size. No limit when omitted. |

**Example:**
```
{
  "type": "hdbscan",
  "params": { "min_cluster_size": 10 }
}
```

---

## OPTICS

**Type string:** `optics`

Produces a reachability-distance ordering that encodes density structure at all radius values
simultaneously, then extracts clusters from this ordering. An extension of DBSCAN that handles
variable-density clusters without fixing `eps`.

**Pros:**
- No need to choose `eps`; `max_eps` only limits the search radius.
- Handles clusters of widely varying densities in a single pass.
- The default cosine metric is well-suited to normalized embedding vectors.

**Cons:**
- Slower than DBSCAN and HDBSCAN on large datasets.
- Cluster extraction can be sensitive to `xi` and `min_samples`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_samples` | `int` | `5` | Minimum points required to form a core point. |
| `max_eps` | `float` | `inf` | Maximum neighborhood radius. Unlimited by default. |
| `metric` | `str` | `"cosine"` | Distance metric. Defaults to cosine (differs from sklearn default `"minkowski"`). |

**Example:**
```
{
  "type": "optics",
  "params": { "min_samples": 5, "metric": "cosine" }
}
```

---

## Agglomerative

**Type string:** `agglomerative`

Bottom-up hierarchical clustering: starts with each point as its own cluster and iteratively
merges the two closest until the target count is reached.

**Pros:**
- No assumptions about cluster shape; elongated clusters work with `complete` or `average` linkage.
- Deterministic; produces the same result every run.

**Cons:**
- Requires specifying `n_clusters` up front (no auto-detection).
- O(n² log n) time and O(n²) memory — does not scale beyond ~10k points.
- Default `ward` linkage minimises within-cluster variance, implicitly favouring convex clusters.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | `int` | `5` | Number of clusters to produce. |
| `linkage` | `str` | `"ward"` | Linkage criterion: `"ward"`, `"complete"`, `"average"`, or `"single"`. |

**Example:**
```
{
  "type": "agglomerative",
  "params": { "n_clusters": 10, "linkage": "average" }
}
```

---

## Spectral

**Type string:** `spectral`

Constructs a similarity graph from the data, then clusters the low-dimensional eigenvectors of
the graph Laplacian. Finds clusters defined by connectivity rather than compactness.

**Pros:**
- Finds non-convex, manifold-shaped clusters that distance-based methods miss.
- The nearest-neighbors affinity adapts naturally to embedding spaces.

**Cons:**
- Requires specifying `n_clusters`.
- O(n³) eigen-decomposition — impractical for datasets larger than a few thousand points.
- Sensitive to `affinity` and `n_neighbors`; wrong settings produce degenerate results.
- Non-deterministic due to random graph construction.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | `int` | `5` | Number of clusters. |
| `affinity` | `str` | `"nearest_neighbors"` | Affinity method: `"nearest_neighbors"` or `"rbf"`. |

**Example:**
```
{
  "type": "spectral",
  "params": { "n_clusters": 7, "affinity": "nearest_neighbors" }
}
```

---

## GMM

**Type string:** `gmm`

Models the data as a mixture of k multivariate Gaussian distributions fitted via
Expectation-Maximisation. Hard labels are the component with the highest posterior. When
`n_clusters` is omitted, k is chosen by minimising the Bayesian Information Criterion (BIC).

**Pros:**
- Soft assignments capture uncertainty near cluster boundaries.
- Flexible covariance allows elongated clusters.
- BIC-based auto-detection is more statistically principled than the KMeans elbow heuristic.

**Cons:**
- Assumes data is generated by Gaussian components — fails for ring-shaped or manifold data.
- EM can converge to local optima; results vary without a fixed `random_state`.
- Full covariance requires O(d²) parameters per component — expensive in high dimensions.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | `int` | *(auto)* | Number of Gaussian components. When omitted, chosen via BIC minimisation. |
| `covariance_type` | `str` | `"full"` | Covariance structure: `"full"`, `"tied"`, `"diag"`, or `"spherical"`. |
| `max_clusters` | `int` | `20` | Upper bound on k when auto-detecting. Only used when `n_clusters` is not set. |

**Example (auto-detect):**
```
{
  "type": "gmm",
  "params": { "covariance_type": "full" }
}
```

**Example (fixed k):**
```
{
  "type": "gmm",
  "params": { "n_clusters": 6, "covariance_type": "diag" }
}
```

---

## Affinity Propagation

**Type string:** `affinity_propagation`

Points exchange "responsibility" and "availability" messages until a set of exemplars (cluster
centres chosen from the data itself) emerges. The number of clusters is determined automatically
by the `preference` parameter.

**Pros:**
- Automatically determines the number of clusters from the data.
- Cluster centres are real data points (exemplars), aiding interpretability.
- Works well when the natural number of clusters is unknown and cannot be estimated.

**Cons:**
- O(n²) memory and O(n² · iterations) time — infeasible for more than a few thousand points.
- Very sensitive to `preference`; small changes can dramatically alter results.
- Often produces too many small clusters on embedding data unless `preference` is tuned carefully.
- Convergence is not guaranteed.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `damping` | `float` | `0.5` | Damping factor in [0.5, 1.0) to avoid numerical oscillations. |
| `preference` | `float` | *(auto)* | Shared preference value. Higher → more clusters. Defaults to the median input similarity when omitted. |

**Example:**
```
{
  "type": "affinity_propagation",
  "params": { "damping": 0.7, "preference": -50 }
}
```
