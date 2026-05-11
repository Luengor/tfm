# Dimensionality reduction

Dimensionality reduction is applied to embeddings before clustering. It is optional: omitting
the `reduction` key (or using `"identity"`) passes the original embeddings directly to the
clusterer. Reduction parameters are forwarded to the class constructor.

---

## PCA

**Type string:** `pca`

Linear dimensionality reduction using Principal Component Analysis. Projects embeddings onto
the directions of maximum variance. Useful as a fast pre-processing step before clustering or
as a preprocessing step before UMAP.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_components` | `int` | `50` | Number of principal components to retain. Automatically capped at the number of samples. |

**Example:**
```json
{
  "type": "pca",
  "params": { "n_components": 50 }
}
```

---

## UMAP

**Type string:** `umap`

Non-linear manifold reduction that preserves local neighbourhood structure. The canonical
choice before density-based clustering (e.g. HDBSCAN). Performs a one-time JIT warm-up on
first instantiation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_components` | `int` | `2` | Number of output dimensions. |
| `n_neighbors` | `int` | `15` | Size of the local neighbourhood used to build the manifold graph. Larger values capture more global structure. |
| `min_dist` | `float` | `0.1` | Minimum distance between embedded points. Lower values produce tighter, more clustered layouts. |
| `metric` | `str` | `"cosine"` | Distance metric used in the input space. `"cosine"` is recommended for L2-normalized embeddings. |

**Example:**
```json
{
  "type": "umap",
  "params": { "n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine" }
}
```

---

## Isomap

**Type string:** `isomap`

Non-linear reduction that approximates geodesic distances on the data manifold via a k-nearest-
neighbour graph. Useful when the data lies on a curved low-dimensional surface.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_components` | `int` | `2` | Number of output dimensions. |
| `n_neighbors` | `int` | `5` | Number of neighbours used to build the manifold graph. Automatically capped at `n_samples - 1`. |

**Example:**
```json
{
  "type": "isomap",
  "params": { "n_components": 2, "n_neighbors": 10 }
}
```

---

## Kernel PCA

**Type string:** `kernel_pca`

Non-linear extension of PCA using the kernel trick. Projects data into a high-dimensional
feature space (implicitly) and then applies PCA. The choice of kernel determines what structure
is preserved.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_components` | `int` | `50` | Number of components to retain. Automatically capped at the number of samples. |
| `kernel` | `str` | `"rbf"` | Kernel function: `"rbf"`, `"linear"`, `"poly"`, `"sigmoid"`, or `"cosine"`. |

**Example:**
```json
{
  "type": "kernel_pca",
  "params": { "n_components": 50, "kernel": "rbf" }
}
```

---

## Identity (no reduction)

**Type string:** `identity`

Passthrough — returns the original embeddings unchanged. Equivalent to omitting the `reduction`
key entirely. Use when you want to cluster in the full embedding space without any reduction.

| Parameter | Type | Default | Description |
|---|---|---|---|
| *(none)* | — | — | No parameters. |

**Example:**
```json
{
  "type": "identity",
  "params": {}
}
```
