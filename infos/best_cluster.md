Based on the properties of neural network embeddings and the graffiti domain,
here's an analysis ranked from most to least suited:

## Best choices

**1. HDBSCAN (strongest recommendation)** Graffiti naturally splits into
clusters of very different sizes and densities — common tag styles vs. rare
mural techniques. HDBSCAN handles varying-density clusters natively,
auto-detects the number of clusters, and labels ambiguous images as noise (-1)
rather than forcing them into a group. The canonical pipeline for embedding
spaces is **UMAP → HDBSCAN**, and it's well-validated in the literature. The
only parameter to tune is `min_cluster_size`.

**2. KMeans** The standard baseline. After UMAP reduction the embedding space
often becomes roughly spherical and well-separated, which is exactly where
KMeans excels. It's fast, deterministic, and easy to evaluate. Use it as the
reference to beat.

**3. GMM** Useful when graffiti styles genuinely overlap (e.g. a throw-up
that's also a tag). Soft probability assignments reflect real ambiguity better
than hard labels, and BIC-based k selection is more statistically principled
than the elbow heuristic.

## Situational

**4. OPTICS** Like HDBSCAN with multi-scale density reasoning. The cosine
metric (default in your implementation) is actually well-suited to normalised
embedding vectors. Slower than HDBSCAN but a valid alternative when you want an
ordering-based analysis.

**5. Agglomerative** Good for small datasets (≤ few thousand images) or
exploratory work — no distance assumptions with `average`/`complete` linkage.
Doesn't scale: O(n²) memory makes it impractical if the dataset grows.

## Avoid or use with caution

**6. DBSCAN** — strictly dominated by HDBSCAN for this use case; the single
fixed `eps` breaks as soon as cluster densities differ.

**7. Spectral** — O(n³) eigen-decomposition won't scale, and the
nearest-neighbors affinity is fragile in high-dimensional pre-reduction space.

**8. Affinity Propagation** — O(n²) memory, tendency to produce too many
micro-clusters on embedding data, and very sensitive to `preference`. Not
practical here.

---

**Practical takeaway:** the two combinations most worth benchmarking are **UMAP
+ HDBSCAN** (density-aware, no k needed) and **UMAP + KMeans** (fast,
interpretable baseline). GMM is worth a run if you want soft assignments. The
rest are mainly useful for ablation or small-scale experimentation.
