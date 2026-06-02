import umap
import numpy as np
from src.abstractions import ReductionBase

class UMAPReduction(ReductionBase):
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'cosine', random_state: int | None = None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            n_jobs=1,
        )

    def reduce(self, embeddings: list[list[float]]) -> list[list[float]]:
        if not embeddings:
            return []

        data = np.array(embeddings)
        n_samples = data.shape[0]
        if n_samples <= self.n_neighbors:
            reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=max(2, n_samples - 1),
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=self.random_state,
                n_jobs=1,
            )
        else:
            reducer = self.reducer

        reduced_data = reducer.fit_transform(data)
        return reduced_data.tolist()
