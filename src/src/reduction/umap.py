import umap
import numpy as np
from src.abstractions import ReductionBase

class UMAPReduction(ReductionBase):
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'cosine'):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )

    def reduce(self, embeddings: list[list[float]]) -> list[list[float]]:
        if not embeddings:
            return []
        
        data = np.array(embeddings)
        # UMAP needs enough samples
        n_samples = data.shape[0]
        if n_samples <= self.n_neighbors:
             # Adjust n_neighbors if too few samples
             self.reducer.n_neighbors = max(2, n_samples - 1)
        
        reduced_data = self.reducer.fit_transform(data)
        return reduced_data.tolist()
