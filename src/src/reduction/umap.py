import umap
import numpy as np
from src.abstractions import ReductionBase

class UMAPReduction(ReductionBase):
    _is_warmed_up = False

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
        self._warmup_if_needed()

    def _warmup_if_needed(self):
        if not UMAPReduction._is_warmed_up:
            # Perform a small dummy run to trigger JIT compilation
            # Use small data to minimize overhead but enough to trigger compilation
            dummy_data = np.random.random((20, 10)).astype(np.float32)
            warmup_reducer = umap.UMAP(n_neighbors=5, n_components=2)
            warmup_reducer.fit_transform(dummy_data)
            UMAPReduction._is_warmed_up = True

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
