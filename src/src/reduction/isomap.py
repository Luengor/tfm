from sklearn.manifold import Isomap
import numpy as np
from src.abstractions import ReductionBase

class IsomapReduction(ReductionBase):
    is_deterministic = True

    def __init__(self, n_components: int = 2, n_neighbors: int = 5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)

    def reduce(self, embeddings: list[list[float]]) -> list[list[float]]:
        if not embeddings:
            return []
        
        data = np.array(embeddings)
        n_samples = data.shape[0]
        
        # Isomap requires n_neighbors < n_samples
        actual_neighbors = min(n_samples - 1, self.n_neighbors)
        if actual_neighbors < 1:
            return data.tolist()

        if actual_neighbors != self.isomap.n_neighbors:
            self.isomap = Isomap(n_components=self.n_components, n_neighbors=actual_neighbors)
            
        reduced_data = self.isomap.fit_transform(data)
        return reduced_data.tolist()
