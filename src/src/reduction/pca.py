from sklearn.decomposition import PCA
import numpy as np
from src.abstractions import ReductionBase

class PCAReduction(ReductionBase):
    is_deterministic = True

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def reduce(self, embeddings: list[list[float]]) -> list[list[float]]:
        if not embeddings:
            return []
        
        data = np.array(embeddings)
        # PCA requires at least n_samples > n_components
        n_samples = data.shape[0]
        actual_n = min(n_samples, self.n_components)
        
        if actual_n != self.pca.n_components:
            self.pca = PCA(n_components=actual_n)
            
        reduced_data = self.pca.fit_transform(data)
        return reduced_data.tolist()
