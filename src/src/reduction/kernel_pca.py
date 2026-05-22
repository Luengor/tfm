from sklearn.decomposition import KernelPCA
import numpy as np
from src.abstractions import ReductionBase

class KernelPCAReduction(ReductionBase):
    is_deterministic = True

    def __init__(self, n_components: int = 50, kernel: str = "rbf"):
        self.n_components = n_components
        self.kernel = kernel
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel)

    def reduce(self, embeddings: list[list[float]]) -> list[list[float]]:
        if not embeddings:
            return []
        
        data = np.array(embeddings)
        n_samples = data.shape[0]
        
        actual_n = min(n_samples, self.n_components)
        
        if actual_n != self.kpca.n_components:
            self.kpca = KernelPCA(n_components=actual_n, kernel=self.kernel)
            
        reduced_data = self.kpca.fit_transform(data)
        return reduced_data.tolist()
