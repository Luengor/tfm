from src.abstractions import ReductionBase

class IdentityReduction(ReductionBase):
    """
    A reduction step that does nothing, returning the original embeddings.
    Used to standardize the pipeline when no actual reduction is desired.
    """
    is_deterministic = True

    def reduce(self, embeddings: list[list[float]]) -> list[list[float]]:
        return embeddings
