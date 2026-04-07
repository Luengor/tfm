from abstractions import StorageBase, EmbeddingBase, ClusteringBase, ImageData
from PIL import Image as PILImage

class Configuration:
    def __init__(self, storage: StorageBase, embedding: EmbeddingBase,
                 clustering: ClusteringBase):
        self.storage = storage
        self.embedding = embedding
        self.clustering = clustering

    def save_image(self, filename: str) -> ImageData:
        # Get the image embedding
        image = PILImage.open(filename).convert("RGB")
        emb = self.embedding.gen_embedding(image)

        # Save the image data
        data = ImageData(filename=filename, embedding=emb)
        self.storage.save(data)

        return data

    def cluster_images(self, **kwargs) -> list[int]:
        # Get all images from storage
        images = self.storage.get_all_images()

        # Cluster the images
        clusters = self.clustering.cluster(images, **kwargs)

        return clusters


