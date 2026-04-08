from src.abstractions import StorageBase, EmbeddingBase, ClusteringBase, ImageData
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

    def get_by_distance(self, filename: str) -> list[ImageData]:
        # Load the target image
        target_data = self.storage.load(filename)

        # Get similar images by distance
        similar_images = self.storage.get_by_distance(target_data.embedding)

        return similar_images

    def cluster_images(self, **kwargs) -> list[int]:
        # Get all images from storage
        images = self.storage.get_all_images()

        # Cluster the images
        clusters = self.clustering.cluster(images, **kwargs)

        return clusters


if __name__ == "__main__":
    from src.storage.sqlite import SQLiteStorage
    from src.embedding.embeddings import get_model, EmbeddingModelNames
    from src.cluster.cluster import KMeansClusterer
    from sys import argv
    from tqdm import tqdm

    config = Configuration(
            SQLiteStorage("test.db"),
            get_model(EmbeddingModelNames.YOLOn),
            KMeansClusterer()
    )

    for filename in tqdm(argv[1:]):
        try:
            config.save_image(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    clusters = config.cluster_images(n_clusters=3)
    print(clusters)
    print([image.filename for image in config.get_by_distance(argv[1])])

