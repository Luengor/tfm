from src.abstractions import StorageBase, ImageData 
from PIL import Image as PILImage
import numpy as np
import struct
import sqlite3

class SQLiteStorage(StorageBase):
    def __init__(self, db_path: str):
        self.db_path = db_path

        # Init db
        self.con = sqlite3.connect(self.db_path)
        self.cur = self.con.cursor()
        self.cur.execute("""CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            embedding BLOB
        );""")
        self.con.commit()

    def __del__(self):
        self.cur.close()
        self.con.close()

    def save(self, data: ImageData) -> None:
        # Convert the embedding to bytes and save it in the database
        embedding_bytes = struct.pack(f"{len(data.embedding)}f", *data.embedding)

        self.cur.execute("INSERT OR REPLACE INTO images (filename, embedding) VALUES (?, ?);",
                            (data.filename, embedding_bytes))
        self.con.commit()

    def _row_to_image_data(self, row) -> ImageData:
        embedding_bytes = row[1]
        embedding = list(struct.unpack(f"{len(embedding_bytes) // 4}f", embedding_bytes))
        return ImageData(filename=row[0], embedding=embedding)

    def has(self, filename: str) -> bool:
        self.cur.execute("SELECT 1 FROM images WHERE filename = ?;", (filename,))
        return self.cur.fetchone() is not None

    def load(self, filename:str) -> ImageData:
        self.cur.execute("SELECT filename, embedding FROM images WHERE filename = ?;", (filename,))
        result = self.cur.fetchone()

        if result is None:
            raise ValueError(f"Image with filename '{filename}' not found in database.")

        return self._row_to_image_data(result)
    
    def get_all_images(self) -> list[ImageData]:
        self.cur.execute("SELECT filename, embedding FROM images;")
        results = self.cur.fetchall()

        images = []
        for row in results:
            images.append(self._row_to_image_data(row))
        return images

    @staticmethod
    def distance(emb1: list[float], emb2: list[float], cos_distance: bool) -> float:
        if cos_distance:
            return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        else:
            # Euclidean distance
            return float(np.linalg.norm(np.array(emb1) - np.array(emb2)))

    def get_by_distance(self, embedding: list[float], cos_distance: bool = True) -> list[ImageData]:
        # Get all images first and calculate distance in Python (not efficient
        # but we can't do much more in sqlite)
        all_images = self.get_all_images()
        sorted_images = sorted(all_images, key=lambda img: self.distance(img.embedding, embedding, cos_distance))
        return sorted_images

if __name__ == "__main__":
    storage = SQLiteStorage("test.db")
    image = PILImage.open("test.png").convert("RGB")
    storage.save(ImageData(filename="test.png", embedding=[0.1, 0.2, 0.3]))
    print(storage.load("test.png"))
    print(storage.get_all_images())
    print(storage.get_by_distance([0.1, 0.2, 0.0]))


