from src.abstractions import StorageBase, ImageData, BoundingBox
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
            filename TEXT,
            embedding BLOB,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            bbox_conf REAL
        );""")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_filename ON images (filename);")
        self.cur.execute("""CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );""")
        self.con.commit()

    def __del__(self):
        self.cur.close()
        self.con.close()

    def save(self, data: ImageData) -> None:
        # Convert the embedding to bytes and save it in the database
        embedding_bytes = struct.pack(f"{len(data.embedding)}f", *data.embedding)
        self.cur.execute("""INSERT INTO images 
            (filename, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf) 
            VALUES (?, ?, ?, ?, ?, ?, ?);""",
            (data.filename, embedding_bytes, data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2, data.bbox.confidence))
        self.con.commit()

    def _row_to_image_data(self, row) -> ImageData:
        filename, embedding_bytes, x1, y1, x2, y2, conf = row
        embedding = list(struct.unpack(f"{len(embedding_bytes) // 4}f", embedding_bytes))
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
            
        return ImageData(filename=filename, embedding=embedding, bbox=bbox)

    def has(self, filename: str) -> bool:
        self.cur.execute("SELECT 1 FROM images WHERE filename = ?;", (filename,))
        return self.cur.fetchone() is not None

    def load(self, filename:str) -> list[ImageData]:
        self.cur.execute("SELECT filename, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf FROM images WHERE filename = ?;", (filename,))
        result = self.cur.fetchall()

        if result is None:
            raise ValueError(f"Image with filename '{filename}' not found in database.")

        return [self._row_to_image_data(row) for row in result]

    def get_all_images(self) -> list[ImageData]:
        self.cur.execute("SELECT filename, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf FROM images;")
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

    def get_by_distance(self, embedding: list[float], max_images: int = -1, cos_distance: bool = True) -> list[ImageData]:
        # Get all images first and calculate distance in Python (not efficient
        # but we can't do much more in sqlite)
        all_images = self.get_all_images()
        sorted_images = sorted(all_images, key=lambda img: self.distance(img.embedding, embedding, cos_distance))

        if max_images > 0:
            sorted_images = sorted_images[:max_images]

        return sorted_images

    def set_metadata(self, key: str, value: str) -> None:
        self.cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?);", (key, value))
        self.con.commit()

    def get_metadata(self, key: str) -> str | None:
        self.cur.execute("SELECT value FROM metadata WHERE key = ?;", (key,))
        result = self.cur.fetchone()
        return result[0] if result else None

    def clear(self) -> None:
        self.cur.execute("DELETE FROM images;")
        self.cur.execute("DELETE FROM metadata;")
        self.con.commit()

if __name__ == "__main__":
    storage = SQLiteStorage("test.db")
    image = PILImage.open("test.png").convert("RGB")
    storage.save(ImageData(filename="test.png", embedding=[0.1, 0.2, 0.3]))
    print(storage.load("test.png"))
    print(storage.get_all_images())
    print(storage.get_by_distance([0.1, 0.2, 0.0]))


