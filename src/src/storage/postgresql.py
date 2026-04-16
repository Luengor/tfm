from src.abstractions import StorageBase, ImageData
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

class Base(DeclarativeBase):
    pass

class ImageModel(Base):
    __tablename__ = 'images'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(unique=True)
    embedding: Mapped[Vector] = mapped_column(Vector)

class PostgreSQLStorage(StorageBase):
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

        # Init db
        self.connection = self.engine.connect()
        self.session = Session(self.connection)

        self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        self.session.commit()

        Base.metadata.create_all(self.engine)

    def __del__(self):
        self.session.close()
        self.connection.close()
        self.engine.dispose()

    def save(self, data: ImageData) -> None:
        image = ImageModel(filename=data.filename, embedding=data.embedding)
        self.session.add(image)
        self.session.commit()

    def has(self, filename: str) -> bool:
        query = self.session.query(ImageModel).filter_by(filename=filename)
        return self.session.query(query.exists()).scalar()

    @staticmethod
    def _2imagedata(image: ImageModel) -> ImageData:
        return ImageData(filename=image.filename, embedding=list(image.embedding))

    def load(self, filename:str) -> ImageData:
        result = self.session.query(ImageModel).filter_by(filename=filename).first()
        if result is None:
            raise ValueError(f"Image with filename '{filename}' not found in database.")
        return self._2imagedata(result)

    def get_all_images(self) -> list[ImageData]:
        return [self._2imagedata(image) for image in
                self.session.query(ImageModel).all()]

    def get_by_distance(self, embedding: list[float], cos_distance: bool = True) -> list[ImageData]:
        if cos_distance:
            distance_expr = ImageModel.embedding.cosine_distance(embedding).label("distance")
        else:
            distance_expr = ImageModel.embedding.l2_distance(embedding).label("distance")
        
        query = self.session.query(ImageModel, distance_expr).order_by(distance_expr)
        return [self._2imagedata(image) for image, _ in query.all()]

