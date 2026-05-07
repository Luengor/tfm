from src.abstractions import StorageBase, ImageData, BoundingBox
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
    bbox_x1: Mapped[float] = mapped_column(nullable=True)
    bbox_y1: Mapped[float] = mapped_column(nullable=True)
    bbox_x2: Mapped[float] = mapped_column(nullable=True)
    bbox_y2: Mapped[float] = mapped_column(nullable=True)
    bbox_conf: Mapped[float] = mapped_column(nullable=True)

class MetadataModel(Base):
    __tablename__ = 'metadata'

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column()

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
        bbox_data = {}
        if data.bbox:
            bbox_data = {
                'bbox_x1': data.bbox.x1,
                'bbox_y1': data.bbox.y1,
                'bbox_x2': data.bbox.x2,
                'bbox_y2': data.bbox.y2,
                'bbox_conf': data.bbox.confidence
            }
        
        image = ImageModel(filename=data.filename, embedding=data.embedding, **bbox_data)
        self.session.add(image)
        self.session.commit()

    def has(self, filename: str) -> bool:
        query = self.session.query(ImageModel).filter_by(filename=filename)
        return self.session.query(query.exists()).scalar()

    @staticmethod
    def _2imagedata(image: ImageModel) -> ImageData:
        bbox = None
        if image.bbox_x1 is not None:
            bbox = BoundingBox(
                x1=image.bbox_x1,
                y1=image.bbox_y1,
                x2=image.bbox_x2,
                y2=image.bbox_y2,
                confidence=image.bbox_conf # type: ignore
            )
        return ImageData(filename=image.filename, embedding=list(image.embedding), bbox=bbox) #type: ignore

    def load(self, filename:str) -> ImageData:
        result = self.session.query(ImageModel).filter_by(filename=filename).first()
        if result is None:
            raise ValueError(f"Image with filename '{filename}' not found in database.")
        return self._2imagedata(result)

    def get_all_images(self) -> list[ImageData]:
        return [self._2imagedata(image) for image in
                self.session.query(ImageModel).all()]

    def get_by_distance(self, embedding: list[float], max_images: int = -1, cos_distance: bool = True) -> list[ImageData]:
        if cos_distance:
            distance_expr = ImageModel.embedding.cosine_distance(embedding).label("distance")
        else:
            distance_expr = ImageModel.embedding.l2_distance(embedding).label("distance")

        query = self.session.query(ImageModel, distance_expr).order_by(distance_expr)

        if max_images > 0:
            query = query.limit(max_images)

        return [self._2imagedata(image) for image, _ in query.all()]

    def set_metadata(self, key: str, value: str) -> None:
        metadata = self.session.query(MetadataModel).filter_by(key=key).first()
        if metadata:
            metadata.value = value
        else:
            metadata = MetadataModel(key=key, value=value)
            self.session.add(metadata)
        self.session.commit()

    def get_metadata(self, key: str) -> str | None:
        metadata = self.session.query(MetadataModel).filter_by(key=key).first()
        return metadata.value if metadata else None

    def clear(self) -> None:
        self.session.query(ImageModel).delete()
        self.session.query(MetadataModel).delete()
        self.session.commit()

