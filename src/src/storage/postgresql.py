from urllib.parse import urlparse, urlunparse

from src.abstractions import StorageBase, ImageData, BoundingBox
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session

class Base(DeclarativeBase):
    pass

class ImageModel(Base):
    __tablename__ = 'images'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column()
    embedding: Mapped[Vector] = mapped_column(Vector)
    bbox_x1: Mapped[float] = mapped_column()
    bbox_y1: Mapped[float] = mapped_column()
    bbox_x2: Mapped[float] = mapped_column()
    bbox_y2: Mapped[float] = mapped_column()
    bbox_conf: Mapped[float] = mapped_column()

class MetadataModel(Base):
    __tablename__ = 'metadata'

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column()

class PostgreSQLStorage(StorageBase):
    _COMMIT_BATCH = 50
    _OPS_MAP = {
        "cosine": "vector_cosine_ops",
        "l2": "vector_l2_ops",
        "ip": "vector_ip_ops",
    }

    def __init__(self, db_url: str, hnsw: bool | dict | None = None):
        self._ensure_database(db_url)

        self.engine = create_engine(db_url)
        self._pending = 0

        # Init db
        self.connection = self.engine.connect()
        self.session = Session(self.connection)

        self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        self.session.commit()

        Base.metadata.create_all(self.engine)

        self._hnsw_cfg = self._normalize_hnsw(hnsw)
        self._hnsw_index_created = False

    @staticmethod
    def _normalize_hnsw(hnsw: bool | dict | None) -> dict | None:
        if not hnsw:
            return None
        cfg = dict(hnsw) if isinstance(hnsw, dict) else {}
        ops = str(cfg.get("ops", "both")).lower()
        if ops == "both":
            ops_list = ["cosine", "l2"]
        elif ops in PostgreSQLStorage._OPS_MAP:
            ops_list = [ops]
        else:
            raise ValueError(f"Invalid hnsw.ops '{ops}'. Use 'cosine', 'l2', 'ip', or 'both'.")
        return {
            "ops": ops_list,
            "m": int(cfg.get("m", 16)),
            "ef_construction": int(cfg.get("ef_construction", 64)),
            "ef_search": cfg.get("ef_search"),
        }

    def ensure_hnsw_index(self) -> None:
        if self._hnsw_index_created or self._hnsw_cfg is None:
            return

        # Flush any pending inserts before DDL.
        if self._pending:
            self.session.commit()
            self._pending = 0

        row = self.session.query(ImageModel).first()
        if row is None:
            return
        dim = len(row.embedding) # type: ignore

        # pgvector HNSW requires a fixed-dim column.
        self.session.execute(text(f"ALTER TABLE images ALTER COLUMN embedding TYPE vector({dim})"))
        self.session.commit()

        cfg = self._hnsw_cfg
        m = cfg["m"]
        ef_construction = cfg["ef_construction"]
        for ops in cfg["ops"]:
            op_class = self._OPS_MAP[ops]
            index_name = f"images_embedding_hnsw_{ops}"
            self.session.execute(text(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON images "
                f"USING hnsw (embedding {op_class}) "
                f"WITH (m = {m}, ef_construction = {ef_construction})"
            ))
        self.session.commit()

        if cfg["ef_search"] is not None:
            self.session.execute(text(f"SET hnsw.ef_search = {int(cfg['ef_search'])}"))

        self._hnsw_index_created = True

    @staticmethod
    def _ensure_database(db_url: str) -> None:
        parsed = urlparse(db_url)
        db_name = parsed.path.lstrip("/")
        if not db_name or db_name == "postgres":
            return

        bootstrap_url = urlunparse(parsed._replace(path="/postgres"))
        engine = create_engine(bootstrap_url, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as conn:
                exists = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :name"),
                    {"name": db_name},
                ).fetchone()
                if not exists:
                    conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        finally:
            engine.dispose()

    def close(self) -> None:
        self.session.commit()
        self.session.close()
        self.connection.close()
        self.engine.dispose()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def save(self, data: ImageData) -> None:
        image = ImageModel(
            filename=data.filename,
            embedding=data.embedding,
            bbox_x1=data.bbox.x1,
            bbox_y1=data.bbox.y1,
            bbox_x2=data.bbox.x2,
            bbox_y2=data.bbox.y2,
            bbox_conf=data.bbox.confidence
        )
        self.session.add(image)
        self._pending += 1
        if self._pending >= self._COMMIT_BATCH:
            self.session.commit()
            self._pending = 0

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

    def load(self, filename:str) -> list[ImageData]:
        result = self.session.query(ImageModel).filter_by(filename=filename).all()
        if not result:
            raise ValueError(f"Image with filename '{filename}' not found in database.")
        return [self._2imagedata(image) for image in result]

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

