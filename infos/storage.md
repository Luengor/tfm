# Storage backends

Storage backends persist image embeddings and bounding boxes, and expose similarity search.
The `storage` key in a configuration run specifies the backend.

---

## SQLite

**Type string:** `sqlite`

A lightweight, file-based backend. No external service required. Similarity search is performed
in Python as a linear scan over all rows, so it is slower than the PostgreSQL backend for large
datasets.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_path` | `str` | `{output_dir}/db/{run_id}.sqlite` | Path to the SQLite database file. Supports `{run_id}` and `{output_dir}` template placeholders. |

**Example:**
```json
{
  "type": "sqlite",
  "params": { "db_path": "benchmark_{run_id}.db" }
}
```

---

## PostgreSQL

**Type strings:** `postgres` or `postgresql`

Uses the `pgvector` extension for native vector similarity search via index-accelerated cosine
and L2 distance operators. The database is created automatically if it does not exist.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db_url` | `str` | *(required)* | SQLAlchemy connection URL, e.g. `postgresql://user:pass@host:port/dbname`. |
| `hnsw` | `bool` \| `dict` \| `null` | `null` | Enables an HNSW ANN index on the embedding column. `true` uses defaults; a dict overrides them — see below. |

**HNSW sub-parameters** (under `params.hnsw`):

| Key | Type | Default | Description |
|---|---|---|---|
| `ops` | `"cosine"` \| `"l2"` \| `"ip"` \| `"both"` | `"both"` | Operator class. `"both"` builds one index per cosine and L2 so either query path is accelerated. |
| `m` | `int` | `16` | Maximum graph degree per layer (build-time). Higher = better recall, larger index, slower build. |
| `ef_construction` | `int` | `64` | Build-time candidate list size. Higher = better recall, slower build. |
| `ef_search` | `int` \| `null` | `null` | Query-time candidate list size. When set, issued as `SET hnsw.ef_search = N` on the storage session. |
| `recreate` | `bool` | `false` | If `true`, drops any existing HNSW index on the column before building. Required when sweeping HNSW parameters on a shared database. |

The index is created lazily by `ensure_hnsw_index()` (called by the
benchmark runner between ingest and the similarity-search stage so the
build cost is excluded from the timed query loop). pgvector's HNSW
dimension limit is 2000 for `vector`; embeddings with `embedding_size > 2000`
(e.g. ResNet50=2048, VGG16=4096) will raise on `CREATE INDEX`.

**Default dev URL:** `postgresql://postgres:changethis@localhost:54321/postgres`

**Example (exact):**
```json
{
  "type": "postgresql",
  "params": { "db_url": "postgresql://postgres:changethis@localhost:54321/mydb" }
}
```

**Example (HNSW):**
```json
{
  "type": "postgresql",
  "params": {
    "db_url": "postgresql://postgres:changethis@localhost:54321/mydb",
    "hnsw": {"ops": "cosine", "m": 16, "ef_construction": 64, "ef_search": 40}
  }
}
```

---

## Notes

- Both backends share the same schema: an `images` table (filename, embedding blob, bounding box
  columns) and a `metadata` key-value table.
- The runner uses `metadata` to persist ingest timing so that re-used databases report the
  original ingestion cost, not the near-zero cache-hit time.
- Set `clear_storage: false` on runs that share the same database to skip re-ingestion.
