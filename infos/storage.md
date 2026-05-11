# Storage backends

Storage backends persist image embeddings and bounding boxes, and expose similarity search.
The `storage` key in a whitelist run specifies the backend.

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

**Default dev URL:** `postgresql://postgres:changethis@localhost:54321/postgres`

**Example:**
```json
{
  "type": "postgresql",
  "params": { "db_url": "postgresql://postgres:changethis@localhost:54321/mydb" }
}
```

---

## Notes

- Both backends share the same schema: an `images` table (filename, embedding blob, bounding box
  columns) and a `metadata` key-value table.
- The runner uses `metadata` to persist ingest timing so that re-used databases report the
  original ingestion cost, not the near-zero cache-hit time.
- Set `clear_storage: false` on runs that share the same database to skip re-ingestion.
