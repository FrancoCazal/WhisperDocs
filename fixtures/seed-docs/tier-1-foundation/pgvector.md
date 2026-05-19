# pgvector

pgvector is an open-source extension for PostgreSQL that adds native support for high-dimensional vectors and similarity search. It turns PostgreSQL into a vector database without the need to run a separate dedicated service.

## What problem it solves

In modern AI applications — especially RAG (Retrieval-Augmented Generation) systems — we need to search information by **meaning**, not by exact text match. A traditional `LIKE '%python%'` search would not find a document about "pythons" (the snake) nor one about "dynamic programming languages" even though both are semantically relevant depending on context.

The standard solution:

1. Pass each fragment of text through an embedding model (for example OpenAI's `text-embedding-3-small`), which returns a high-dimensional numeric vector (1536 floats) representing the meaning of the text.
2. Do the same with the user's question.
3. Search for the fragments whose vector is mathematically closest to the question vector.

pgvector lets you do steps 2 and 3 directly in SQL, taking advantage of PostgreSQL indexes, filters, joins, and transactions.

## Main features

- Native `vector(n)` data type to store n-dimensional vectors.
- Three supported distance metrics:
  - **Cosine** (`<=>`): the most used in NLP. Measures the angle between vectors, ignoring magnitude.
  - **L2 / Euclidean** (`<->`): straight-line distance. Useful in computer vision.
  - **Inner product** (`<#>`): faster if vectors are normalized.
- Approximate indexes for fast searches at scale:
  - **IVFFlat**: partitions the space into clusters. Requires training the index with existing data.
  - **HNSW** (since pgvector 0.5): hierarchical graph, better recall and speed, no retraining needed.
- Full integration with the PostgreSQL ecosystem: `WHERE`, `JOIN`, `ORDER BY`, ACID transactions, standard backups, replicas.

## Installation and setup

There's an official Docker image ready to use:

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
```

Once the database is running, the extension must be explicitly enabled in the database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

In projects with migrations (Alembic, Django migrations, etc.), this statement should live in the first migration of the project so that any new environment ends up in the correct state automatically.

## Basic SQL usage

Create a table with a vector column:

```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1536)
);
```

Insert a vector:

```sql
INSERT INTO chunks (text, embedding)
VALUES ('Python is an interpreted language', '[0.013, -0.24, ..., 0.08]');
```

Find the 5 most similar chunks to a question using cosine distance:

```sql
SELECT text, embedding <=> '[0.02, -0.19, ..., 0.11]' AS distance
FROM chunks
ORDER BY embedding <=> '[0.02, -0.19, ..., 0.11]'
LIMIT 5;
```

Create an IVFFlat index to speed up searches at scale:

```sql
CREATE INDEX ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

The `lists` parameter controls the trade-off between speed and accuracy: more lists is faster but with lower recall. A common rule of thumb is `lists = rows / 1000` for large tables.

## Using pgvector from Python with SQLAlchemy

The `pgvector` package provides a custom type that integrates with SQLAlchemy:

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
```

Similarity search with async SQLAlchemy:

```python
from sqlalchemy import select

async def find_similar(session, query_embedding: list[float], top_k: int = 5):
    result = await session.execute(
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    return result.scalars().all()
```

## pgvector vs other vector databases

| Aspect | pgvector | Pinecone | Qdrant | Chroma |
|--------|----------|----------|--------|--------|
| Type | Postgres extension | Managed SaaS | Self-hosted or cloud | Embedded / server |
| Extra infrastructure | None | External service | Extra service | Extra service |
| Cost | Free | Pay-per-use | Free self-hosted | Free |
| Typical scale | Up to ~10M vectors | 100M+ | 100M+ | ~1M |
| Relational queries | Yes, full JOINs and WHERE | No | Limited | No |
| ACID transactions | Yes | No | No | No |
| Learning curve | Low if you already know SQL | Low | Medium | Low |

The most important advantage of pgvector for small-to-medium RAG applications is being able to combine vector search with relational filters in a single query:

```sql
SELECT c.text, c.embedding <=> $1 AS distance
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE d.collection_id = $2
  AND d.created_at > NOW() - INTERVAL '30 days'
ORDER BY distance
LIMIT 5;
```

Doing the same thing in Pinecone requires filtering on metadata, with less flexibility and unpredictable performance on poorly selective filters.

## When not to use pgvector

- When the dataset exceeds 10 to 20 million vectors and latency is critical. Dedicated databases like Qdrant or Weaviate are more optimized for that regime.
- When you need specific advanced features that pgvector does not implement (hybrid sparse + dense search out-of-the-box, aggressive quantization, automatic sharding).
- When the vector write load is extremely high and competes for resources with the OLTP load of the same Postgres.

For most projects that are starting out, pgvector is the pragmatic choice: one system to operate, one backup to maintain, one source of truth.

## Performance considerations

- **Dimensionality**: search cost grows linearly with dimensions. 1536 (OpenAI) is manageable; 3072 starts to be heavy. Models like `text-embedding-3-small` support dimension truncation (for example to 512) with minimal quality loss.
- **Indexes**: without an index, pgvector does a sequential scan. It works for thousands of rows but not millions. IVFFlat or HNSW are mandatory at scale.
- **Query parameters**: for IVFFlat, `SET ivfflat.probes = 10;` controls how many clusters to visit during a search. More probes = better recall, more latency.
- **Warm cache**: the first query against a large index can be slow because it loads the index from disk. Subsequent ones use Postgres's buffer cache.

## Ecosystem and adoption

pgvector has official and third-party integrations with the most-used frameworks:

- **LangChain** and **LlamaIndex** support it as a vector store directly.
- **Supabase** includes it as a flagship feature of its managed Postgres offering.
- **Neon**, **Railway**, **Render**, and most Postgres PaaS providers have it available.
- AWS RDS for PostgreSQL supports it since version 15.2.

Its growing adoption in 2024-2025 made it the practical default for RAG projects that don't have specific reasons to pick something else.

## References

- Official repository: https://github.com/pgvector/pgvector
- Python client: https://github.com/pgvector/pgvector-python
- Official Docker image: https://hub.docker.com/r/pgvector/pgvector
