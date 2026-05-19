# Semantic search and embeddings

Semantic search is a technique that finds information by **meaning** rather than by literal word match. It is the technical foundation behind modern systems like RAG, recommendation engines, automatic text classification, and conversational assistants that query proprietary knowledge bases.

## The problem with traditional search

For decades, text search was solved with lexical techniques: exact match (`LIKE '%python%'`), inverted indexes (like those used by Elasticsearch with BM25), synonym expansion, stemming, and lemmatization. These techniques are fast and accurate when the user types exactly the words that appear in the document, but they fail in very common scenarios:

- **Synonymy**: searching for "car" doesn't find documents that talk about "vehicle" or "automobile".
- **Paraphrasing**: the question "how do I configure the database?" doesn't match a document titled "PostgreSQL setup".
- **Ambiguity**: "python" can refer to the programming language or the snake, and lexical search has no way to distinguish.
- **Multilingual context**: a user asks in Spanish about documentation written in English.
- **Long questions**: the user writes a full natural-language sentence instead of keywords.

Semantic search solves all these cases because it compares meanings instead of strings.

## What is an embedding

An **embedding** is a numerical representation of a piece of text (or image, audio, code) as a high-dimensional vector. Each dimension captures some latent aspect of meaning, learned during the model's training.

Intuitively, an embedding translates "understanding a sentence" into "placing a point in a high-dimensional space where nearby points represent similar meanings".

A simplified example with 3 imaginary dimensions:

```
"the cat eats fish"        → [0.8, 0.1, 0.9]
"the feline devours tuna"  → [0.79, 0.12, 0.88]
"Python is a language"     → [0.1, 0.95, 0.2]
```

The first two vectors are almost identical because they mean the same thing even though they don't share any words. The third one is far away because it's about a different topic. In practice, real embeddings have between 384 and 3072 dimensions, not 3, but the idea is the same.

Embeddings are not directly interpretable: there's no single dimension that means "animal" or "food". The model learns the structure of the space during training, and the dimensions are an abstract mixture of features that turned out useful for the target task.

## How they are generated

Modern embeddings are produced by neural models trained specifically for the task. The most-used ones in 2025-2026:

- **OpenAI `text-embedding-3-small`** (1536 dimensions, configurable): cheap, fast, very high quality. The pragmatic default for most projects.
- **OpenAI `text-embedding-3-large`** (3072 dimensions): better quality at higher cost and storage.
- **Cohere Embed v3**: strong in multilingual scenarios.
- **Sentence Transformers** (e.g. `all-MiniLM-L6-v2`, `bge-large-en`): open-source models that run locally with no per-token cost. Lower quality but no external dependency or network latency.
- **Voyage AI**: models specialized in specific domains (code, legal, finance).

Basic usage with the OpenAI Python API:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def embed(text: str) -> list[float]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding
```

Modern APIs accept batches (multiple texts in a single call), which drastically reduces cost when ingesting a large corpus.

## How similarity is measured

Once you have two embeddings, you need a function that measures how close they are. The standard metrics are:

### Cosine distance

Measures the angle between two vectors, ignoring their magnitude. It is the **most-used metric in NLP** because what matters is the direction of the vector (the "topic" of the text), not its size.

$$\text{cosine similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

It returns values between -1 and 1. Two identical vectors give 1, orthogonal vectors give 0, opposite ones give -1. "Cosine distance" is simply `1 - similarity`.

In pgvector it's used with the `<=>` operator:

```sql
SELECT text FROM chunks ORDER BY embedding <=> $1 LIMIT 5;
```

### Euclidean distance (L2)

The straight-line distance between two points in the space. Useful when the vector's magnitude does matter (for example in some computer vision models). In NLP it is rarely the best choice.

In pgvector: `<->` operator.

### Inner product

Faster to compute than cosine. Equivalent to cosine when the vectors are **normalized** to unit norm, which many embedding models already do as their final step.

In pgvector: `<#>` operator.

**Rule of thumb**: for RAG with OpenAI or Cohere embeddings, use cosine distance. It's the metric those models were optimized to produce.

## The full semantic search pipeline

A typical semantic search system follows these steps:

### Indexing phase (offline, done once per document)

1. **Ingestion**: read the source document (PDF, markdown, HTML, etc.) and extract plain text.
2. **Cleaning**: normalize the text — strip repeated headers/footers, extra whitespace, weird characters.
3. **Chunking**: split the text into fragments of the appropriate size. A whole document is too large to embed precisely; a single word is too little context. The typical sweet spot is **300 to 800 tokens per chunk**, with an overlap of 50 to 100 tokens between consecutive chunks so that information at the edges isn't lost.
4. **Embedding**: each chunk is passed through the embedding model and its vector is obtained.
5. **Storage**: chunks, vectors, and metadata (source document, position, page, etc.) are stored in a vector store like pgvector, Pinecone, Qdrant, or Chroma.

### Search phase (online, on every user query)

1. **Query embedding**: the user's question is passed through **the same embedding model** that was used for the chunks. This detail is critical: embeddings from different models live in incompatible vector spaces.
2. **Vector search**: the vector store is searched for the top-K chunks closest to the question vector according to the chosen metric.
3. **Optional re-ranking**: the K candidate chunks are passed through a more precise but slower cross-encoder model that reorders them by actual relevance. This improves quality when you have few candidates but want maximum precision.
4. **Context assembly**: the final chunks are concatenated and passed as context to an LLM in the RAG case, or returned directly to the user in the case of a search bar.

## Chunking: the forgotten art

Chunking quality is probably the factor that most impacts the precision of a RAG system, and yet it is systematically underestimated. Good chunking requires making conscious decisions about several parameters:

- **Size**: very short chunks lose context ("this function does X" without knowing what "this function" is). Very long chunks dilute meaning — the embedding ends up averaging different topics and semantic similarity degrades.
- **Overlap**: how much text is repeated between consecutive chunks. Without overlap, critical information may sit exactly at the boundary and get lost. With too much overlap, you duplicate storage and embedding cost.
- **Boundaries**: do you split by exact token counts or do you respect paragraphs, sentences, section titles? Respecting semantic structure (markdown headers, HTML `<p>`, etc.) yields more coherent chunks.
- **Enrichment**: sometimes it's worth prefixing each chunk with the document title or section header so that the embedding captures that context.

There is no universally optimal chunking. Each corpus and use case requires experimentation and measurement with an evaluation dataset.

## Evaluating semantic search

A semantic search system without metrics is a system that nobody can improve with confidence. The standard metrics:

- **Precision@K**: of the K results returned, how many are relevant.
- **Recall@K**: of the relevant results that exist in the corpus, how many are in the top K.
- **MRR (Mean Reciprocal Rank)**: how high in the ranking the first relevant result appears, averaged over a query set.
- **NDCG (Normalized Discounted Cumulative Gain)**: a more sophisticated metric that rewards rankings where the most relevant items are at the top, with a logarithmic discount by position.

Measuring these metrics requires **ground truth**: a query set with correct answers manually labeled. It's tedious but indispensable. A dataset of 50 to 100 query-answer pairs is enough to compare configurations and detect regressions.

## Hybrid search

Pure semantic search doesn't always win. There are queries where exact lexical match is what the user wants: product codes, proper names, technical identifiers. For those cases, BM25 (the classic lexical search with tf-idf) is still unbeatable.

**Hybrid search** combines both: it runs BM25 and vector search in parallel and merges the rankings with a technique like **Reciprocal Rank Fusion (RRF)**. The result is more robust than either approach alone.

PostgreSQL supports BM25 via native `ts_vector` and `tsquery`. Combining both in a single query — BM25 + pgvector — is one of the reasons Postgres is attractive as a unified backend for search.

## Costs to keep in mind

- **Initial corpus embedding**: with `text-embedding-3-small` at \$0.02 per million tokens, embedding a 10-million-token corpus (roughly 20 long books) costs \$0.20. Very cheap.
- **Query embedding in production**: each user query spends between 10 and 100 tokens. Negligible even with thousands of daily queries.
- **Storage**: a 1536-dimensional vector in float32 takes 6 KB. A million chunks is ~6 GB of embeddings alone. That's nothing for Postgres but it matters for Pinecone, which charges per stored vector.
- **Re-indexing**: if you change embedding models, you have to re-embed the whole corpus. This is a significant hidden cost when a better model appears.

## Practical considerations

- **Caching**: if the same chunk will be embedded many times (for example during development), cache by text hash so you don't pay twice.
- **Model versioning**: store in the database which model generated each embedding. Avoids silent corruption if models get mixed.
- **Dimension truncation**: modern models like `text-embedding-3-*` support a `dimensions` parameter that returns shorter vectors with minimal quality loss. Reducing from 1536 to 512 saves 66% of storage with almost the same precision.
- **Language**: monolingual models beat multilingual ones in their native language, but multilingual ones are more robust if the corpus mixes languages. For Spanish mixed with technical English, multilingual is the safe choice.
- **Source code**: for code search, general-purpose embeddings work poorly. Specialized models like Voyage Code or `text-embedding-3-large` with good chunking deliver better results.

## Relationship with LLMs and RAG

Embeddings are not LLMs. They are much smaller, faster, and cheaper models, trained for a different task: producing vector representations of text. LLMs (GPT, Claude, Llama) do text generation. In a RAG system they play complementary roles:

1. **Embeddings** find the relevant fragments of the corpus for a given question.
2. The **LLM** receives those fragments as context along with the question, and produces a natural-language answer citing the retrieved information.

This design is what allows an LLM to answer questions about documents it never saw during training, without hallucinating, and with verifiable citations. The final quality of the system depends as much, or more, on retrieval (embeddings + search) as on generation (the LLM). A powerful LLM with bad retrieval produces very convincing incorrect answers. A modest LLM with precise retrieval produces useful, verifiable answers.

That's why people say that in RAG, 80% of the engineering effort should go to retrieval, and only 20% to the LLM prompt.

## References

- OpenAI embeddings guide: https://platform.openai.com/docs/guides/embeddings
- Sentence Transformers: https://www.sbert.net/
- MTEB (Massive Text Embedding Benchmark) to compare models: https://huggingface.co/spaces/mteb/leaderboard
- Original RAG paper: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
