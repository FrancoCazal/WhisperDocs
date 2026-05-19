# RAG: fundamentals

RAG (Retrieval-Augmented Generation) is an architectural pattern that combines two distinct capabilities: searching over an external corpus of documents, and generating natural language with an LLM. The idea is simple: before answering a question, the system finds the most relevant fragments of a knowledge base, passes them to the LLM as context, and asks it to generate the answer based on that context. The LLM acts as the "reasoner" over information it never saw during training.

## The problem it solves

LLMs suffer from three fundamental limitations when used alone:

1. **Static knowledge**: they know what they learned during training, up to a cutoff date. They cannot answer about private documents, recent trends, or organization-specific information.
2. **Hallucination**: when they don't know something, they tend to invent plausible but incorrect responses. An LLM asked about a non-existent product can give you detailed specs that are completely false.
3. **Lack of citability**: their responses are not verifiable. They can't say "I got this from this document, line 42".

RAG attacks all three problems simultaneously:

- The knowledge is provided by an external corpus that can be updated without retraining the model.
- Hallucination is drastically reduced because the prompt explicitly says "answer only based on this context".
- Citations are traceable because you know exactly which fragments were passed as context.

## Basic architecture

A RAG system has two phases: indexing (offline) and query (online).

### Indexing phase

Runs once per document, when it's uploaded to the system:

```
Document → Text extraction → Chunking → Embedding → Storage in vector store
```

Each step has important nuances:

- **Extraction**: converting native formats (PDF, HTML, markdown) to plain text. PDFs are the trickiest: scanned PDFs require OCR, complex layouts require tools like Unstructured or Docling.
- **Chunking**: dividing the text into manageable fragments. A whole document is too large to embed meaningfully; a lone token has no context. See the dedicated chunking strategies document.
- **Embedding**: each chunk is passed through a model (e.g. OpenAI `text-embedding-3-small`) that converts it into a high-dimensional numeric vector.
- **Storage**: chunks + vectors + metadata (source document, position, page) are stored in a vector store like pgvector, Pinecone, Qdrant, or Weaviate.

### Query phase

Runs every time the user asks something:

```
Question → Query embedding → Vector search → Top-K chunks → Constructed prompt → LLM → Answer
```

The LLM receives a prompt that typically has this structure:

```
You are an assistant that answers questions based ONLY on the provided context.
If the context does not contain enough information, say "I don't have information to answer".
Cite your sources using [n] where n is the fragment number.

Context:
[1] (from document "pgvector.md", chunk 3) ...
[2] (from document "semantic-search.md", chunk 1) ...
[3] ...

Question: What distance metrics does pgvector support?

Answer:
```

The key part of the prompt is constraining the LLM to what's in the context. A weak prompt ("use the context and your knowledge") allows free hallucination. A strict prompt ("only what's in the context, say you don't know if info is missing") minimizes it.

## Grounding

Grounding is the term that describes how tied the response is to the retrieved material. A well-grounded response only asserts things present in the context. A poorly-grounded response invents, extrapolates, or mixes with the LLM's pre-training.

Techniques to improve grounding:

- **Strict system prompt**: explicitly forbid making things up.
- **Low temperature**: 0.0 to 0.3. High temperatures induce creativity, which is counter-productive in RAG.
- **Few-shot examples**: show examples of questions where the context is insufficient and the correct answer is "I don't know".
- **Citation requirement**: requiring `[n]` on every claim forces the LLM to anchor explicitly.
- **Post-hoc verification**: a second LLM (or regex) checks that every claim in the response has a citation.

## Retrieval: the real bottleneck

The most determining factor of RAG quality is not the LLM — it's retrieval. A powerful LLM with poor retrieval produces very convincing incorrect answers. A modest LLM with precise retrieval produces useful, verifiable answers.

Put another way: if the retrieved chunks don't contain the answer, the LLM can't generate it — unless it invents it, which is exactly what we want to avoid.

That's why in a mature RAG system, 80% of the engineering goes into retrieval:

- Chunking quality
- Embedding model quality
- Appropriate similarity metric
- Re-ranking to refine results
- Hybrid search (vector + BM25)
- Metadata filtering

The remaining 20% is the LLM prompting.

## Advanced variants

### RAG fusion / multi-query

Instead of a single search, the system generates multiple reformulations of the question and searches with all of them. The results are merged with techniques like Reciprocal Rank Fusion (RRF). Improves recall when the user's question is poorly formulated.

### HyDE (Hypothetical Document Embeddings)

The LLM generates a "hypothetical" answer to the question, without context. That hypothetical answer is embedded and used as the query vector. The intuition: a hypothetical answer has more similarity with a chunk that contains it than the original question does. Works surprisingly well for abstract questions.

### Re-ranking

The first K chunks from the vector search are passed through a cross-encoder (more precise but slower model) that reorders them by actual relevance to the question. Improves precision at the cost of latency. Cohere Rerank and BGE rerankers are popular options.

### Hybrid search

Combine semantic search (vector) with lexical search (BM25). Vector captures synonymy and paraphrasing; lexical captures critical exact matches (proper names, codes, acronyms). Merged with RRF.

### Agentic RAG

The LLM decides when to search, what to search for, whether the answer is sufficient or whether another round is needed. Implemented with LangGraph or other frameworks. More powerful for multi-step questions, but more complex and slower.

## Evaluating RAG

Without metrics, no improvement. The main metrics:

- **Answer accuracy**: is the answer correct against ground truth? Typically evaluated with LLM-as-judge comparing the answer vs ground truth.
- **Hallucination rate**: does the answer contain claims absent from the context? Measurable by comparing claims in the answer against the retrieved chunks.
- **Retrieval precision@K**: of the K retrieved chunks, how many are relevant?
- **Retrieval recall@K**: of the existing relevant chunks, how many are in the top K?
- **Latency**: total query time.
- **Cost per query**: tokens in + tokens out × prices.

Tools like RAGAS automate several of these metrics using LLM-as-judge.

## Common errors

- **Naive chunking**: splitting at fixed size without respecting structure (titles, paragraphs). Causes chunks that mix different topics and degrade embeddings.
- **Weak prompt**: not constraining the LLM enough. Produces answers that mix context with pre-training, impossible to verify.
- **Ignoring re-ranking**: pure retrieval with K=10 often brings irrelevant chunks in high positions. Re-ranking is cheap compared to the improvement.
- **Different embedding model for query and documents**: obvious but critical. Changing the model requires re-indexing the whole corpus.
- **Logging nothing**: without logs of queries, retrieved chunks, responses, and latency, it's impossible to debug when something fails.
- **Evaluating by intuition instead of a dataset**: "it works well" is opinion. A dataset of 50 questions with ground truth turns analysis into data.

## When NOT to use RAG

- **Answers require reasoning without a specific factual basis**: writing a poem, solving a math problem, translating. The LLM alone is enough.
- **Knowledge fits in the prompt**: if your "corpus" is 2 paragraphs, send them directly in the prompt without retrieval engineering.
- **Very low latency tolerance**: RAG adds one or more vector queries before the LLM. If you need sub-100ms responses, RAG adds unavoidable overhead.
- **Corpus changes very frequently**: re-indexing constantly can be expensive. For very dynamic content, fine-tuning or creative caching are alternatives.
- **Answers require aggregations/calculations over structured data**: RAG over documents does not answer "how many customers did we have last year?". That's a job for Text-to-SQL or tools that invoke SQL queries.

## Trade-offs vs alternatives

| Approach | Pros | Cons |
|----------|------|------|
| RAG | Updatable, citable knowledge, no retraining | Complexity, extra latency, depends on retrieval quality |
| Fine-tuning | Low latency, knowledge baked into model | Requires labeled data, retrain on every change, not citable |
| LLM with long prompt | Simple | Limited context window, expensive with huge prompts |
| Traditional search without LLM | Fast, cheap | No synthesis, responses are lists of documents not answers |

RAG is the right choice when the corpus is large, changes frequently, and you want synthesized answers with traceability.

## References

- Original paper: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- RAGAS (evaluation): https://github.com/explodinggradients/ragas
- LangChain RAG guides: https://python.langchain.com/docs/tutorials/rag/
- HyDE paper: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
