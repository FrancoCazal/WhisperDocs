# Chunking strategies

Chunking is the act of splitting a document into smaller fragments that are embedded and indexed individually. It's one of the steps that most impacts the quality of a RAG system, and also one of the most underestimated. Good chunking can improve accuracy 20-40% over a naive approach, without changing the embedding model or the LLM.

## Why it matters so much

The embedding of a chunk is a weighted average of the meanings of its tokens. If a chunk mixes two topics (e.g. a paragraph about Docker and another about PostgreSQL in the same chunk), the embedding ends up "between" both, and retrieval for either topic recovers it with low precision.

Optimal chunks:

- **Are coherent**: each chunk covers a single topic or sub-topic.
- **Are the right size**: not so small that they lose context, not so large that they dilute meaning.
- **Carry useful metadata**: document title, section, position.
- **Allow context reconstruction**: overlap or enrichment so the LLM doesn't receive orphan sentences.

## Fixed-size chunking

The simplest strategy: split by a fixed token count, with configurable overlap.

```python
from dataclasses import dataclass
import tiktoken

@dataclass
class Chunk:
    text: str
    index: int
    tokens: int

def fixed_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    chunks = []
    stride = chunk_size - overlap
    start = 0
    index = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = encoder.decode(tokens[start:end])
        chunks.append(Chunk(text=chunk_text, index=index, tokens=end - start))
        if end == len(tokens):
            break
        start += stride
        index += 1
    return chunks
```

Advantages:

- Simple and predictable.
- Fast.
- Unit-testable without external dependencies.

Disadvantages:

- Splits in the middle of sentences, paragraphs, code blocks.
- Ignores the document's semantic structure.

For general cases with well-written docs, fixed-size with 300-800 tokens per chunk and 10-20% overlap works reasonably well. It's a valid baseline.

## Parameters: chunk_size and overlap

### Chunk size

- **Too small (< 200 tokens)**: chunks lose context. Isolated sentences are hard to embed well.
- **Too large (> 1500 tokens)**: the embedding averages too much, reducing retrieval precision.
- **Typical sweet spot**: 300-800 tokens, depending on the domain.

Documents with short paragraphs and dense flow of ideas (technical documentation): smaller chunks (300-500). Narrative documents with ideas that develop slowly (essays, books): larger chunks (600-1000).

### Overlap

Overlap between consecutive chunks prevents information loss at the boundaries. If an idea is partly at the end of one chunk and partly at the beginning of the next, without overlap the idea is cut and its embedding is poor in both chunks.

- **10-20% of chunk size** is a reasonable range.
- **Overlap 0**: minimal storage but risk of loss.
- **Overlap > 30%**: diminishing returns and duplicated embedding cost.

In a 500-token chunk, 50-75 of overlap (10-15%) is standard.

## Recursive character chunking

An evolution of fixed-size that respects natural boundaries. It defines a hierarchy of separators (`\n\n`, `\n`, `. `, ` `) and splits using the highest one possible that keeps chunks within the limit.

```python
separators = ["\n\n", "\n", ". ", " "]

def recursive_chunk(text, max_tokens, separators):
    if count_tokens(text) <= max_tokens:
        return [text]

    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if count_tokens(candidate) <= max_tokens:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part
            if current:
                chunks.append(current)
            return chunks
    return [text]  # fallback
```

LangChain implements this with `RecursiveCharacterTextSplitter` and it's the recommended default for free text. It solves the main problem of fixed-size: it doesn't split in the middle of sentences if there's a paragraph available as a boundary.

## Structure-aware chunking

For formats with explicit structure (markdown, HTML, code), parsing the structure and chunking while respecting its levels is significantly better.

### Markdown

Headers (`#`, `##`, `###`) define a hierarchy. A chunk can be a complete section, or if it's too large, it can be subdivided within that section.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
)
chunks = splitter.split_text(markdown_text)
```

Each chunk carries metadata about the header that contains it, which lets you enrich it:

```
[Header 1: RAG fundamentals] [Header 2: Architecture]
...chunk text...
```

This enrichment helps the embedding because it includes semantic context that would otherwise be lost.

### HTML

Structural tags (`<h1>`, `<article>`, `<section>`) serve as separators. You can also strip noisy tags (`<script>`, `<style>`, navigation) before chunking.

### Code

Naive chunking over code splits in the middle of functions, classes, conditionals. Using a parser (AST) and chunking by syntactic units (entire function, entire class) preserves coherence:

```python
from tree_sitter import Language, Parser

parser = Parser()
parser.set_language(Language.build_library(...))
tree = parser.parse(code.encode())
# iterate functions and classes, extract each as a chunk
```

Libraries like Tree-sitter support most languages.

## Semantic chunking

Instead of size, split when the topic changes. Each sentence (or group of sentences) is embedded, and "breakpoints" are detected where similarity drops abruptly.

Simplified algorithm:

1. Split into sentences.
2. Embed each sentence.
3. Compute similarity between consecutive sentences.
4. Identify low-similarity peaks (topic changes).
5. Split at those breakpoints.

Advantage: chunks are thematically coherent by design. Disadvantage: requires an extra embedding pass (cost and latency).

Useful for very heterogeneous documents where explicit structure doesn't help (transcripts, reports without clear headers).

## Hierarchical chunking

Chunks of multiple sizes at the same time. For example:

- Small chunks (~200 tokens) for high-precision retrieval.
- Large chunks (~2000 tokens) that contain them, to give the LLM more context once the relevant small chunk is identified.

At retrieval time, you search by small-chunk vectors and pass the corresponding large chunk to the LLM. The technique is known as "parent-child chunking" or "small-to-big retrieval".

Advantage: combines retrieval precision with broad context for generation. Disadvantage: operational complexity, double storage.

## Late chunking

A recent Jina technique. Instead of embedding small chunks, you embed the entire document with a model that produces per-token embeddings, and then aggregate (mean-pool) into "chunks" over the already-computed embedding sequence. That way each chunk sees the global context of the document through the token embeddings.

Advantage: chunks have implicit global context. Disadvantage: requires models with long-context support and the implementation is more complex.

For most cases, traditional chunking + re-ranking gives equivalent results with less complexity.

## Chunk enrichment

Adding metadata to the chunk text before embedding improves retrieval:

```
Document: "RAG fundamentals"
Section: "Basic architecture"
Subsection: "Indexing phase"

[chunk text]
```

The chunk's embedding now captures not only its content but its hierarchical context. Questions like "what happens in the indexing phase?" retrieve this chunk better because the word "indexing" appears in the embedding.

Risk: enrichment can dominate the embedding if the chunk content is short. Balance: short enrichment (1-2 lines), content always longer than the enrichment.

## Chunking for source code

AST-aware parsing gives the best results:

- Each independent function is a chunk.
- Each class (with its methods) is another chunk, or is subdivided if very large.
- Imports and module-level constants go into a shared "header" chunk.

Add useful metadata:

```
File: app/services/retriever.py
Class: VectorRetriever
Method: retrieve
```

Without this structure, searching "how do I retrieve chunks" pulls in irrelevant functions from other modules with similar keywords.

## Chunking for conversations

Transcripts and conversations are mostly heterogeneous. Useful techniques:

- **Per turn**: each message is a chunk.
- **Per topic** using semantic chunking.
- **Sliding window** with heavy overlap.

Enrichment with author/role is critical: a message without context of who said it loses meaning.

## Measurement

As with everything in RAG, chunking without measurement is guesswork. A ground-truth dataset plus retrieval metrics (precision@K, recall@K) let you compare strategies.

A typical experiment:

1. Index the corpus with chunking A (fixed-size 500).
2. Index the same corpus with chunking B (markdown-aware).
3. Run the same query dataset against both.
4. Compare precision@5, recall@5, answer accuracy.
5. Iterate.

Typical results: structure-aware beats fixed-size on well-structured docs; the difference disappears on flat docs.

## Common errors

- **Default chunk size adopted without experimentation**: "500 tokens" is not universal law; it depends on the domain.
- **Overlap 0 "to save storage"**: the storage savings are negligible, the quality loss can be large.
- **Naive chunking over HTML or markdown**: ignores structure, degrades embeddings.
- **Not storing metadata**: chunk_index, document_id, section. You lose the ability to produce good citations.
- **Re-chunking without re-embedding**: if you change the strategy, the old embeddings are useless. Re-indexing is mandatory.

## When each strategy is best

| Doc type | Recommended strategy |
|----------|----------------------|
| Technical documentation with markdown headers | Structure-aware (markdown) |
| Narrative books / PDFs | Recursive or fixed-size 800-1200 |
| Source code | AST-aware |
| HTML with lots of noise | Cleanup + structure-aware |
| Conversations / transcripts | Per-turn + semantic |
| Plain unstructured text | Recursive character |

## References

- LangChain text splitters: https://python.langchain.com/docs/how_to/#text-splitters
- Semantic chunking tutorial: https://python.langchain.com/docs/how_to/semantic-chunker/
- Late chunking paper: Günther et al. (2024)
- Anthropic "Contextual Retrieval": https://www.anthropic.com/news/contextual-retrieval
