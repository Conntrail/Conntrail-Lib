# Adaptive RAG Agent

Adapted from: `langchain-ai/langgraph` examples/rag/langgraph_adaptive_rag.ipynb

## Description

An adaptive retrieval agent that selects its retrieval strategy based on query type:
- `direct_answer` — factual/definitional questions answerable from LLM knowledge
- `vector_search` — semantic search over a local document store
- `web_search` — current events, recent data, or external information

## Why it's a good Conntrail test subject

Retrieval strategy routing has clear semantic dimensions (recency, specificity, domain). The boundary between "answer from knowledge" and "search documents" is a useful ambiguity for Conntrail to analyse.

## Note on document store

For testing, the vector store uses a small in-memory fixture with 5 documents. No external database required.

## Source

```
git clone https://github.com/langchain-ai/langgraph
# See: examples/rag/langgraph_adaptive_rag.ipynb
```
