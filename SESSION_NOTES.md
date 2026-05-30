# RAG Project — Session Notes

## Project Overview

A **Company Policies RAG** built with LangChain + Ollama + ChromaDB + Streamlit.  
Covers four policy documents: incident escalation, HR leave, IT support, expense claims.  
No conversation memory — every query is fully independent.

---

## Stack

| Layer | Choice | Notes |
|---|---|---|
| LLM | `llama3.2` via Ollama | Local, no API key |
| Embeddings | `nomic-embed-text` via Ollama | Local, no API key |
| Vector Store | ChromaDB | Persistent to `./chroma_db/` |
| Reranker | Cohere `rerank-english-v3.0` | Optional, requires API key |
| UI | Streamlit | `layout="wide"` |
| Evaluation | Ragas 0.4.3 | Judge model = llama3.2 |

---

## File Structure

```
dsai-m6-rag-team5/
├── doc/
│   ├── incident_escalation.md
│   ├── hr_leave_policy.md
│   ├── it_support_policy.md
│   └── expense_claims.md
├── rag.py              # Core pipeline logic
├── app.py              # Streamlit UI
├── golden_set.json     # 10-question evaluation set
└── requirements.txt
```

---

## requirements.txt

```
langchain
langchain-community
langchain-ollama
langchain-chroma
langchain-text-splitters
chromadb
streamlit
cohere
langchain-cohere
ragas
```

---

## 8 Classic RAG Components

| # | Component | Class | Detail |
|---|---|---|---|
| 1 | Document Loader | `DirectoryLoader` + `TextLoader` | Loads all `*.md` from `doc/` |
| 2 | Text Splitter | `RecursiveCharacterTextSplitter` | `chunk_size=1000`, `chunk_overlap=100` |
| 3 | Embedding Model | `OllamaEmbeddings` | `model="nomic-embed-text"` |
| 4 | Vector Store | `Chroma` | Persists to `./chroma_db/`, collection `"company_policies"` |
| 5 | Retriever | `vectorstore.as_retriever` | `k=3` |
| 6 | LLM | `ChatOllama` | `model="llama3.2"` |
| 7 | Prompt Template | `ChatPromptTemplate` | Grounded: "Answer using ONLY the context below" |
| 8 | Chain | LCEL pipe | `retriever \| format_docs → prompt → llm → StrOutputParser` |

### Why each component exists

- **Loader**: reads source files and wraps text in a `Document` object (with `page_content` + `metadata`). Decouples source format from the rest of the pipeline.
- **Splitter**: breaks the document into chunks so (a) embeddings are focused on one topic and (b) only the relevant portion is injected into the prompt, not the whole corpus.
- **Embeddings**: converts text to a dense numerical vector. Semantically similar text → close vectors. Runs at index time (chunks) and query time (question).
- **Vector Store**: stores `(chunk_text, embedding_vector, metadata)` triples. Enables fast nearest-neighbour search.
- **Retriever**: embeds the query, searches ChromaDB, returns the top-k most similar `Document` objects.
- **Prompt Template**: injects `{context}` (retrieved chunks) + `{question}` into a structured message. The "ONLY the context" instruction grounds the LLM and prevents hallucination.
- **LLM**: reads the assembled prompt and synthesises a fluent answer. Does not do retrieval — works only with what the retriever handed it.
- **Chain**: LCEL `|` pipe wires all components left-to-right. `RunnablePassthrough()` passes the raw question string through unchanged. Stateless — no `ConversationBufferMemory`, no history.

---

## Advanced Pipeline (`run_pipeline`)

```python
run_pipeline(
    question,
    cohere_api_key=None,
    cohere_model="rerank-english-v3.0",
    enable_rerank=False,
    enable_rewrite=True,
    enable_self_eval=False,
    max_rewrites=2,
    score_threshold=0.3,   # L2 relevance scale; 0.3 avoids over-triggering
)
```

### Flow

```
User question
    │
    ▼
retrieve_with_scores()          # ChromaDB similarity_search_with_relevance_scores
    │
    ├─ top_score >= threshold? → proceed
    └─ top_score <  threshold AND enable_rewrite AND attempts left?
           │
           ▼
        rewrite_query()         # LLM reformulates query
           │
           └─ retry (up to max_rewrites times)
    │
    ▼
[Cohere rerank if enabled]      # rerank_documents() via CohereRerank
    │
    ▼
generate_chain.invoke()         # _prompt | _llm | StrOutputParser
    │
    ▼
[Self-eval if enabled]          # 4 LLM calls, one per metric
    │
    ▼
append to trace_log             # module-level list, grows per session
    │
    ▼
return {"answer", "sources", "trace"}
```

### Score threshold calibration

ChromaDB defaults to **L2 distance**. `langchain-chroma` converts to relevance via `1 − d/√2`.  
For typical semantic matches, this gives scores in the **0.45–0.55 range**.  
`score_threshold=0.5` therefore triggers rewrites on almost every valid query.  
`score_threshold=0.3` only rewrites on genuine retrieval failures.

---

## Trace Structure

Each `run_pipeline` call appends one entry to `trace_log`:

```python
{
    "timestamp":     "2026-05-30T12:00:00",
    "original_query": "...",
    "final_query":    "...",    # may differ if rewriting fired
    "rewrite_count":  0,
    "answer":         "...",
    "events": [
        {"type": "retrieval", "attempt": 1, "query": "...",
         "scores": [0.52, 0.48, 0.41], "top_score": 0.52},
        # if rewrite:
        {"type": "rewrite", "original": "...", "rewritten": "..."},
        # if Cohere rerank:
        {"type": "rerank", "model": "rerank-english-v3.0", "scores": [0.91, 0.74, 0.62]},
        {"type": "answer", "preview": "..."},
        # if self-eval:
        {"type": "self_eval",
         "faithfulness": 0.8, "answer_relevancy": 0.9,
         "context_precision": 0.7, "context_recall": 0.6,
         "flagged": False},
    ]
}
```

---

## Streamlit UI — Four Tabs

| Tab | Content |
|---|---|
| 💬 Answer | LLM-generated answer in Markdown |
| 📄 Sources | Expandable chunks, labelled with source filename + char count |
| 🔎 Trace | Pipeline timeline — coloured cards per event, query history table |
| 📊 Evaluate | Ragas golden-set scorecard + per-question breakdown |

### Sidebar Settings

| Setting | Default | Notes |
|---|---|---|
| Cohere API Key | `""` | Password input |
| Cohere model | `rerank-english-v3.0` | Selectbox |
| Enable Cohere Reranking | `False` | Disabled if no key |
| Enable Query Rewriting | `True` | |
| Max Rewrites | `2` | Slider 1–5 |
| Score Threshold | `0.3` | Slider 0.0–1.0 step 0.05 |
| Score faithfulness after each answer | `False` | Adds 4 LLM calls per query |

---

## Self-Evaluation — 4 Metrics (per-query)

When "Score faithfulness after each answer" is ON, four LLM calls fire after generation:

| Metric | Prompt inputs | What it measures |
|---|---|---|
| Faithfulness | context + answer | Every answer claim is grounded in context |
| Answer Relevancy | question + answer | Answer actually addresses the question |
| Context Precision | question + context | Retrieved chunks are on-topic (not noisy) |
| Context Recall | question + context | Context covers all info needed to answer |

Each prompt asks for a single integer 0–10. Score is divided by 10 → float in [0, 1].  
Any metric < 0.7 flags the card red in the Trace tab.

---

## Ragas Batch Evaluation — Golden Set

### How it works

1. All 10 golden questions run through `run_pipeline` (rewrite=OFF, rerank=OFF for clean baseline).
2. Results wrapped in `EvaluationDataset(samples=[SingleTurnSample(...)])`.
3. Ragas scores 4 metrics using llama3.2 as judge + nomic-embed-text for embeddings.
4. Aggregate means computed; weakest metric identified; actionable advice shown.

### The 4 Ragas Metrics

| Metric | Requires ground truth? | What it measures |
|---|---|---|
| Faithfulness | No | LLM answer is grounded in retrieved context |
| AnswerRelevancy | No | Answer is relevant to the question |
| ContextPrecision | No | Relevant chunks rank higher than irrelevant ones |
| ContextRecall | **Yes** | All gold-truth information was actually retrieved |

### Weakest Metric → Next Investment

| Weakest | Root cause | Fix |
|---|---|---|
| Faithfulness | LLM adds claims not in context | Tighten system prompt; lower TOP_K |
| Answer Relevancy | Answers drift off-topic | Shorten / sharpen the system prompt |
| Context Precision | Retrieved chunks are noisy | Reduce TOP_K; increase CHUNK_SIZE |
| Context Recall | Relevant chunks are missed | Increase TOP_K; reduce CHUNK_SIZE; enable rewriting |

---

## Golden Set (golden_set.json) — 10 Questions

| # | Document | Question | Ground Truth |
|---|---|---|---|
| 1 | incident_escalation | Immediate actions for SEV1? | Page on-call engineer immediately; notify manager within 5 min |
| 2 | incident_escalation | How often are SEV1 status updates? | Every 15 minutes |
| 3 | incident_escalation | Primary on-call no response in 10 min? | Page the secondary |
| 4 | hr_leave_policy | Annual leave days per year? | 20 days |
| 5 | hr_leave_policy | Unused leave carry-over? | Up to 5 days; excess forfeited 31 Dec |
| 6 | hr_leave_policy | Notice for leave > 2 days? | At least 10 working days |
| 7 | it_support_policy | P1 first-response target? | Within 30 minutes |
| 8 | it_support_policy | Security incident report deadline? | Within 1 hour |
| 9 | expense_claims | Daily meal cap when travelling? | $80/day (breakfast + lunch + dinner) |
| 10 | expense_claims | How long keep original receipts? | 90 days after submission |

---

## Key Bugs Found and Fixed

### 1. Chunk size caused orphaned header chunks
- **Problem**: `chunk_size=500` created tiny chunks (50 chars, 20 chars) that were pure headers with no content. The SEV1 paging instruction was in a chunk that never reached top-3 retrieval.
- **Fix**: `chunk_size=1000, chunk_overlap=100` → 3 clean section-aligned chunks for the incident escalation doc; 11 chunks total across 4 docs.

### 2. Score threshold triggered rewrites on every valid query
- **Problem**: Default `score_threshold=0.5`; L2 relevance scores for valid matches sit at 0.45–0.55. Result: every query fired 2 rewrites (3 LLM calls + 2 retrieval calls) even when the answer was correct.
- **Fix**: `score_threshold=0.3` — rewrites only fire when retrieval genuinely fails.

### 3. ragas 0.4.3 import failure
- **Problem**: `ragas/llms/base.py` imports `ChatVertexAI` from `langchain_community.chat_models.vertexai` at module load. This module was removed in `langchain-community 0.4`.
- **Fix**: Created a stub file at `langchain_community/chat_models/vertexai.py` that falls back to an empty class if `langchain-google-vertexai` is not installed.

---

## ChromaDB Persistence Logic

```python
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    # Second+ run: load existing embeddings — fast
    vectorstore = Chroma(persist_directory=..., embedding_function=embeddings)
else:
    # First run: load docs → split → embed → write to disk
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, ...)
```

Re-index from scratch: `rm -rf chroma_db/` then restart.

---

## Ollama Setup (one-time)

```bash
ollama pull llama3.2          # LLM (2.0 GB)
ollama pull nomic-embed-text  # Embeddings (274 MB)
```

---

## Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

---

## Component Interview Summary

Scores from self-assessment interview (answers scored on depth/accuracy):

| Component | Score | Main gap |
|---|---|---|
| Document Loader | 2/5 | Missed `Document` object and source-agnostic standardisation |
| Text Splitter | 2/5 | Said "split into chunks" without explaining token limits, retrieval precision, or overlap |
| Embedding Model | 2/5 | Described retrieval instead of what a vector is |
| Vector Store | 2/5 | Missed vector+text pairing and persistence guard |
| Retriever | 2/5 | Missed that k=3 returns multiple chunks, not one |
| Prompt Template | 2/5 | Missed template structure and grounding purpose |
| LLM | 3/5 | Right direction; missed synthesis-not-retrieval distinction |
| Chain | 2/5 | Said "connect"; missed `RunnablePassthrough` and no-memory explanation |

**Pattern**: correct intuition about *what* happens, weak on *why* each component exists as a distinct piece.

---

## Document Key Facts (for golden-set reference)

### incident_escalation.md
- SEV1: page on-call immediately, notify manager within **5 min**
- SEV2: page on-call, manager notification encouraged but not required
- SEV3: file a ticket, no paging
- SEV4: tracking only
- Escalation: primary → 10 min → secondary → 20 min → engineering manager
- Status updates: every **30 min** (SEV1: every **15 min**)
- Postmortem: within **5 working days** for SEV1/SEV2; blameless

### hr_leave_policy.md
- Annual leave: **20 days/year**, accrues at 1.67 days/month
- Carry-over: up to **5 days**; excess forfeited 31 Dec
- Leave request notice: **10 working days** in advance (for > 2 days)
- Sick leave: **10 days/year**; cert required for 3+ consecutive days
- Primary caregiver parental: **16 weeks**; secondary: **4 weeks**
- Carer's leave: up to **5 days/year** (draws from sick leave)

### it_support_policy.md
- P1 (cannot work): response **30 min**, resolution **4 working hours**
- P2 (degraded): response **2 working hours**, resolution **1 working day**
- P3 (minor): response **1 working day**, resolution **5 working days**
- Security incident: report within **1 hour**
- Software approval: **5 working days**

### expense_claims.md
- Daily meal cap: **$80** (breakfast + lunch + dinner)
- Alcohol: not reimbursable (except pre-approved client dinner)
- Manager approval target: **3 working days**
- Finance processing: **5 working days** after manager approval
- Keep receipts: **90 days** after submission
- Submit in original currency; portal converts automatically
