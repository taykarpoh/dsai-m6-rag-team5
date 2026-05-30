import os
import datetime

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
DOC_DIR = "doc"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "company_policies"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 3

# --- Module-level components (populated by init_rag) ---
_vectorstore: Chroma | None = None
_retriever = None
_llm = None
_prompt = None

# --- Trace store — grows with every run_pipeline() call ---
trace_log: list[dict] = []


def format_docs(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)


def init_rag():
    """
    Build all 8 classic RAG components. Sets module-level vars so the
    advanced pipeline functions can use them. Returns (chain, retriever)
    for backward compatibility.
    """
    global _vectorstore, _retriever, _llm, _prompt

    # 1. Embedding Model
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 2+3+4. Load → Split → Embed into Vector Store (skip if already persisted)
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        _vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
    else:
        loader = DirectoryLoader(DOC_DIR, glob="*.md", loader_cls=TextLoader,
                                  show_progress=False, use_multithreading=False)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        _vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )

    # 5. Retriever
    _retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # 6. LLM
    _llm = ChatOllama(model=LLM_MODEL)

    # 7. Prompt Template — grounded, no prior knowledge
    system_prompt = (
        "You are an assistant for answering questions about company policies and procedures, "
        "including incident escalation, HR leave, IT support, and expense claims. "
        "Answer using ONLY the context below. "
        "If the answer is not in the context, say 'I don't have that information.'\n\n"
        "Context:\n{context}"
    )
    _prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # 8. LCEL Chain — stateless, no memory, no conversation history
    chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | _prompt
        | _llm
        | StrOutputParser()
    )

    return chain, _retriever


# ---------------------------------------------------------------------------
# Advanced pipeline components
# ---------------------------------------------------------------------------

_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at reformulating search queries to improve document retrieval. "
     "Rewrite the query to be more specific and retrievable. "
     "Return ONLY the rewritten query, no explanation."),
    ("human", "Query: {question}"),
])


def retrieve_with_scores(question: str) -> list[tuple]:
    """Return (Document, relevance_score) pairs. Score in [0, 1], higher = more similar."""
    pairs = _vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
    return [(doc, max(0.0, min(1.0, score))) for doc, score in pairs]


def rewrite_query(question: str) -> str:
    """Use LLM to rewrite question for better retrieval. Returns plain string."""
    return (_rewrite_prompt | _llm | StrOutputParser()).invoke(
        {"question": question}
    ).strip()


def rerank_documents(
    question: str,
    docs: list,
    cohere_api_key: str,
    model: str = "rerank-english-v3.0",
    top_n: int = TOP_K,
) -> list[tuple]:
    """Cohere rerank. Returns (Document, relevance_score) list ordered by score desc."""
    from langchain_cohere import CohereRerank
    reranker = CohereRerank(cohere_api_key=cohere_api_key, model=model, top_n=top_n)
    compressed = reranker.compress_documents(docs, question)
    return [
        (doc, float(doc.metadata.get("relevance_score", 0.0)))
        for doc in compressed
    ]


_SELF_EVAL_PROMPTS = {
    "faithfulness": ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict evaluator. Rate 0-10 whether every claim in the answer is "
         "directly supported by the context. "
         "10 = fully grounded, 0 = answer contradicts or ignores the context. "
         "Reply with ONLY a single integer 0-10, nothing else."),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ]),
    "answer_relevancy": ChatPromptTemplate.from_messages([
        ("system",
         "Rate 0-10 how relevant the answer is to the question. "
         "10 = answer directly and completely addresses the question, "
         "0 = answer is completely off-topic. "
         "Reply with ONLY a single integer 0-10, nothing else."),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ]),
    "context_precision": ChatPromptTemplate.from_messages([
        ("system",
         "Rate 0-10 how precise the retrieved context is for answering the question. "
         "10 = every retrieved passage is directly relevant, "
         "0 = all retrieved passages are irrelevant noise. "
         "Reply with ONLY a single integer 0-10, nothing else."),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]),
    "context_recall": ChatPromptTemplate.from_messages([
        ("system",
         "Rate 0-10 how completely the retrieved context covers the information needed "
         "to answer the question. "
         "10 = context has everything required, 0 = context is missing all relevant info. "
         "Reply with ONLY a single integer 0-10, nothing else."),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]),
}


def _score_metric(prompt: ChatPromptTemplate, inputs: dict) -> float | None:
    """Invoke a self-eval prompt and parse the 0-10 integer reply → float in [0, 1]."""
    try:
        raw = (prompt | _llm | StrOutputParser()).invoke(inputs).strip()
        return int(raw.split()[0]) / 10.0
    except Exception:
        return None


def run_pipeline(
    question: str,
    cohere_api_key: str = None,
    cohere_model: str = "rerank-english-v3.0",
    enable_rerank: bool = True,
    enable_rewrite: bool = True,
    enable_self_eval: bool = False,
    max_rewrites: int = 2,
    score_threshold: float = 0.3,
) -> dict:
    """
    Full RAG pipeline:
      retrieval (with scores) → optional rewrite-retry loop
      → optional Cohere rerank → LLM generation → trace log append.

    Returns: {"answer": str, "sources": list[Document], "trace": dict}
    """
    trace = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "original_query": question,
        "final_query": question,
        "rewrite_count": 0,
        "events": [],
    }

    current_query = question
    final_docs: list = []

    # --- Retrieval loop with optional query rewriting ---
    for attempt in range(max_rewrites + 1):
        pairs = retrieve_with_scores(current_query)
        docs = [d for d, _ in pairs]
        scores = [s for _, s in pairs]
        top_score = max(scores, default=0.0)

        trace["events"].append({
            "type": "retrieval",
            "attempt": attempt + 1,
            "query": current_query,
            "scores": scores,
            "top_score": top_score,
        })

        # Good enough, or rewriting disabled, or exhausted budget → stop
        if top_score >= score_threshold or not enable_rewrite or attempt >= max_rewrites:
            final_docs = docs
            break

        # Score below threshold — rewrite and retry
        new_query = rewrite_query(current_query)
        trace["events"].append({
            "type": "rewrite",
            "original": current_query,
            "rewritten": new_query,
        })
        current_query = new_query
        trace["rewrite_count"] += 1

    trace["final_query"] = current_query

    # --- Cohere reranking ---
    if enable_rerank and cohere_api_key:
        try:
            ranked = rerank_documents(current_query, final_docs, cohere_api_key, model=cohere_model)
            final_docs = [d for d, _ in ranked]
            trace["events"].append({
                "type": "rerank",
                "model": cohere_model,
                "scores": [s for _, s in ranked],
            })
        except Exception as exc:
            trace["events"].append({"type": "rerank_error", "error": str(exc)})

    # --- Generation ---
    generate_chain = _prompt | _llm | StrOutputParser()
    answer = generate_chain.invoke({
        "context": format_docs(final_docs),
        "question": current_query,
    })

    trace["answer"] = answer
    trace["events"].append({
        "type": "answer",
        "preview": answer[:120] + "..." if len(answer) > 120 else answer,
    })

    # --- Per-query self-evaluation — all 4 metrics (optional) ---
    if enable_self_eval:
        ctx = format_docs(final_docs)
        inputs_fa = {"context": ctx, "answer": answer}
        inputs_qa = {"question": current_query, "answer": answer}
        inputs_qc = {"question": current_query, "context": ctx}

        scores_4 = {
            "faithfulness":     _score_metric(_SELF_EVAL_PROMPTS["faithfulness"],    inputs_fa),
            "answer_relevancy": _score_metric(_SELF_EVAL_PROMPTS["answer_relevancy"], inputs_qa),
            "context_precision":_score_metric(_SELF_EVAL_PROMPTS["context_precision"],inputs_qc),
            "context_recall":   _score_metric(_SELF_EVAL_PROMPTS["context_recall"],   inputs_qc),
        }
        flagged = any(v is not None and v < 0.7 for v in scores_4.values())
        trace["events"].append({"type": "self_eval", "flagged": flagged, **scores_4})

    trace_log.append(trace)
    return {"answer": answer, "sources": final_docs, "trace": trace}


# ---------------------------------------------------------------------------
# Ragas batch evaluation
# ---------------------------------------------------------------------------

ADVICE_MAP = {
    "faithfulness":
        "LLM is generating claims not grounded in the retrieved context. "
        "→ Tighten the system prompt, reduce chunk size so each chunk is more focused, "
        "or lower TOP_K to reduce noise.",
    "answer_relevancy":
        "Answers are drifting off-topic. "
        "→ Review the system prompt instruction; make it shorter and more directive. "
        "Also check whether the LLM model is too verbose.",
    "context_precision":
        "Retrieved chunks contain too much noise relative to signal. "
        "→ Reduce TOP_K, increase CHUNK_SIZE to keep related content together, "
        "or add source metadata filters.",
    "context_recall":
        "Relevant passages are being missed at retrieval time. "
        "→ Increase TOP_K, reduce CHUNK_SIZE for finer granularity, "
        "or enable query rewriting to broaden the search.",
}


def run_evaluation(golden_set_path: str = "golden_set.json") -> dict:
    """
    Run the RAG pipeline against the golden set and score with Ragas.

    Returns:
        {
          "scores":         {"faithfulness": float, ...},
          "per_question":   list[dict],
          "weakest_metric": str,
          "weakest_advice": str,
        }
    """
    import json
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    with open(golden_set_path) as f:
        golden = json.load(f)

    judge_llm = LangchainLLMWrapper(_llm)
    judge_emb  = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_emb),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    # Generous timeout for local Ollama; 1 retry to handle transient failures
    run_cfg = RunConfig(timeout=120, max_retries=1, max_wait=10)

    samples: list = []
    per_question: list = []

    for item in golden:
        result = run_pipeline(item["question"],
                              enable_rerank=False, enable_rewrite=False)
        contexts = [doc.page_content for doc in result["sources"]]
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            retrieved_contexts=contexts,
            reference=item["ground_truth"],
        ))
        per_question.append({
            "question":     item["question"],
            "answer":       result["answer"],
            "ground_truth": item["ground_truth"],
        })

    dataset = EvaluationDataset(samples=samples)
    ragas_result = evaluate(dataset, metrics=metrics, run_config=run_cfg)

    df = ragas_result.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = df[metric_cols].mean().to_dict()

    for i, row in df.iterrows():
        per_question[i].update({col: row.get(col) for col in metric_cols})

    weakest = min(scores, key=lambda k: scores[k] if scores[k] is not None else 1.0)

    return {
        "scores":         scores,
        "per_question":   per_question,
        "weakest_metric": weakest,
        "weakest_advice": ADVICE_MAP[weakest],
    }
