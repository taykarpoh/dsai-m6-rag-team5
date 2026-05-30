import pandas as pd
import streamlit as st

from rag import init_rag, run_pipeline, run_evaluation, trace_log

st.set_page_config(page_title="Company Policies RAG", layout="wide")


# ---------------------------------------------------------------------------
# Trace rendering helpers
# ---------------------------------------------------------------------------

def _card(color: str, title: str, body: str = "") -> None:
    st.markdown(
        f"""<div style="border-left:4px solid {color};padding:8px 14px;
        margin:6px 0;background:#f8f9fa;border-radius:4px">
        <strong>{title}</strong>{"<br>" + body if body else ""}
        </div>""",
        unsafe_allow_html=True,
    )


def render_trace_timeline(trace: dict) -> None:
    st.subheader("Pipeline Timeline")
    threshold = trace.get("_threshold", 0.5)

    for event in trace["events"]:
        etype = event["type"]

        if etype == "retrieval":
            top = event["top_score"]
            color = "#28a745" if top >= threshold else "#fd7e14"
            status = f"top score {top:.3f} {'≥' if top >= threshold else '<'} threshold"
            _card(color, f"🔍 Retrieval — attempt {event['attempt']} — {status}",
                  f"Query: <em>{event['query']}</em>")
            cols = st.columns(len(event["scores"]))
            for i, (col, score) in enumerate(zip(cols, event["scores"])):
                col.metric(f"Chunk {i + 1}", f"{score:.3f}")

        elif etype == "rewrite":
            _card("#fd7e14", "✏️ Query Rewrite")
            c1, arr, c2 = st.columns([5, 1, 5])
            c1.info(f"**Original:** {event['original']}")
            arr.markdown("<br><br><div style='text-align:center;font-size:1.5em'>→</div>",
                         unsafe_allow_html=True)
            c2.success(f"**Rewritten:** {event['rewritten']}")

        elif etype == "rerank":
            _card("#007bff", f"⚡ Cohere Rerank — {event['model']}")
            cols = st.columns(len(event["scores"]))
            for i, (col, score) in enumerate(zip(cols, event["scores"])):
                col.metric(f"Ranked {i + 1}", f"{score:.4f}")

        elif etype == "rerank_error":
            st.error(f"Rerank failed: {event['error']}")

        elif etype == "self_eval":
            flagged = event.get("flagged", False)
            color = "#dc3545" if flagged else "#28a745"
            _card(color, "🧠 Self-Evaluation — 4 metrics")
            metric_keys = [
                ("faithfulness",     "Faithfulness"),
                ("answer_relevancy", "Ans. Relevancy"),
                ("context_precision","Ctx. Precision"),
                ("context_recall",   "Ctx. Recall"),
            ]
            cols = st.columns(4)
            for col, (key, label) in zip(cols, metric_keys):
                val = event.get(key)
                display = f"{val * 10:.0f}/10" if val is not None else "—"
                col.metric(label, display)
            if flagged:
                st.warning("One or more metrics scored below 7/10.")

        elif etype == "answer":
            _card("#6c757d", "💬 Answer generated", event["preview"])

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def render_trace_history() -> None:
    if len(trace_log) < 2:
        return
    st.divider()
    st.subheader("Query History")
    rows = [{
        "Time": t["timestamp"],
        "Original Query": t["original_query"],
        "Final Query": t["final_query"],
        "Rewrites": t["rewrite_count"],
        "Answer": (t.get("answer", "")[:80] + "…")
                  if len(t.get("answer", "")) > 80 else t.get("answer", ""),
    } for t in trace_log]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Evaluate tab helpers
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "faithfulness":     "Faithfulness",
    "answer_relevancy": "Ans. Relevancy",
    "context_precision": "Ctx. Precision",
    "context_recall":   "Ctx. Recall",
}


def _render_scorecard(eval_result: dict) -> None:
    scores  = eval_result["scores"]
    weakest = eval_result["weakest_metric"]

    st.subheader("Aggregate Scores")
    cols = st.columns(4)
    for col, (metric, val) in zip(cols, scores.items()):
        is_weakest = metric == weakest
        col.metric(
            METRIC_LABELS.get(metric, metric),
            f"{val:.3f}" if val is not None else "—",
            delta="← weakest" if is_weakest else None,
            delta_color="inverse" if is_weakest else "normal",
        )

    st.subheader("Per-Question Breakdown")
    rows = [{
        "Question":       q["question"][:65] + "…" if len(q["question"]) > 65 else q["question"],
        "Faithfulness":   round(q.get("faithfulness") or 0, 3),
        "Ans. Relevancy": round(q.get("answer_relevancy") or 0, 3),
        "Ctx. Precision": round(q.get("context_precision") or 0, 3),
        "Ctx. Recall":    round(q.get("context_recall") or 0, 3),
    } for q in eval_result["per_question"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    st.subheader(f"⚠️ Weakest Metric: {METRIC_LABELS.get(weakest, weakest)}")
    st.error(eval_result["weakest_advice"])


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Cohere Reranking")
    cohere_key = st.text_input("Cohere API Key", type="password",
                                placeholder="sk-…", key="cohere_key")
    cohere_model = st.selectbox(
        "Rerank model",
        ["rerank-english-v3.0", "rerank-multilingual-v3.0", "rerank-v3.5"],
        index=0,
    )
    enable_rerank = st.toggle(
        "Enable Cohere Reranking",
        value=False,
        disabled=not cohere_key,
        help="Requires a Cohere API key above.",
    )

    st.divider()
    st.subheader("Query Rewriting")
    enable_rewrite = st.toggle("Enable Query Rewriting", value=True)
    max_rewrites = st.slider("Max Rewrites", min_value=1, max_value=5, value=2,
                              disabled=not enable_rewrite)
    score_threshold = st.slider("Score Threshold", min_value=0.0, max_value=1.0,
                                 value=0.3, step=0.05,
                                 help="Trigger rewrite when top similarity < this value.",
                                 disabled=not enable_rewrite)

    st.divider()
    st.subheader("Self-Evaluation")
    enable_self_eval = st.toggle(
        "Score faithfulness after each answer",
        value=False,
        help="Fires a second LLM call to rate the answer 0–10 against the context.",
    )


# ---------------------------------------------------------------------------
# RAG initialisation (once per server session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_rag():
    init_rag()


load_rag()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Company Policies RAG")
st.caption("Covers: incident escalation · HR leave · IT support · expense claims. "
           "No memory — each question is fully independent.")

question = st.text_input("Ask a question about company policy:")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Running pipeline…"):
        result = run_pipeline(
            question=question,
            cohere_api_key=cohere_key or None,
            cohere_model=cohere_model,
            enable_rerank=enable_rerank,
            enable_rewrite=enable_rewrite,
            enable_self_eval=enable_self_eval,
            max_rewrites=max_rewrites,
            score_threshold=score_threshold,
        )
        result["trace"]["_threshold"] = score_threshold
    st.session_state["last_result"] = result

# ---------------------------------------------------------------------------
# Results — four tabs
# ---------------------------------------------------------------------------

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    tab_answer, tab_sources, tab_trace, tab_eval = st.tabs(
        ["💬 Answer", "📄 Sources", "🔎 Trace", "📊 Evaluate"]
    )

    with tab_answer:
        st.markdown("### Answer")
        st.markdown(result["answer"])

    with tab_sources:
        st.markdown("### Retrieved Chunks")
        for i, doc in enumerate(result["sources"], 1):
            src = doc.metadata.get("source", "unknown")
            label = f"Chunk {i} — {src.split('/')[-1]}  ({len(doc.page_content)} chars)"
            with st.expander(label):
                st.text(doc.page_content)

    with tab_trace:
        render_trace_timeline(result["trace"])
        render_trace_history()

    with tab_eval:
        st.markdown("### Ragas Evaluation — Golden Set (10 questions)")
        st.caption(
            "Runs all 10 golden questions through the pipeline with rewrite and rerank "
            "disabled for a clean baseline. Uses llama3.2 as the judge LLM."
        )
        if st.button("▶ Run Evaluation", type="primary", key="run_eval"):
            with st.spinner("Running 10 questions × 4 Ragas metrics — takes 3–8 min…"):
                try:
                    eval_result = run_evaluation()
                    st.session_state["eval_result"] = eval_result
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

        if "eval_result" in st.session_state:
            _render_scorecard(st.session_state["eval_result"])
else:
    # Show Evaluate tab even before any question is asked
    st.markdown("---")
    st.markdown("#### 📊 Evaluate")
    st.caption(
        "Ask a question first to unlock the Answer/Sources/Trace tabs, "
        "or jump straight to evaluation:"
    )
    if st.button("▶ Run Evaluation", type="primary", key="run_eval_pre"):
        with st.spinner("Running 10 questions × 4 Ragas metrics — takes 3–8 min…"):
            try:
                eval_result = run_evaluation()
                st.session_state["eval_result"] = eval_result
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    if "eval_result" in st.session_state:
        _render_scorecard(st.session_state["eval_result"])
