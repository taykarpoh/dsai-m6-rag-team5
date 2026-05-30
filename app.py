import streamlit as st
from rag import init_rag

st.set_page_config(page_title="Incident Escalation RAG", layout="centered")
st.title("Incident Escalation RAG")
st.caption("Classic RAG — no memory. Each question is fully independent.")


@st.cache_resource
def load_rag():
    return init_rag()


chain, retriever = load_rag()

question = st.text_input("Ask a question about incident escalation:")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating..."):
        answer = chain.invoke(question)
        sources = retriever.invoke(question)

    st.markdown("### Answer")
    st.markdown(answer)

    with st.expander("Retrieved source chunks"):
        for i, doc in enumerate(sources, 1):
            st.markdown(f"**Chunk {i}**")
            st.text(doc.page_content)
            st.divider()
