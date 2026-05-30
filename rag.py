import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
DOC_PATH = "doc/incident_escalation.md"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "incident_escalation"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 3


def format_docs(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)


def init_rag():
    """
    Build all 8 classic RAG components and return (chain, retriever).
    Components are stateless — no memory, no conversation history.
    """

    # 1. Embedding Model
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 2+3+4. Load → Split → Embed into Vector Store (skip if already persisted)
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
    else:
        # 2. Document Loader
        loader = TextLoader(DOC_PATH)
        docs = loader.load()

        # 3. Text Splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)

        # 4. Vector Store — first-time ingest
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )

    # 5. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # 6. LLM
    llm = ChatOllama(model=LLM_MODEL)

    # 7. Prompt Template — grounded, no prior knowledge
    system_prompt = (
        "You are an assistant for answering questions about incident escalation procedures. "
        "Answer using ONLY the context below. "
        "If the answer is not in the context, say 'I don't have that information.'\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # 8. LCEL Chain — stateless, no memory, no conversation history
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
