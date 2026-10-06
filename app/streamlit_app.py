from pathlib import Path
import re

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

# Project structure:
#
# university-knowledge-rag/
# ├── app/
# │   └── streamlit_app.py
# ├── data/
# │   └── raw/
# ├── src/
# └── ...
#
# __file__ = .../university-knowledge-rag/app/streamlit_app.py
# parent   = .../university-knowledge-rag/app
# parent   = .../university-knowledge-rag

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="University Knowledge RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .source-box {
            border-left: 4px solid #4F46E5;
            padding: 0.6rem 1rem;
            margin-top: 0.5rem;
            background-color: #f7f7fb;
            border-radius: 4px;
        }

        .status-box {
            padding: 0.75rem;
            border-radius: 6px;
            background-color: #f5f5f5;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_documents():
    """
    Return supported documents from data/raw/.
    """
    if not RAW_DATA_DIR.exists():
        return []

    supported_extensions = {
        ".txt",
        ".pdf",
        ".md",
    }

    return sorted(
        [
            path
            for path in RAW_DATA_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in supported_extensions
        ],
        key=lambda p: p.name.lower(),
    )


def read_text_file(path: Path) -> str:
    """
    Read TXT/MD files safely.
    """
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return ""


def read_pdf_file(path: Path) -> str:
    """
    Extract text from a PDF using pypdf.

    Install with:
        pip install pypdf
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))

        pages = []

        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                pages.append(text)
            except Exception:
                continue

        return "\n\n".join(pages)

    except ImportError:
        return (
            "PDF support is unavailable because pypdf is not installed. "
            "Run: pip install pypdf"
        )
    except Exception as exc:
        return f"Unable to read PDF: {exc}"


@st.cache_data(show_spinner=False)
def load_document_text(path_string: str) -> str:
    """
    Load document text.

    Cached because Streamlit reruns the script on interaction.
    """
    path = Path(path_string)

    if not path.exists():
        return ""

    if path.suffix.lower() in {".txt", ".md"}:
        return read_text_file(path)

    if path.suffix.lower() == ".pdf":
        return read_pdf_file(path)

    return ""


def clean_text(text: str) -> str:
    """
    Basic text normalisation.
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150):
    """
    Simple character-based chunking.

    This is a lightweight fallback RAG implementation.
    You can replace it later with your LangChain/LlamaIndex
    chunker if your project already has one.
    """
    text = clean_text(text)

    if not text:
        return []

    if overlap >= chunk_size:
        overlap = chunk_size // 5

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def keyword_search(query: str, documents, top_k: int = 5):
    """
    Simple local retrieval fallback.

    It scores chunks based on matching query terms.

    This allows the Streamlit app to work before you connect
    your real embedding/vector database backend.
    """
    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]{3,}\b",
            query.lower(),
        )
    )

    if not query_words:
        return []

    results = []

    for document_path in documents:
        text = load_document_text(str(document_path))

        if not text:
            continue

        chunks = split_into_chunks(text)

        for index, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()

            score = 0

            for word in query_words:
                if word in chunk_lower:
                    score += 1

            if score > 0:
                results.append(
                    {
                        "score": score,
                        "source": document_path.name,
                        "path": str(document_path),
                        "chunk_index": index,
                        "text": chunk,
                    }
                )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]


def generate_answer(query: str, results):
    """
    Generate a transparent extractive answer.

    This intentionally does NOT pretend to be an LLM.
    It returns the most relevant retrieved passages.

    Replace this function later with your actual LLM call.
    """

    if not results:
        return (
            "I could not find relevant information in the documents "
            "currently stored in data/raw/."
        )

    answer_parts = []

    answer_parts.append(
        "Based on the university documents currently indexed, "
        "the most relevant information I found is:"
    )

    for result in results[:3]:
        answer_parts.append(
            f"\n**{result['source']}**\n\n"
            f"{result['text']}"
        )

    return "\n\n".join(answer_parts)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 University RAG")

    st.markdown(
        "University knowledge retrieval system"
    )

    st.divider()

    st.subheader("Project")

    st.write(
        f"**Project root:**\n"
        f"`{PROJECT_ROOT}`"
    )

    st.write(
        f"**Document folder:**\n"
        f"`{RAW_DATA_DIR}`"
    )

    st.divider()

    documents = get_documents()

    st.metric(
        "Documents found",
        len(documents),
    )

    if documents:

        st.subheader("Indexed documents")

        for document in documents:
            st.write(f"📄 {document.name}")

    else:

        st.warning(
            "No documents found in data/raw/"
        )

        st.info(
            "Add your University of Bradford PDF/TXT files "
            "to data/raw/ and refresh the page."
        )

    st.divider()

    st.caption(
        "University Knowledge RAG Project"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 University Knowledge RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Ask questions about the university documents stored in your RAG corpus.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DOCUMENT STATUS
# ============================================================

if not RAW_DATA_DIR.exists():

    st.error(
        f"Document directory does not exist:\n\n"
        f"`{RAW_DATA_DIR}`"
    )

    st.info(
        "Create the folder and add your documents:\n\n"
        "`data/raw/`"
    )

    st.stop()


if not documents:

    st.warning(
        "No documents are currently available."
    )

    st.info(
        "Add PDF or TXT files to:\n\n"
        f"`{RAW_DATA_DIR}`"
    )

else:

    st.success(
        f"{len(documents)} document(s) available for retrieval."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-box">
                            <strong>{source}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about the university documents..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if query:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve documents
    with st.chat_message("assistant"):

        with st.spinner("Searching university documents..."):

            results = keyword_search(
                query=query,
                documents=documents,
                top_k=5,
            )

            answer = generate_answer(
                query=query,
                results=results,
            )

        st.markdown(answer)

        # Display retrieved sources
        if results:

            unique_sources = []

            for result in results:

                source = result["source"]

                if source not in unique_sources:
                    unique_sources.append(source)

            with st.expander("📚 Retrieved sources"):

                for source in unique_sources:

                    st.markdown(
                        f"""
                        <div class="source-box">
                            📄 {source}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.divider()

                for number, result in enumerate(results, start=1):

                    st.markdown(
                        f"**Result {number} — "
                        f"{result['source']}**"
                    )

                    st.caption(
                        f"Retrieval score: {result['score']}"
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()

        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": (
                    [
                        result["source"]
                        for result in results
                    ]
                    if results
                    else []
                ),
            }
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Documents are loaded locally from data/raw/. "
    "This application does not upload your university documents "
    "to an external service."
)