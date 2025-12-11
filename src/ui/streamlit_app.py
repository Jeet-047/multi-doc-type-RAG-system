import os
import time
import requests
import streamlit as st


st.set_page_config(
    page_title="Multi-Doc RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple theming
st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .block-container { padding-top: 2rem; }
    .title-text { font-size: 2rem; font-weight: 700; color: #f8fafc; }
    .sub-text { color: #cbd5e1; }
    .chat-bubble { background: #1e293b; padding: 1rem; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_backend_url() -> str:
    return st.session_state.get("backend_url", "http://localhost:8000")


def set_backend_url(url: str) -> None:
    st.session_state["backend_url"] = url.rstrip("/")


def api_health() -> bool:
    try:
        resp = requests.get(f"{get_backend_url()}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def api_status() -> dict:
    """Check the processing status."""
    try:
        resp = requests.get(f"{get_backend_url()}/status", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def api_load(files, url: str, progress_placeholder) -> dict:
    """Load documents and poll for completion."""
    payload = {}
    if url:
        payload["url"] = url.strip()
    files_param = [("files", (f.name, f, f.type)) for f in files] if files else None
    
    # Start the load process
    resp = requests.post(
        f"{get_backend_url()}/load",
        files=files_param,
        data=payload,
        timeout=30,  # Reduced timeout since we return immediately
    )
    resp.raise_for_status()
    result = resp.json()
    
    # If processing started, poll for completion
    if result.get("status") == "processing":
        max_wait = 300  # 5 minutes max wait
        start_time = time.time()
        poll_interval = 2  # Check every 2 seconds
        
        while time.time() - start_time < max_wait:
            status_resp = api_status()
            status = status_resp.get("status", "processing")
            
            if status == "ready":
                progress_placeholder.success("✅ Documents indexed successfully!")
                time.sleep(0.5)  # Brief pause to show success message
                progress_placeholder.empty()  # Clear the placeholder
                return {"status": "ready", "message": "Documents indexed. Ready for queries."}
            elif status == "error":
                error_msg = status_resp.get("error", "Unknown error occurred")
                progress_placeholder.error(f"❌ Error: {error_msg}")
                raise Exception(error_msg)
            else:
                # Still processing
                elapsed = int(time.time() - start_time)
                progress_placeholder.info(f"⏳ Processing documents... ({elapsed}s elapsed)")
                time.sleep(poll_interval)
        
        # Timeout
        progress_placeholder.error("⏱️ Processing timed out after 5 minutes.")
        raise Exception("Processing timed out. Please try again or check server logs.")
    
    return result


def api_query(query: str) -> str:
    resp = requests.post(
        f"{get_backend_url()}/query",
        json={"query": query},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("answer", "")


def sidebar():
    st.sidebar.title("⚙️ Settings")
    backend_url = st.sidebar.text_input("Backend URL", get_backend_url())
    set_backend_url(backend_url)

    health_ok = api_health()
    status = "🟢 Backend reachable" if health_ok else "🔴 Backend offline"
    st.sidebar.markdown(f"**Status:** {status}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Load documents**")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF/DOCX/TXT/MD", type=["pdf", "docx", "doc", "txt", "md"], accept_multiple_files=True
    )
    url_input = st.sidebar.text_input("or Provide a URL")

    if st.sidebar.button("Index documents", type="primary"):
        if not uploaded_files and not url_input.strip():
            st.sidebar.warning("Upload at least one file or provide a URL.")
            return
        
        progress_placeholder = st.sidebar.empty()
        try:
            resp = api_load(uploaded_files, url_input, progress_placeholder)
            st.session_state["ready"] = True
            st.sidebar.success(resp.get("message", "Ready"))
        except Exception as e:
            st.sidebar.error(f"Failed to index documents: {e}")
            st.session_state["ready"] = False


def chat_area():
    st.markdown('<div class="title-text">Multi-Doc RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Upload a document or drop a URL, then ask questions with grounded answers.</div>', unsafe_allow_html=True)

    ready = st.session_state.get("ready", False)
    query = st.text_input("Ask a question", placeholder="e.g., Summarize the methodology", disabled=not ready)

    if st.button("Get Answer", type="primary", disabled=not ready):
        if not query.strip():
            st.warning("Enter a question.")
            return
        with st.spinner("Retrieving and generating..."):
            try:
                answer = api_query(query.strip())
                st.success("Answer ready")
                st.markdown(f'<div class="chat-bubble">{answer}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Query failed: {e}")


def main():
    sidebar()
    chat_area()


if __name__ == "__main__":
    main()

