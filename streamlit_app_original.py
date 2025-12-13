import os
import time
import html
import requests
import streamlit as st
from typing import Dict, Any


# --- Configuration and Initial Setup ---
st.set_page_config(
    page_title="Multi-Doc RAG Assistant by Jeet Majumder",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "backend_url" not in st.session_state:
    st.session_state["backend_url"] = "http://localhost:8000"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_status" not in st.session_state:
    st.session_state["rag_status"] = {"status": "idle", "loaded_documents": []}


# Custom theming with Tailwind-inspired colors for a modern look
st.markdown(
    """
    <style>
    /* Main Streamlit App and Background */
    .stApp { background-color: #0f172a; color: #e2e8f0; } 
    /* Title and Subtitle */
    .title-text { font-size: 2.5rem; font-weight: 800; color: #93c5fd; } /* Blue 300 */
    .sub-text { color: #94a3b8; margin-bottom: 2rem; } /* Slate 400 */
    
    /* Streamlit chat message specific styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
    }
    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }
    .answer-title {
        color: #fbbf24;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .answer-body {
        color: #e2e8f0;
        line-height: 1.55;
        font-size: 1rem;
    }
    /* Source chunk styling */
    .chunk-block {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 0.5rem;
    }
    .chunk-header {
        color: #93c5fd;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .chunk-body {
        color: #cbd5e1;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    .chunk-body mark {
        background-color: #fde68a;
        color: #111827;
        padding: 0.1rem 0.15rem;
        border-radius: 4px;
    }
    
    /* Make chat message text bright and readable */
    .stChatMessage p {
        margin-bottom: 0.75rem;
        color: #f1f5f9 !important; /* Bright slate-100 */
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .stChatMessage strong {
        color: #e2e8f0 !important; /* Even brighter for bold text */
        font-weight: 600;
    }
    
    .stChatMessage em {
        color: #cbd5e1 !important; /* Bright for italic text */
    }
    
    .stChatMessage code {
        background-color: #1e293b !important;
        color: #60a5fa !important; /* Bright blue for code */
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9em;
    }

    /* Code blocks and preformatted text */
    .stChatMessage pre, .answer-body pre {
        background-color: #0b1220 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.85rem;
        overflow: auto;
        white-space: pre;
        word-break: normal;
        line-height: 1.45;
        font-size: 0.95em;
        font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        max-height: 520px;
    }

    .stChatMessage code, .answer-body code {
        background-color: #0b1220 !important;
        color: #e2e8f0 !important;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 0.25rem 0.4rem;
        font-size: 0.95em;
        font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    
    /* Lists styling */
    .stChatMessage ul, .stChatMessage ol {
        margin-left: 1.5rem;
        padding-left: 0;
        color: #f1f5f9 !important;
    }
    
    .stChatMessage li {
        margin-bottom: 0.5rem;
        color: #f1f5f9 !important;
    }
    
    /* Headings */
    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3 {
        color: #93c5fd !important; /* Bright blue */
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .stChatMessage h3 {
        font-size: 1.3rem;
        color: #60a5fa !important; /* Even brighter blue */
    }
    
    /* Blockquotes */
    .stChatMessage blockquote {
        border-left: 3px solid #60a5fa;
        padding-left: 1rem;
        margin-left: 0;
        color: #cbd5e1 !important;
        font-style: italic;
    }
    
    /* Links */
    .stChatMessage a {
        color: #60a5fa !important;
        text-decoration: underline;
    }
    
    .stChatMessage a:hover {
        color: #93c5fd !important;
    }
    
    /* Caption styling for sources */
    .stChatMessage .stCaption {
        color: #94a3b8 !important; /* Slate-400 for captions */
    }
    
    /* Expander styling for sources */
    .streamlit-expanderHeader {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        color: #f1f5f9 !important;
    }
    
    /* Source citation styling */
    .source-citation {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #f97316;
    }
    
    .source-citation strong {
        color: #fbbf24 !important; /* Bright amber for source names */
    }
    
    .source-citation code {
        color: #60a5fa !important;
        background-color: #0f172a;
    }
    
    /* Streamlit widgets styling for consistency */
    /* FIX 1: Add caret-color to ensure cursor visibility in the input box */
    .stTextInput>div>div>input {
        background-color: #1e293b !important;
        border: 1px solid #334155;
        color: #f8fafc;
        border-radius: 8px;
        caret-color: #f97316 !important; /* Bright amber color for contrast */
    }
    /* File uploader styling */
    .stFileUploader>div>div {
        background-color: #1e293b !important;
        border: 1px solid #334155;
        color: #f8fafc;
        border-radius: 8px;
    }
    
    /* FIX 3: Custom Button Styling for dark theme */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    /* Secondary Button Styling (Clear Chat/Cleanup) to make it dark/muted */
    .stButton:has(button[kind="secondary"]) button {
        background-color: #334155 !important; /* Dark Slate background */
        border-color: #475569 !important;
        color: #e2e8f0 !important;
    }
    
    /* Ensure primary buttons stand out */
    .stButton:has(button[kind="primary"]) button {
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- API Functions ---
def get_backend_url() -> str:
    return st.session_state["backend_url"].rstrip("/")


def set_backend_url(url: str) -> None:
    st.session_state["backend_url"] = url.rstrip("/")


def api_health() -> bool:
    try:
        resp = requests.get(f"{get_backend_url()}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def api_status() -> dict:
    """Check the processing status and fetch loaded documents."""
    try:
        resp = requests.get(f"{get_backend_url()}/status", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e), "loaded_documents": []}


def api_load(files, url: str, progress_placeholder) -> dict:
    """Load documents and poll for completion."""
    payload = {}
    if url:
        payload["url"] = url.strip()
    
    files_param = []
    if files:
        for f in files:
            files_param.append(("files", (f.name, f, f.type)))

    # 1. Start the load process
    try:
        resp = requests.post(
            f"{get_backend_url()}/load",
            files=files_param,
            data=payload,
            timeout=10, 
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e))
        raise Exception(f"Server Error: {error_detail}")
    except Exception as e:
        raise Exception(f"Connection Error: {e}")

    # 2. If processing started, poll for completion
    if result.get("status") == "processing":
        max_wait = 300 
        start_time = time.time()
        poll_interval = 2 
        
        while time.time() - start_time < max_wait:
            status_resp = api_status()
            status = status_resp.get("status", "processing")
            
            if status == "ready":
                st.session_state["rag_status"] = status_resp
                progress_placeholder.success("✅ Documents indexed successfully!")
                time.sleep(0.5) 
                progress_placeholder.empty() 
                return status_resp
            elif status == "error":
                error_msg = status_resp.get("error", "Unknown error occurred during processing.")
                progress_placeholder.error(f"❌ Indexing Error: {error_msg}")
                st.session_state["rag_status"] = status_resp
                raise Exception(error_msg)
            else:
                elapsed = int(time.time() - start_time)
                # Use a progress bar for better visualization
                progress_placeholder.progress(min(elapsed / max_wait, 1.0), text=f"⏳ Processing documents... ({elapsed}s elapsed)")
                time.sleep(poll_interval)
        
        # Timeout
        progress_placeholder.error("⏱️ Processing timed out after 5 minutes.")
        raise Exception("Processing timed out. Please check server logs.")
    
    # If already ready from the server side (fingerprint reuse)
    return api_status()


# MODIFIED: Now returns the structured dictionary from the backend
def api_query(query: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{get_backend_url()}/query",
        json={"query": query},
        timeout=180,
    )
    resp.raise_for_status()
    # Return the full structured response: {"answer": str, "sources": list}
    return resp.json() 

def api_cleanup() -> dict:
    resp = requests.post(f"{get_backend_url()}/cleanup", timeout=10)
    resp.raise_for_status()
    return resp.json()

def api_cleanup_selected(paths: list[str]) -> dict:
    resp = requests.post(
        f"{get_backend_url()}/cleanup_selected",
        json={"paths": paths},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


# --- Streamlit UI Components ---

def initial_status_check():
    """Checks the RAG status on page load."""
    if st.session_state["rag_status"]["status"] in ["idle", "processing", "error"]:
        status_resp = api_status()
        st.session_state["rag_status"] = status_resp
        
        # Display persistent status message if not ready
        if status_resp["status"] == "processing":
            st.toast("Indexing in progress...", icon="⏳")
        elif status_resp["status"] == "error":
            st.toast(f"Error: {status_resp.get('error', 'Check server logs.')}", icon="❌")
        elif status_resp["status"] == "ready":
            st.toast("RAG Pipeline is ready for queries.", icon="✅")


def sidebar():
    st.sidebar.title("⚙️ RAG System Controls")
    
    # 1. Connection Settings
    with st.sidebar.expander("🔗 Backend Connection", expanded=False):
        backend_url = st.text_input("FastAPI URL", get_backend_url(), key="backend_url_input")
        set_backend_url(backend_url)

        health_ok = api_health()
        status_icon = "🟢" if health_ok else "🔴"
        st.markdown(f"**{status_icon} Backend Status:** {'Reachable' if health_ok else 'Offline'}")

    st.sidebar.markdown("---")
    
    # 2. Document Indexing
    st.sidebar.header("📚 Load Documents")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF/DOCX/TXT/MD", 
        type=["pdf", "docx", "doc", "txt", "md"], 
        accept_multiple_files=True
    )
    url_input = st.sidebar.text_input("or Provide a URL to index", placeholder="e.g., https://example.com/report.pdf")

    if st.sidebar.button("Index Documents", type="primary", use_container_width=True):
        if not uploaded_files and not url_input.strip():
            st.sidebar.warning("Upload at least one file or provide a URL.")
            return
        
        # Use a status component for real-time indexing feedback
        with st.sidebar.status("Starting document indexing...", expanded=True) as status_box:
            try:
                status_box.update(label="Saving uploads and sending to backend...", state="running")
                resp = api_load(uploaded_files, url_input, status_box)
                status_box.update(label="✅ Documents Indexed Successfully!", state="complete")
                
                # Update main session state
                st.session_state["rag_status"] = resp
                st.toast("Indexing complete! You can now ask questions.", icon="🎉")
                st.rerun() 
            except Exception as e:
                st.sidebar.error(f"Failed to index documents: {e}")
                status_box.update(label=f"❌ Indexing Failed: {e}", state="error")
    
    st.sidebar.markdown("---")

    # 3. Document Management & Status
    status_data = st.session_state["rag_status"]
    current_status = status_data["status"]
    loaded_docs = status_data.get("loaded_documents", [])

    st.sidebar.header("Current Context")
    
    if current_status == "ready" and loaded_docs:
        st.sidebar.success(f"Context Ready: {len(loaded_docs)} source(s) indexed.")
        with st.sidebar.expander("Indexed Sources", expanded=True):
            for doc in loaded_docs:
                doc_name = doc.get("name", doc.get("path", "Unknown Source"))
                st.caption(f"📄 {doc_name}")
        
        # Context cleanup controls
        doc_options = {doc.get("name", doc.get("path", "Unknown Source")): doc.get("path") for doc in loaded_docs}
        selected_labels = st.sidebar.multiselect(
            "Clear specific contexts",
            options=list(doc_options.keys()),
            placeholder="Select one or more sources",
        )
        clear_scope = st.sidebar.radio(
            "Clear scope",
            ["Selected", "All"],
            horizontal=True,
            key="clear_scope_choice",
        )

        if st.sidebar.button("Clear Selected Context", type="secondary", use_container_width=True):
            try:
                if clear_scope == "All":
                    cleanup_resp = api_cleanup()
                    st.session_state["rag_status"] = {"status": "idle", "loaded_documents": []}
                    st.session_state["messages"] = []
                    st.sidebar.success(cleanup_resp.get("message", "All context cleared."))
                else:
                    if not selected_labels:
                        st.sidebar.warning("Select at least one source to clear.")
                    else:
                        selected_paths = [doc_options[label] for label in selected_labels if doc_options.get(label)]
                        resp = api_cleanup_selected(selected_paths)
                        # Refresh status after partial cleanup
                        status_resp = api_status()
                        st.session_state["rag_status"] = status_resp
                        # If nothing remains, clear chat history
                        if not status_resp.get("loaded_documents"):
                            st.session_state["messages"] = []
                        st.sidebar.success(resp.get("message", "Selected context cleared."))
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Context clear failed: {e}")
    elif current_status == "processing":
        st.sidebar.info("Indexing in Progress...")
    elif current_status == "error":
        st.sidebar.error(f"Error State: {status_data.get('error', 'Check server logs.')}")
    else:
        st.sidebar.warning("No documents loaded yet. Please index sources to begin.")

    # 4. Cleanup Button
    if st.sidebar.button("Clear Indexed Documents (Cleanup)", type="secondary", use_container_width=True):
        try:
            with st.spinner("Clearing RAG pipeline and files..."):
                cleanup_resp = api_cleanup()
                st.session_state["rag_status"] = {"status": "idle", "loaded_documents": []}
                st.session_state["messages"] = [] # Clear chat history on cleanup
                st.success(cleanup_resp["message"])
                st.rerun() 
        except Exception as e:
            st.sidebar.error(f"Cleanup failed: {e}")


def chat_area():
    st.markdown('<div class="title-text">Multi-Doc RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Upload documents or URLs in the sidebar, then ask grounded questions about your data.</div>', unsafe_allow_html=True)

    rag_ready = st.session_state["rag_status"]["status"] == "ready"

    # Manage chat input state safely before rendering the widget
    if "chat_query" not in st.session_state:
        st.session_state["chat_query"] = ""
    if st.session_state.get("clear_chat_query"):
        st.session_state["chat_query"] = ""
        st.session_state["clear_chat_query"] = False

    # Display Chat History using st.chat_message
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            # Assistant message content is the structured result: {"answer": str, "sources": list}
            response_content = message["content"]
            with st.chat_message("assistant", avatar="🤖"):
                # 1. Display the structured answer with enhanced styling
                answer_text = response_content.get("answer", "")
                if answer_text:
                    st.markdown(
                        """
                        <div class="answer-card">
                            <div class="answer-title">Answer</div>
                            <div class="answer-body">
                        """,
                        unsafe_allow_html=True,
                    )
                    # Render markdown so code blocks stay intact and readable
                    st.markdown(answer_text, unsafe_allow_html=False)
                    st.markdown(
                        """
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                # 2. Display Sources in an Expander with better styling and highlighted chunks
                sources = response_content.get("sources", [])
                if sources:
                    with st.expander(f"📚 Sources Used ({len(sources)} citation{'s' if len(sources) > 1 else ''})", expanded=False):
                        for i, source in enumerate(sources):
                            source_name = source.get('name', 'Unknown Source')
                            page_info = source.get('page_info', 'N/A')
                            source_path = source.get('path', '')
                            chunk_html = source.get('highlighted_chunk') or ""
                            raw_chunk = source.get('chunk', '')
                            # If no highlighted version, escape raw chunk
                            if not chunk_html and raw_chunk:
                                chunk_html = html.escape(raw_chunk)
                            # Limit excessively long chunks for UI
                            if chunk_html and len(chunk_html) > 3000:
                                chunk_html = chunk_html[:3000] + "..."
                            
                            st.markdown(
                                f"""
                                <div class="source-citation" style="margin-bottom: 1rem; padding: 1rem; background-color: #1e293b; border-left: 3px solid #f97316; border-radius: 6px;">
                                    <p style="margin-bottom: 0.5rem; color: #fbbf24; font-weight: 600; font-size: 1.05rem;">
                                        {i+1}. {source_name}
                                    </p>
                                    <p style="margin-bottom: 0.5rem; color: #94a3b8; font-size: 0.9rem;">
                                        📍 Location: <span style="color: #cbd5e1;">{page_info}</span>
                                    </p>
                                    <p style="margin-bottom: 0.25rem; color: #94a3b8; font-size: 0.85rem; font-family: monospace;">
                                        📄 Path: <code style="color: #60a5fa; background-color: #0f172a; padding: 0.2rem 0.4rem; border-radius: 4px;">{os.path.basename(source_path) if source_path else 'N/A'}</code>
                                    </p>
                                    {f'<div class="chunk-block"><div class="chunk-header">Context Chunk</div><div class="chunk-body">{chunk_html}</div></div>' if chunk_html else ''}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.markdown(
                        '<p style="color: #94a3b8; font-size: 0.9rem; font-style: italic;">No specific sources were cited for this response.</p>',
                        unsafe_allow_html=True
                    )


    # Chat Input
    with st.form("chat_form", clear_on_submit=False):
        query = st.text_input(
            "Ask a question about the indexed documents", 
            placeholder="e.g., Summarize the key findings from the uploaded report.", 
            disabled=not rag_ready,
            label_visibility="collapsed",
            key="chat_query",
        )
        col1, col2 = st.columns([1, 6])
        with col1:
            submitted = st.form_submit_button("Send", type="primary", disabled=not rag_ready, use_container_width=True)
        # Removed st.button("Clear Chat", ...) from inside the form

    # Clear Chat Button (moved outside the form)
    # Using st.columns here to maintain alignment from the original structure
    col_clear, _ = st.columns([1, 6])
    with col_clear:
        if st.button("Clear Chat", type="secondary", use_container_width=True): # Use use_container_width to match form_submit_button size
             st.session_state["messages"] = []
             st.rerun() 


    if submitted and query.strip():
        # Add user message to history
        st.session_state["messages"].append({"role": "user", "content": query.strip()})
        
        # Get assistant response immediately (before rerun)
        with st.spinner("Retrieving and generating answer..."):
            try:
                # api_query returns the full structured dictionary
                result = api_query(query.strip())
                
                # Validate result structure
                if not isinstance(result, dict):
                    raise ValueError(f"Unexpected response format: {type(result)}")
                
                if "answer" not in result:
                    raise ValueError("Response missing 'answer' field")
                
                # Ensure sources is a list
                if "sources" not in result:
                    result["sources"] = []
                
                # Add the full structured result to history
                st.session_state["messages"].append({"role": "assistant", "content": result})
                
            except requests.exceptions.HTTPError as e:
                error_detail = "Unknown error"
                try:
                    if e.response is not None:
                        error_detail = e.response.json().get("detail", str(e))
                except:
                    error_detail = str(e)
                
                error_message = f"Query failed: {error_detail}"
                st.error(error_message)
                # Remove the user message if query failed
                if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
                    st.session_state["messages"].pop()
                    
            except Exception as e:
                error_message = f"Query failed: {str(e)}"
                st.error(error_message)
                # Remove the user message if query failed
                if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
                    st.session_state["messages"].pop()
        
        # Rerun to display messages
        # Flag to clear the input box on next render to avoid Streamlit key conflicts
        st.session_state["clear_chat_query"] = True
        st.rerun()


def main():
    sidebar()
    initial_status_check() 
    chat_area()


if __name__ == "__main__":
    main()