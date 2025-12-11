import os
from pathlib import Path

project_name = "src"

list_of_files = [

    f"{project_name}/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/settings.py",  
    f"{project_name}/ingestion/__init__.py",
    f"{project_name}/ingestion/loaders.py",
    f"{project_name}/ingestion/detectors.py",
    f"{project_name}/preprocessing/__init__.py",
    f"{project_name}/preprocessing/normalize.py",
    f"{project_name}/preprocessing/chunking.py",
    f"{project_name}/embedding/__init__.py",
    f"{project_name}/embedding/embedder.py",
    f"{project_name}/vectorstore/__init__.py",
    f"{project_name}/vectorstore/faiss_store.py",
    f"{project_name}/retrieval/__init__.py",
    f"{project_name}/retrieval/retriever.py",
    f"{project_name}/retrieval/reranker.py",
    f"{project_name}/rag/__init__.py",
    f"{project_name}/rag/prompts.py",
    f"{project_name}/rag/pipelines.py",
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/app.py",
    f"{project_name}/ui/__init__.py",
    f"{project_name}/ui/streamlit_app.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    "requirements.txt",
    "Dockerfile",
    "pyproject.toml",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")