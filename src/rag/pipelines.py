import os
import sys
from typing import Dict, List, Sequence

from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from src.exception import MyException
from src.ingestion.extractor import DocumentExtractor
from src.ingestion.loaders import DocumentLoader
from src.logger import logging
from src.preprocessing.clean_normalize import DocumentNormalizationAndCleaning
from src.preprocessing.chunking import DocumentChunker
from src.rag import prompts
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RerankMMRRetriever
from src.utils.main_utils import num_tokens_from_string, read_yaml_file
from src.vectorstore.faiss_store import FaissVectorStore


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    1) Load + clean + chunk documents
    2) Build vector store
    3) Retrieve -> rerank -> MMR
    4) Choose Stuff vs Map-Reduce prompting and query LLM
    """

    def __init__(self, config_dir: str = "configs"):
        self.config = self._load_configs(config_dir)

        gen_cfg = self.config.get("generation", {})
        self.llm = ChatOllama(
            model=gen_cfg.get("llm_model", "llama3"),
            temperature=gen_cfg.get("temperature", 0.2),
        )

        retr_cfg = self.config.get("retrieval", {})
        self.reranker = CrossEncoderReranker(
            model_name=retr_cfg.get(
                "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )

        self.vector_store = None
        self.retriever = None

    # ----------------------------
    # Configuration loading
    # ----------------------------
    def _load_configs(self, config_dir: str) -> Dict:
        try:
            configs = {}
            for name in ["ingestion", "chunking", "retrieval", "generation", "pipeline"]:
                path = os.path.join(config_dir, f"{name}.yaml")
                if os.path.exists(path):
                    configs.update(read_yaml_file(path) or {})
                else:
                    logging.warning("Config file missing: %s", path)
            return configs
        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Data preparation
    # ----------------------------
    def prepare_vector_store(self) -> None:
        """Load documents, clean, chunk, and build FAISS vector store."""
        try:
            docs_cfg = self.config.get("documents", [])
            if not docs_cfg:
                raise MyException("No documents configured for processing.", sys)
            
            chunk_cfg = self.config.get("chunking", {})
            logging.info("Starting vector store preparation with %d document(s)", len(docs_cfg))

            loader = DocumentLoader()
            extractor = DocumentExtractor()
            cleaner = DocumentNormalizationAndCleaning()
            chunker = DocumentChunker()

            all_chunks: list = []
            for idx, doc_info in enumerate(docs_cfg, 1):
                if not doc_info.get("enabled", True):
                    logging.info("Skipping disabled document: %s", doc_info.get("path", "unknown"))
                    continue
                path = doc_info["path"]
                logging.info("[%d/%d] Processing document: %s", idx, len(docs_cfg), path)

                try:
                    loaded = loader.load_document(path)
                    logging.debug("Document loaded successfully: %s", path)
                    
                    extracted = extractor.extract_document_info(loaded, path)
                    logging.debug("Document extracted successfully: %s", path)
                    
                    cleaned = cleaner.initialize_document_normalizer(extracted)
                    logging.debug("Document cleaned successfully: %s", path)
                    
                    chunks = chunker.chunk_document(
                        cleaned,
                        target_chunk_size=chunk_cfg.get("target_chunk_size", 500),
                        chunk_overlap=chunk_cfg.get("chunk_overlap", 100),
                    )
                    logging.info("Generated %d chunks from document: %s", len(chunks), path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logging.error("Failed to process document %s: %s", path, e)
                    raise MyException(f"Error processing document {path}: {e}", sys)

            if not all_chunks:
                raise MyException("No chunks generated; check document config and ensure documents are enabled.", sys)

            logging.info("Creating vector store with %d total chunks...", len(all_chunks))
            self.vector_store = FaissVectorStore().create_vector_store(all_chunks)
            self.retriever = RerankMMRRetriever(self.vector_store, self.reranker)
            logging.info("Vector store prepared successfully with %d chunks", len(all_chunks))
        except Exception as e:
            logging.exception("Failed to prepare vector store: %s", e)
            raise MyException(e, sys)

    # ----------------------------
    # Retrieval + Routing
    # ----------------------------
    def retrieve(self, query: str) -> List[Document]:
        if self.retriever is None:
            raise MyException("Retriever not initialized. Call prepare_vector_store().", sys)

        logging.info("Retrieving documents for query: %s", query[:100] if len(query) > 100 else query)
        
        retr_cfg = self.config.get("retrieval", {})
        retrieve_kwargs = {
            "lambda_mult": retr_cfg.get("lambda_mult", 0.5),
        }

        for key in ["initial_k", "rerank_k", "mmr_k", "initial_pct", "rerank_pct", "mmr_pct"]:
            value = retr_cfg.get(key)
            if value is not None:
                retrieve_kwargs[key] = value

        documents = self.retriever.retrieve(query, **retrieve_kwargs)
        logging.info("Retrieved %d documents for query", len(documents))
        return documents

    def answer(self, query: str) -> str:
        """Retrieve context and generate an answer with grounding + citations."""
        try:
            logging.info("Generating answer for query: %s", query[:100] if len(query) > 100 else query)
            
            documents = self.retrieve(query)
            if not documents:
                logging.warning("No documents retrieved for query: %s", query)
                return "I don't have enough information to answer this question based on the provided documents."

            gen_cfg = self.config.get("generation", {})
            max_docs = gen_cfg.get("max_context_documents", 8)
            selected_docs = documents[:max_docs]
            logging.info("Using %d documents for answer generation (max: %d)", len(selected_docs), max_docs)

            total_tokens = sum(
                num_tokens_from_string(doc.page_content) for doc in selected_docs
            )
            logging.info("Total context tokens: %d (limit: %d)", total_tokens, gen_cfg.get("stuff_context_token_limit", 2000))
            
            if total_tokens <= gen_cfg.get("stuff_context_token_limit", 2000):
                answer = self._answer_with_stuff(query, selected_docs)
            else:
                answer = self._answer_with_map_reduce(query, selected_docs)
            
            logging.info("Answer generated successfully (length: %d chars)", len(answer))
            return answer
        except Exception as e:
            logging.exception("Failed to generate answer: %s", e)
            raise MyException(e, sys)

    # ----------------------------
    # Prompting strategies
    # ----------------------------
    def _answer_with_stuff(self, query: str, docs: Sequence[Document]) -> str:
        context_str = self._build_context(docs)
        prompt_tpl = prompts.build_stuff_prompt()
        prompt = prompt_tpl.format(
            system_prompt=prompts.SYSTEM_PROMPT, context=context_str, question=query
        )
        logging.info("Using Stuff strategy with %d docs", len(docs))
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    def _answer_with_map_reduce(self, query: str, docs: Sequence[Document]) -> str:
        map_tpl = prompts.build_map_prompt()
        map_outputs = []
        logging.info("Using Map-Reduce strategy with %d docs", len(docs))
        for doc in docs:
            ctx = self._format_chunk_with_citation(doc)
            map_prompt = map_tpl.format(
                system_prompt=prompts.SYSTEM_PROMPT, context=ctx, question=query
            )
            res = self.llm.invoke(map_prompt)
            map_outputs.append(getattr(res, "content", str(res)))

        reduce_tpl = prompts.build_reduce_prompt()
        reduce_prompt = reduce_tpl.format(
            system_prompt=prompts.SYSTEM_PROMPT,
            map_summaries="\n\n".join(map_outputs),
            question=query,
        )
        reduced = self.llm.invoke(reduce_prompt)
        return getattr(reduced, "content", str(reduced))

    # ----------------------------
    # Helpers
    # ----------------------------
    def _build_context(self, docs: Sequence[Document]) -> str:
        """Concatenate documents with inline citations."""
        parts = []
        for doc in docs:
            parts.append(self._format_chunk_with_citation(doc))
        return "\n\n".join(parts)

    def _format_chunk_with_citation(self, doc: Document) -> str:
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "N/A")
        doc_type = meta.get("doc_type", "doc")
        citation = f"(source: {doc_type}: {source}, page {page})"
        return f"{doc.page_content}\n{citation}"

