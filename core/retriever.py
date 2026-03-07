from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        # Fallback if both fail, although classic seems to work here
        from langchain_community.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
import numpy as np
import torch
import os

class AdvancedRetriever:
    def __init__(self, documents, embedding_model_name="BAAI/bge-small-en-v1.5", reranker_model_name="BAAI/bge-reranker-base"):
        """
        Hybrid Retriever: BM25 (Keyword) + Vector (Semantic)
        Optimized for both CPU and GPU (e.g., RTX 3090).
        """
        # 0. Hardware Detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            # Optimization for CPU: use multiple threads
            torch.set_num_threads(os.cpu_count() or 4)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_root = os.path.join(project_root, "models")
        embedding_model_name = self._resolve_model_source(embedding_model_name, models_root)
        reranker_model_name = self._resolve_model_source(reranker_model_name, models_root)
        
        # 1. Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': self.device}
        )
        
        # 2. Vector Store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        # 3. BM25 Retriever (Keyword matching)
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 10
        
        # 4. Ensemble (Hybrid Search)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.3, 0.7]
        )
        
        self.reranker = None
        self.reranker_available = False
        try:
            self.reranker = CrossEncoder(reranker_model_name, device=self.device)
            self.reranker_available = True
        except Exception:
            self.reranker = None
            self.reranker_available = False

    @staticmethod
    def _resolve_model_source(model_name, models_root):
        if os.path.isdir(model_name):
            return model_name

        direct_path = os.path.join(models_root, model_name)
        if os.path.isdir(direct_path):
            return direct_path

        base_name = model_name.split("/")[-1]
        base_path = os.path.join(models_root, base_name)
        if os.path.isdir(base_path):
            return base_path

        return model_name

    def retrieve_with_rerank(self, query, top_k=5):
        """
        Retrieves docs using hybrid search and then reranks them for precision.
        """
        # Phase 1: Hybrid Retrieval
        initial_docs = self.ensemble_retriever.invoke(query)
        
        if not initial_docs:
            return []
            
        if not self.reranker_available:
            return initial_docs[:top_k]

        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]
