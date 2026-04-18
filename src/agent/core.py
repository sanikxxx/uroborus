import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import hashlib
import os
import numpy as np

# Следующие импорты могут потребовать установки: pip install sentence-transformers faiss-cpu
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("WARNING: sentence-transformers или faiss-cpu не установлены. RAG не будет работать.")


class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self.skills[name] = func

    def get(self, name: str) -> Callable:
        return self.skills[name]

    def list(self) -> List[str]:
        return list(self.skills.keys())


class RAGSelector:
    def __init__(self, docs_dir: str = "docs/raw", cache_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.cache_dir = Path(cache_dir)
        self.documents: List[Dict[str, Any]] = []
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self._load_model()
        self._load_or_build_index()

    def _load_model(self):
        if RAG_AVAILABLE:
            try:
                self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            except Exception as e:
                print(f"WARNING: Не удалось загрузить модель эмбеддингов: {e}")
                self.model = None
        else:
            print("WARNING: sentence-transformers не установлен. RAG не будет работать.")

    def _doc_text(self, doc: Dict[str, Any]) -> str:
        """Формирует строку для эмбеддинга из всех полей документа."""
        return f"{doc.get('title', '')}. {doc.get('content', '')}. {doc.get('category', '')}. {' '.join(doc.get('tags', []))}"

    def _compute_hash(self, content: str) -> str:
        """Вычисляет хеш контента для определения изменений."""
        return hashlib.md5(content.encode()).hexdigest()

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Загружает документы из markdown файлов в docs/raw/."""
        documents = []
        for file_path in self.docs_dir.glob("*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Парсинг простого markdown в структуру
            doc = {
                "path": str(file_path),
                "content": content,
                "title": "",
                "category": "",
                "tags": []
            }
            
            # Извлечение заголовка (первый # заголовок)
            lines = content.split("\n")
            for line in lines:
                if line.startswith("# "):
                    doc["title"] = line[2:].strip()
                    break
            
            documents.append(doc)
        return documents

    def _build_index(self, documents: List[Dict[str, Any]]):
        """Строит Faiss индекс для быстрого поиска."""
        if not self.model:
            return
        
        texts = [self._doc_text(doc) for doc in documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Нормализация для cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Создание индекса
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
        self.index.add(embeddings)
        
        self.documents = documents

    def _load_or_build_index(self):
        """Загружает индекс из кэша или строит заново."""
        index_file = self.cache_dir / "rag_index.bin"
        
        # Проверка необходимости перестройки
        if index_file.exists() and RAG_AVAILABLE:
            try:
                self.index = faiss.read_index(str(index_file))
                print("RAG: индекс загружен из кэша.")
                return
            except Exception as e:
                print(f"RAG: ошибка загрузки кэша: {e}. Перестройка индекса.")

        # Построение индекса
        self.documents = self._load_knowledge_base()
        self._build_index(self.documents)
        
        # Сохранение в кэш
        if self.index and RAG_AVAILABLE:
            faiss.write_index(self.index, str(index_file))
            print(f"RAG: индекс построен и сохранен ({len(self.documents)} документов).")

    def select(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Выполняет RAG-поиск по запросу."""
        if not self.model or not self.index:
            # Fallback: случайный выбор при отсутствии RAG
            return self.documents[:k] if self.documents else []
        
        # Encode запроса
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        # Поиск
        scores, indices = self.index.search(query_vec, k)
        
        # Формирование результатов
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(scores[0][i]),
                    "rank": i + 1
                })
        
        return results


class DigitalRubleAgent:
    def __init__(self):
        self.registry = SkillRegistry()
        self.rag = RAGSelector("docs/raw", "docs")
        # Временно заглушка для RAG — пока не установлены пакеты
        # self.rag = RAGSelector("docs/raw", "docs")

    def register_skill(self, name: str, func: Callable):
        self.registry.register(name, func)

    def execute(self, query: str) -> str:
        docs = self.rag.select(query, k=3)
        if not docs:
            return f"Не найдено подходящих документов. Запрос: {query}"
        
        # Используем самый релевантный документ
        best_doc = docs[0]["document"]
        return f"Выбран документ: {best_doc['path']} (score={docs[0]['score']:.3f}). Запрос: {query}\n\nКонтекст:\n{best_doc['content'][:500]}..."
