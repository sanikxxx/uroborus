#!/usr/bin/env python3
"""Семантический поиск через OLLAMA эмбеддинги."""
from typing import List, Dict, Tuple
import json
from pathlib import Path
import math
from .embeddings import load_encoder_model


class SemanticSearcher:
    """Семантический поиск документов."""
    
    def __init__(self, instructions_path: str, encoder_model: str = "nomic-embed-text"):
        self.instructions_path = instructions_path
        self.encoder_model_name = encoder_model
        self.encoder = load_encoder_model(encoder_model)
        self.documents: List[Dict[str, str]] = []
        self.embeddings: List[List[float]] = []
        self._load_documents()
    
    def _load_documents(self) -> None:
        """Загрузить документы из JSON."""
        path = Path(self.instructions_path)
        if not path.exists():
            print(f"[WARN] Файл инструкций не найден: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data.get('instructions', [])
        
        print(f"[INIT] Загружено {len(self.documents)} документов")
    
    def _compute_embeddings(self) -> None:
        """Вычислить эмбеддинги (lazy)."""
        if not self.documents:
            return
        
        if not self.encoder.available:
            print("[WARN] OLLAMA недоступен, семантический поиск не работает")
            return
        
        texts = [self._format_doc(d) for d in self.documents]
        print(f"[INIT] Вычисление эмбеддингов для {len(texts)} текстов...")
        self.embeddings = self.encoder.encode_batch(texts)
        print(f"[INIT] Готово: {len(self.embeddings)} эмбеддингов")
    
    def _format_doc(self, doc: Dict[str, str]) -> str:
        """Форматировать документ для эмбеддинга."""
        parts = [doc.get('id', ''), doc.get('topic', ''), doc.get('content', '')]
        return '\n'.join(filter(None, parts))
    
    def search(self, query: str, top_k: int = 2) -> List[Tuple[Dict[str, str], float]]:
        """Найти похожие документы."""
        if not self.documents:
            return []
        
        # Ленивая инициализация
        if not self.embeddings:
            self._compute_embeddings()
        
        if not self.embeddings or not self.encoder.available:
            # Fallback: простой поиск по ключевым словам
            return self._keyword_search(query, top_k)
        
        # Эмбеддинг запроса
        query_emb = self.encoder.encode(query)
        if not query_emb:
            return []
        
        # Косинусная похожесть
        similarities = self._cosine_similarities(query_emb, self.embeddings)
        
        # Сортировка и топ-K
        indexed = list(zip(range(len(similarities)), similarities))
        indexed.sort(key=lambda x: x[1], reverse=True)
        
        results = [(self.documents[i], score) for i, score in indexed[:top_k]]
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Dict[str, str], float]]:
        """Fallback: поиск по ключевым словам."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            text = self._format_doc(doc).lower()
            # Простая метрика: сколько слов из запроса есть в документе
            query_words = set(query_lower.split())
            doc_words = set(text.split())
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words) if query_words else 0
            results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _cosine_similarities(self, vec1: List[float], vecs: List[List[float]]) -> List[float]:
        """Косинусная похожесть vec1 с каждым vecs[i]."""
        norm1 = math.sqrt(sum(x*x for x in vec1))
        if norm1 == 0:
            return [0.0] * len(vecs)
        
        return [
            self._cosine_similarity(vec1, vec2, norm1)
            for vec2 in vecs
        ]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float], norm1: float) -> float:
        """Косинусная похожесть двух векторов."""
        if len(vec1) != len(vec2) or len(vec1) == 0:
            return 0.0
        
        dot = sum(a*b for a, b in zip(vec1, vec2))
        norm2 = math.sqrt(sum(x*x for x in vec2))
        
        if norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
