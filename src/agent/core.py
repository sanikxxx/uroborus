import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List
import numpy as np
from scipy.spatial.distance import cosine

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
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Dict[str, np.ndarray] = {}

    def load_docs(self):
        for file in self.docs_dir.glob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            self.documents.append({"path": str(file), "content": content})
            self.embeddings[str(file)] = np.random.rand(100)

    def select(self, query: str) -> Dict[str, Any]:
        query_vec = np.random.rand(100)
        best_doc = None
        best_sim = float("inf")
        for path, emb in self.embeddings.items():
            sim = cosine(query_vec, emb)
            if sim < best_sim:
                best_sim = sim
                best_doc = path
        return next(d for d in self.documents if d["path"] == best_doc)

class DigitalRubleAgent:
    def __init__(self):
        self.registry = SkillRegistry()
        self.rag = RAGSelector("docs")
        self.rag.load_docs()

    def register_skill(self, name: str, func: Callable):
        self.registry.register(name, func)

    def execute(self, query: str) -> str:
        doc = self.rag.select(query)
        return f"Выбран документ: {doc['path']}. Запрос: {query}"
