#!/usr/bin/env python3
"""Облегчённые эмбеддинги через OLLAMA (локально)."""
from typing import List
import requests


class OllamaEmbeddings:
    """Эмбеддинги через OLLAMA API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url
        self.model = model
        self._check_connection()
    
    def _check_connection(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if resp.status_code != 200:
                raise ConnectionError(f"OLLAMA недоступен: {resp.status_code}")
            print(f"[INIT] OLLAMA Emb доступен модель: {self.model}")
        except Exception as e:
            print(f"[WARN] Ошибка подключения OLLAMA Embs: {e}")
    
    @property
    def available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False
    
    def encode(self, text: str) -> List[float]:
        payload = {
            "model": self.model,
            "prompt": text
        }
        try:
            resp = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding", [])
        except Exception as e:
            print(f"[ERROR] OLLAMA embeddings failed: {e}")
            return []
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            emb = self.encode(text)
            embeddings.append(emb if emb else [0.0] * 768)
        return embeddings


def load_encoder_model(model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
    return OllamaEmbeddings(base_url=base_url, model=model_name)
