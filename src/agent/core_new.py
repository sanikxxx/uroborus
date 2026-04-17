"""
Agent Core - Ядро агента

Состоит из двух основных функций:
1. Чтение нужной инструкции (RAG-выборка по релевантным документам)
2. Выполнение зарегистрированных функций (скиллов)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger('AgentCore')

# Попытка импорта зависимостей с обработкой ошибок
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"chromadb не найден: {e}. RAG-функции будут недоступны")
    CHROMADB_AVAILABLE = False


class AgentCore:
    def __init__(self, tool_registry, docs_path: str, logs_path: str):
        """
        Инициализация ядра агента
        
        Args:
            tool_registry: Реестр зарегистрированных функций
            docs_path: Путь к документации
            logs_path: Путь к логам
        """
        self.tool_registry = tool_registry
        self.docs_path = Path(docs_path)
        self.logs_path = Path(logs_path)
        
        # Инициализация RAG-системы для поиска по документации
        self.client = None
        self.collection = None
        
        if CHROMADB_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=str(self.logs_path / "rag_cache"))
                self.collection = self.client.get_or_create_collection(
                    name="docs_index",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Инициализация RAG-системы завершена. Документация: {self.docs_path}")
            except Exception as e:
                logger.error(f"Ошибка инициализации chromadb: {e}")
                logger.warning("RAG-функции будут недоступны")
        else:
            logger.warning("chromadb недоступен — RAG-функции отключены")
        
        logger.info(f"Ядро агента инициализировано. Документация: {self.docs_path}")
    
    def _load_documents(self) -> List[Dict[str, Any]]:
        """Загрузка всех документов из папки docs"""
        documents = []
        
        if not self.docs_path.exists():
            logger.warning(f"Папка с документами не найдена: {self.docs_path}")
            return documents
        
        for file_path in self.docs_path.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        "id": file_path.stem,
                        "content": content,
                        "source": str(file_path)
                    })
            except Exception as e:
                logger.error(f"Ошибка загрузки документа {file_path}: {e}")
        
        logger.info(f"Загружено {len(documents)} документов")
        return documents
