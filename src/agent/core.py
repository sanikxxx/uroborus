"""
Agent Core - Ядро агента

Состоит из двух основных функций:
1. Чтение нужной инструкции (RAG-выборка по релевантным документам)
2. Выполнение зарегистрированных функций (скиллов)
"""

import os
import json
import logging
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger('AgentCore')


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
        self.client = chromadb.PersistentClient(path=str(self.logs_path / "rag_cache"))
        self.collection = self.client.get_or_create_collection(
            name="docs_index",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Инициализация ядра агента завершена. Документация: {self.docs_path}")
    
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
    
    def _index_documents(self):
        """Индексация документов для RAG-поиска"""
        documents = self._load_documents()
        
        if not documents:
            logger.warning("Нет документов для индексации")
            return
        
        # Получаем текущее количество элементов в коллекции
        current_count = self.collection.count()
        
        # Индексируем только если коллекция пуста (или добавляем новые)
        if current_count == 0:
            for doc in documents:
                self.collection.add(
                    ids=[doc["id"]],
                    documents=[doc["content"]],
                    metadatas=[{"source": doc["source"]}]
                )
            logger.info(f"Индексация документов завершена: {len(documents)} документов")
        else:
            logger.info(f"Документы уже проиндексированы ({current_count} элементов)")
    
    def _search_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу (RAG)
        
        Args:
            query: Пользовательский запрос
            top_k: Количество результатов
            
        Returns:
            Список релевантных документов
        """
        # Индексируем документы, если еще не индексированы
        if self.collection.count() == 0:
            self._index_documents()
        
        # Поиск в векторной БД
        results = self.collection.query(
            query_embeddings=[query],  # В будущем заменить на эмбеддинги
            n_results=top_k
        )
        
        # Формируем результаты
        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            docs.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "score": results["distances"][0][i],  # Чем меньше, тем лучше
                "source": results["metadatas"][0][i].get("source", "unknown")
            })
        
        return docs
    
    def _get_instruction(self, query: str) -> str:
        """
        Получение инструкции по релевантным документам
        
        Args:
            query: Пользовательский запрос
            
        Returns:
            Сформированная инструкция на основе релевантных документов
        """
        # Поиск релевантных документов
        relevant_docs = self._search_relevant_docs(query, top_k=2)
        
        if not relevant_docs:
            return "Инструкция не найдена. Попробуйте переформулировать запрос или добавить документацию."
        
        # Формирование инструкции на основе найденных документов
        instruction = "Найдены следующие релевантные документы:\n\n"
        for doc in relevant_docs:
            instruction += f"## {doc['source']}\n{doc['content'][:500]}...\n\n"
        
        return instruction
    
    def _execute_tools(self, query: str, context: str) -> str:
        """
        Выполнение зарегистрированных функций (скиллов)
        
        Args:
            query: Пользовательский запрос
            context: Контекст из документов
            
        Returns:
            Результат выполнения
        """
        # Получаем список доступных функций
        tools = self.tool_registry.list_tools()
        
        if not tools:
            return "Нет зарегистрированных функций для выполнения."
        
        # В MVP просто возвращаем список доступных функций
        # В будущем реализуем автоматический выбор и вызов функций
        tool_list = "Доступные функции:\n"
        for tool in tools:
            tool_list += f"- {tool['name']}: {tool['description']}\n"
        
        return tool_list
    
    def process_query(self, query: str) -> str:
        """
        Обработка запроса пользователя
        
        1. Чтение инструкции по RAG-поиску
        2. Выполнение зарегистрированных функций
        
        Args:
            query: Пользовательский запрос
            
        Returns:
            Ответ агента
        """
        logger.info(f"Обработка запроса: '{query}'")
        
        # 1. Получение инструкции
        instruction = self._get_instruction(query)
        logger.debug(f"Получена инструкция: {instruction[:100]}...")
        
        # 2. Выполнение инструментов
        tool_result = self._execute_tools(query, instruction)
        logger.debug(f"Результат выполнения инструментов: {tool_result[:100]}...")
        
        # Формирование финального ответа
        response = f"""Я обработал ваш запрос.

### Инструкция:
{instruction}

### Доступные функции:
{tool_result}

В будущей версии агент будет автоматически выбирать и выполнять функции.
Сейчас вы можете использовать интерфейс для взаимодействия с функциями."""
        
        return response


if __name__ == '__main__':
    # Тестирование ядра агента
    from tools.registry import ToolRegistry
    
    tool_registry = ToolRegistry()
    core = AgentCore(
        tool_registry=tool_registry,
        docs_path='/opt/ouroboros/digital_ruble_agent/docs',
        logs_path='/opt/ouroboros/digital_ruble_agent/logs'
    )
    
    # Тестовый запрос
    result = core.process_query("Что такое цифровой рубль?")
    print(result)
