"""
Tool Registry - Реестр зарегистрированных функций (скиллов)

Каждая функция имеет:
- name: название
- description: описание
- params: ожидаемые параметры
- execute: функция-обработчик
"""

import logging
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger('ToolRegistry')


class Tool:
    def __init__(self, name: str, description: str, execute: Callable):
        self.name = name
        self.description = description
        self.execute = execute


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._logger = logging.getLogger('ToolRegistry')
        self._logger.info("Инициализация реестра инструментов")
        
        # Регистрируем базовые инструменты для MVP
        self._register_base_tools()
    
    def _register_base_tools(self):
        """Регистрация базовых инструментов для MVP"""
        # Примеры базовых инструментов
        self.register(
            name="read_doc",
            description="Чтение документа по названию или ID",
            execute=self._read_doc
        )
        
        self.register(
            name="search_docs",
            description="Поиск в документации по запросу",
            execute=self._search_docs
        )
        
        self.register(
            name="get_api_logs",
            description="Получение логов через API",
            execute=self._get_api_logs
        )
        
        self._logger.info(f"Зарегистрировано {len(self._tools)} базовых инструментов")
    
    def register(self, name: str, description: str, execute: Callable):
        """
        Регистрация нового инструмента
        
        Args:
            name: Название инструмента
            description: Описание функциональности
            execute: Функция-обработчик
        """
        self._tools[name] = Tool(name, description, execute)
        self._logger.info(f"Зарегистрирован инструмент: {name}")
    
    def unregister(self, name: str):
        """Удаление инструмента"""
        if name in self._tools:
            del self._tools[name]
            self._logger.info(f"Удален инструмент: {name}")
        else:
            self._logger.warning(f"Попытка удалить несуществующий инструмент: {name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Получение инструмента по названию"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """Список всех зарегистрированных инструментов"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]
    
    def execute(self, name: str, **kwargs) -> Any:
        """Выполнение инструмента по названию"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Инструмент '{name}' не найден")
        
        self._logger.info(f"Выполнение инструмента: {name}")
        return tool.execute(**kwargs)
    
    # Базовые обработчики
    def _read_doc(self, doc_name: str = None, doc_id: str = None) -> str:
        """Чтение документа"""
        return f"Чтение документа: {doc_name or doc_id}"
    
    def _search_docs(self, query: str, top_k: int = 5) -> str:
        """Поиск в документации"""
        return f"Поиск в документации по запросу: '{query}' (top_k={top_k})"
    
    def _get_api_logs(self, endpoint: str = "/api/logs", limit: int = 100) -> Dict[str, Any]:
        """Получение логов через API"""
        return {
            "endpoint": endpoint,
            "limit": limit,
            "logs": [],  # В будущем будет реальный запрос
            "status": "ready"
        }


# Инициализация реестра при импорте
registry = ToolRegistry()
