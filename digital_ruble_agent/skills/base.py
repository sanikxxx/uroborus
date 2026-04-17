"""Base skill class for Digital Ruble Agent."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseSkill(ABC):
    """Базовый класс для навыка агента."""
    
    name: str = "base_skill"  # Имя навыка для вызова
    description: str = "Базовый навык без описания"  # Описание для RAG
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Выполнить навык с переданными аргументами.
        
        Returns:
            Dict с результатом выполнения.
            Ключ 'success': bool — успех или ошибка
            Ключ 'result': Any — результат выполнения
            Ключ 'error': Optional[str] — сообщение об ошибке
        """
        pass
