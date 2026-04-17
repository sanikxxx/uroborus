"""Digital Ruble Agent — skill-based agent with RAG."""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from skills.registry import registry, SkillRegistry
from skills.base import BaseSkill
from rag.searcher import SemanticSearcher
from inference.ollama_client import OllamaClient


class DigitalRubleAgent:
    """Агент цифрового рубля.
    
    Две основные функции:
    1. read_instruction(instruction_id) — загрузить инструкцию по ID
    2. execute_skill(skill_name, **kwargs) — выполнить зарегистрированный навык
    """
    
    def __init__(
        self,
        instructions_path: str = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "mistral",
        encoder_model: str = "sentence-t5-xl"
    ):
        # Загрузка инструкций
        if instructions_path is None:
            instructions_path = Path(__file__).parent / "data" / "instructions.json"
        
        self.instructions_path = str(instructions_path)
        self.instructions = self._load_instructions()
        
        # RAG
        self.searcher = SemanticSearcher(self.instructions_path, encoder_model)
        
        # OLLAMA
        self.ollama = OllamaClient(ollama_url, ollama_model)
        
        # Skills
        self.skill_registry = registry
    
    def _load_instructions(self) -> Dict[str, Dict]:
        """Загрузить инструкции из JSON-файла."""
        path = Path(self.instructions_path)
        if not path.exists():
            print(f"[WARN] Файл инструкций не найден: {path}")
            return {}
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Индексация по ID
        by_id = {inst.get('id', ''): inst for inst in data.get('instructions', [])}
        return by_id
    
    def read_instruction(self, instruction_id: str) -> Optional[Dict[str, Any]]:
        """Загрузить инструкцию по ID.
        
        Returns:
            Dict с полями: id, topic, content. None, если не найдена.
        """
        return self.instructions.get(instruction_id)
    
    def execute_skill(self, skill_name: str, **kwargs) -> Dict[str, Any]:
        """Выполнить зарегистрированный навык.
        
        Args:
            skill_name: Имя навыка из реестра
            **kwargs: Аргументы для навыка
        
        Returns:
            Dict с полями: success (bool), result (Any), error (Optional[str])
        """
        skill_class = self.skill_registry.get(skill_name)
        if skill_class is None:
            return {
                "success": False,
                "result": None,
                "error": f"Навык '{skill_name}' не найден"
            }
        
        try:
            skill = skill_class()
            return skill.execute(**kwargs)
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Ошибка выполнения навыка: {e}"
            }
    
    def query_rag(self, user_question: str, top_k: int = 2) -> Dict[str, Any]:
        """Задать вопрос через RAG — найти похожие документы, получить ответ от модели.
        
        Returns:
            Dict с:
            - question: исходный вопрос
            - documents: список найденных документов с score
            - answer: ответ модели
        """
        results = self.searcher.search(user_question, top_k=top_k)
        
        context_docs = [doc for doc, score in results]
        context_text = self._format_context(context_docs)
        
        if not context_docs:
            return {
                "question": user_question,
                "documents": [],
                "answer": "По базе знаний ничего не найдено. Попробуйте переформулировать вопрос."
            }
        
        prompt = f"""Контекст из базы знаний:

{context_text}

Вопрос: {user_question}

Ответ:"""
        
        answer = self.ollama.generate(prompt) if self.ollama.available else (
            "Mistral недоступен. Ответ не сгенерирован."
        )
        
        return {
            "question": user_question,
            "documents": [
                {"doc": doc, "score": score} for doc, score in results
            ],
            "answer": answer
        }
    
    def _format_context(self, documents: List[Dict[str, str]]) -> str:
        """Форматировать документы для контекста."""
        parts = []
        for idx, doc in enumerate(documents, 1):
            parts.append(f"Документ {idx} (ID: {doc.get('id', 'N/A')}, Тема: {doc.get('topic', 'N/A')}):")
            parts.append(doc.get('content', ''))
            parts.append('')
        return '\n'.join(parts)
    
    def list_skills(self) -> Dict[str, str]:
        """Получить список всех зарегистрированных навыков."""
        return self.skill_registry.list_all()


# Пример навыка
from skills.base import BaseSkill

class GetExchangeRateSkill(BaseSkill):
    """Получить курс цифрового рубля (пример навыка)."""
    
    name = "get_exchange_rate"
    description = "Получить текущий обменный курс цифрового рубля"
    
    def execute(self, currency: str = "USD") -> Dict[str, Any]:
        """Курс для зарпла-валюты (заглушка)."""
        # TODO: интеграция с ЦБР
        return {
            "success": True,
            "result": {"currency": currency, "rate": 1.0},
            "error": None
        }


# Регистрация при импорте
registry.register(GetExchangeRateSkill)
