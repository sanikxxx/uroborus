"""Skill registry for Digital Ruble Agent."""
from typing import Dict, Type, Optional
from .base import BaseSkill


class SkillRegistry:
    """Реестр зарегистрированных навыков."""
    
    def __init__(self):
        self._skills: Dict[str, Type[BaseSkill]] = {}
    
    def register(self, skill_class: Type[BaseSkill]) -> None:
        """Зарегистрировать класс навыка."""
        name = skill_class.name
        if name in self._skills:
            raise ValueError(f"Skill '{name}' уже зарегистрирован")

        self._skills[name] = skill_class
    
    def get(self, name: str) -> Optional[Type[BaseSkill]]:
        """Получить класс навыка по имени."""
        return self._skills.get(name)
    
    def list_all(self) -> Dict[str, str]:
        """Получить список всех навыков с описаниями."""
        return {name: skill.description for name, skill in self._skills.items()}
    
    @property
    def skills(self) -> Dict[str, Type[BaseSkill]]:
        """Доступ к словарю навыков."""
        return self._skills


# Глобальный реестр
registry = SkillRegistry()
