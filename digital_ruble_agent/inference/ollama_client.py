"""OLLAMA client for local inference."""
from typing import Optional, Dict, Any, List
import requests


class OllamaClient:
    """Клиент для локальной модели OLLAMA с системным промптом."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        self.base_url = base_url
        self.model = model
        self._check_connection()
    
    def _check_connection(self) -> None:
        """Проверить, доступен ли OLLAMA."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if resp.status_code != 200:
                raise ConnectionError(f"OLLAMA недоступен: {resp.status_code}")
            print(f"[INIT] OLLAMA доступен ({self.base_url})")
        except Exception as e:
            print(f"[WARN] Ошибка подключения к OLLAMA: {e}")
    
    @property
    def available(self) -> bool:
        """Доступна ли модель?"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Сгенерировать ответ от модели.
        
        Args:
            prompt: Запрос к модели.
            system_prompt: Системный промпт. При None — по умолчанию.
        
        Returns:
            Текст ответа модели.
        """
        # Системный промпт по умолчанию
        default_system = """Ты — агент цифрового рубля. Твоя задача — отвечать на вопросы,
используя предоставленную базу знаний. Отвечай кратко и по делу.
Если информации недостаточно, так и скажи."""
        
        system = system_prompt or default_system
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,  # ??? Ollama: ?? ????
            "stream": False
        }
        
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            resp_data = resp.json()
            return resp_data.get("response", "")
        except Exception as e:
            print(f"[ERROR] OLLAMA request failed: {e}")
            return f"Ошибка генерации: {e}"
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Chat-компил (messages с role: system/user/assistant)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            print(f"[ERROR] OLLAMA chat failed: {e}")
            return f"Ошибка чата: {e}"
