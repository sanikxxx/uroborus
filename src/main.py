"""
Digital Ruble Agent - MVP
Основной файл агента

Архитектура MVP:
- agent/ — основная логика агента
- tools/ — зарегистрированные функции (скиллы)
- docs/ — инструкции и документация
- logs/ — сбор логов
- ui/ — веб-интерфейс (в будущем)
"""

import os
import sys
import json
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/ouroboros/digital_ruble_agent/logs/agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('DigitalRubleAgent')

# Импорты
from agent.core import AgentCore
from tools.registry import ToolRegistry


def main():
    """Основная точка входа в агента"""
    logger.info("Запуск Digital Ruble Agent MVP")
    
    # Загрузка конфигурации из .env
    from dotenv import load_dotenv
    load_dotenv()
    
    # Инициализация реестра инструментов
    tool_registry = ToolRegistry()
    
    # Инициализация ядра агента
    agent_core = AgentCore(
        tool_registry=tool_registry,
        docs_path='/opt/ouroboros/digital_ruble_agent/docs',
        logs_path='/opt/ouroboros/digital_ruble_agent/logs'
    )
    
    logger.info("Digital Ruble Agent MVP готов к работе")
    
    # Основной цикл работы
    try:
        while True:
            # Получение входного запроса (в будущем через API или UI)
            user_query = input("\nВведите ваш запрос (или 'exit' для выхода): ").strip()
            
            if user_query.lower() in ['exit', 'выход', 'quit']:
                logger.info("Завершение работы агента")
                break
            
            if not user_query:
                continue
            
            # Обработка запроса
            result = agent_core.process_query(user_query)
            
            # Вывод ответа
            print(f"\nОтвет агента:\n{result}")
            
            # Логирование
            logger.info(f"Запрос: '{user_query}' -> Ответ: {result[:100]}...")
            
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, завершение работы")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)


if __name__ == '__main__':
    main()
