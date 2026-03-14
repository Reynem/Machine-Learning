# Восстановление пунктуации в казахском языке

Задача токенизированной классификации (Token Classification) для автоматической расстановки знаков препинания.

## Методы обучения

### Модель

- **XLM-RoBERTa Base** — многоязычная трансформер-модель
- Альтернатива: `kz-transformers/kaz-roberta-kw-base` (казахская RoBERTa)

### Архитектура

- Token Classification с классификационной головой на 4 класса:
  - `O` — без знака препинания
  - `COMMA` — запятая
  - `PERIOD` — точка
  - `QUESTION` — вопросительный знак

### Техники оптимизации

| Метод | Описание |
| **Weighted Cross-Entropy** | Веса классов обратно пропорциональны частоте (для борьбы с дисбалансом) |
| **Mixed Precision (FP16)** | AMP через `GradScaler` для ускорения обучения на GPU |
| **AdamW** | Оптимизатор с weight decay (0.01) |
| **Linear Schedule with Warmup** | Планировщик learning rate с 10% warmup |
| **Gradient Clipping** | Ограничение градиентов (max norm = 1.0) |

### Инференс

- **Aggregation** — агрегация предсказаний с приоритетом non-O классов и confidence

## Запуск

Просто запустите ноутбук

## Структура данных

- `train_example.csv` — обучающая выборка (id, input_text, labels)
- `test.csv` — тестовые данные
- `outputs/submission.csv` — результат инференса
