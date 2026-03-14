"""
Генерация обучающего датасета для восстановления пунктуации в казахском языке.
Источник: kz-transformers/multidomain-kazakh-dataset (HuggingFace, streaming)

Использование:
    python 03_build_dataset.py --limit 50000 --output train_full.csv
    python 03_build_dataset.py --limit 200000 --output train_full.csv --hf_token hf_xxx
"""

import re
import os
import argparse
import pandas as pd
from datasets import load_dataset

# ─────────────────────────────────────────────
# ПАРСИНГ ТОКЕНА И ПУНКТУАЦИИ
# ─────────────────────────────────────────────

# Знаки препинания которые мы восстанавливаем
PUNCT_MAP = {
    '?': 'QUESTION',
    '.': 'PERIOD',
    ',': 'COMMA',
}

# Символы которые просто удаляем (не несут смысловой нагрузки)
STRIP_CHARS = '«»""„"\'()[]{}*•·…'


def parse_token(token: str):
    """
    Принимает один токен с возможной пунктуацией.
    Возвращает (чистое_слово, метка) или None если токен нужно пропустить.

    Примеры:
        'жақсы,'  -> ('жақсы', 'COMMA')
        'бар.'    -> ('бар', 'PERIOD')
        'келді?'  -> ('келді', 'QUESTION')
        'сөз'     -> ('сөз', 'O')
        '2009'    -> None  (число)
        '№'       -> None  (мусор)
    """
    # Убираем обрамляющие кавычки и скобки
    token = token.strip(STRIP_CHARS)

    if not token:
        return None

    # Определяем метку по последнему символу
    label = 'O'
    if token[-1] in PUNCT_MAP:
        label = PUNCT_MAP[token[-1]]
        token = token[:-1]  # убираем знак

    # Ещё раз чистим после удаления знака
    token = token.strip(STRIP_CHARS)

    if not token:
        return None

    # Приводим к нижнему регистру
    token = token.lower()

    # Фильтруем мусорные токены
    if _is_junk_token(token):
        return None

    return token, label


def _is_junk_token(token: str) -> bool:
    if re.fullmatch(r'[\d\-–—]+', token):
        return True
    # УБРАТЬ: одиночные буквы — в казахском это валидные частицы
    # if re.fullmatch(r'[а-яәіңғүұқөһa-z]', token):
    #     return True
    if re.fullmatch(r'[№@#$%^&+=<>|\\~`]+', token):
        return True
    if re.search(r'https?|www\.', token):
        return True
    return False


# ─────────────────────────────────────────────
# ОБРАБОТКА ОДНОГО ТЕКСТА
# ─────────────────────────────────────────────

def process_text(text: str):
    """
    Принимает сырой текст с пунктуацией.
    Возвращает список (слово, метка) пар.
    """
    pairs = []
    for token in text.split():
        result = parse_token(token)
        if result is not None:
            pairs.append(result)
    return pairs


# ─────────────────────────────────────────────
# РАЗБИВКА НА ЧАНКИ (2-3 предложения)
# ─────────────────────────────────────────────

def split_into_chunks(pairs, min_words=8, max_words=60):
    """
    Разбивает список пар на чанки по границам предложений (PERIOD/QUESTION).
    Группирует по 2-3 предложения, соблюдая min/max длину.
    """
    chunks = []
    current = []
    sentence_count = 0

    for word, label in pairs:
        current.append((word, label))

        if label in ('PERIOD', 'QUESTION'):
            sentence_count += 1

            # Накопили 2-3 предложения или достигли максимума
            if sentence_count >= 2 or len(current) >= max_words:
                if len(current) >= min_words:
                    chunks.append(current)
                current = []
                sentence_count = 0

    # Остаток — добавляем если достаточно длинный
    if len(current) >= min_words and sentence_count >= 1:
        chunks.append(current)

    return chunks


# ─────────────────────────────────────────────
# ФИЛЬТРАЦИЯ ЧАНКОВ
# ─────────────────────────────────────────────

def is_valid_chunk(words, labels):
    # УБРАТЬ это условие:
    # if labels[-1] not in ('PERIOD', 'QUESTION'):
    #     return False
    
    # Оставить только минимальные проверки:
    if len(words) < 8 or len(words) > 60:
        return False
    punct_ratio = sum(1 for l in labels if l != 'O') / len(words)
    if punct_ratio < 0.03 or punct_ratio > 0.40:
        return False
    return True


# ─────────────────────────────────────────────
# ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────

def build_dataset(limit: int, output_path: str, hf_token: str = None):
    print(f"Загрузка датасета (streaming, limit={limit:,})...")

    kwargs = {"streaming": True, "split": "train"}
    if hf_token:
        kwargs["token"] = hf_token

    ds = load_dataset("kz-transformers/multidomain-kazakh-dataset", **kwargs)

    rows = []
    texts_read = 0
    texts_skipped = 0

    for i, row in enumerate(ds):
        if texts_read >= limit:
            break

        text = row.get('text') or row.get('content') or ''
        text = text.strip()
        if not text:
            continue

        # Обрабатываем текст
        pairs = process_text(text)
        if not pairs:
            continue

        # Разбиваем на чанки
        chunks = split_into_chunks(pairs)

        for chunk in chunks:
            words = [w for w, _ in chunk]
            labels = [l for _, l in chunk]

            if not is_valid_chunk(words, labels):
                texts_skipped += 1
                continue

            rows.append({
                "id": f"kzp_train_{len(rows):06d}",
                "input_text": ' '.join(words),
                "labels": ' '.join(labels),
            })

        texts_read += 1

        if texts_read % 5000 == 0:
            print(f"  Прочитано текстов: {texts_read:,} | "
                  f"Сохранено чанков: {len(rows):,} | "
                  f"Отфильтровано: {texts_skipped:,}")

    # Сохраняем
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"Готово!")
    print(f"  Прочитано текстов:  {texts_read:,}")
    print(f"  Итого чанков:       {len(df):,}")
    print(f"  Отфильтровано:      {texts_skipped:,}")
    print(f"  Сохранено в:        {output_path}")

    # Статистика классов
    all_labels = ' '.join(df['labels']).split()
    from collections import Counter
    counts = Counter(all_labels)
    total = sum(counts.values())
    print(f"\nРаспределение классов:")
    for label in ['O', 'COMMA', 'PERIOD', 'QUESTION']:
        cnt = counts.get(label, 0)
        print(f"  {label:<10} {cnt:>8,}  ({cnt/total*100:.1f}%)")

    return df


# ─────────────────────────────────────────────
# ВАЛИДАЦИЯ ФИНАЛЬНОГО ФАЙЛА
# ─────────────────────────────────────────────

def validate_dataset(path: str):
    """Проверяет что количество слов == количество меток в каждой строке."""
    df = pd.read_csv(path)
    errors = 0
    for _, row in df.iterrows():
        n_words = len(row['input_text'].split())
        n_labels = len(row['labels'].split())
        if n_words != n_labels:
            print(f"  ОШИБКА {row['id']}: слов={n_words}, меток={n_labels}")
            errors += 1
    if errors == 0:
        print(f"Валидация пройдена: все {len(df):,} строк корректны.")
    else:
        print(f"Найдено {errors} ошибок!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50_000,
                        help="Сколько текстов прочитать из датасета")
    parser.add_argument("--output", default="train_full.csv",
                        help="Путь к выходному CSV файлу")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace токен (необязательно)")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    df = build_dataset(
        limit=args.limit,
        output_path=args.output,
        hf_token=args.hf_token,
    )

    print("\nВалидация датасета...")
    validate_dataset(args.output)