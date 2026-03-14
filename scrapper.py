import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

from generator import process_text, split_into_chunks, is_valid_chunk

def scrape_tengri_article(url: str) -> str:
    """Возвращает сырой текст статьи с пунктуацией."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # Основной текст статьи
    paragraphs = soup.find_all('p')
    text = ' '.join(p.get_text() for p in paragraphs)
    return text.strip()
    
def texts_to_dataframe(texts: list[str], id_offset: int = 0) -> pd.DataFrame:
    """
    Принимает список сырых текстов с пунктуацией.
    Возвращает DataFrame в том же формате что train_full.csv.
    """
    rows = []
    
    for text in texts:
        pairs = process_text(text)
        if not pairs:
            continue
            
        chunks = split_into_chunks(pairs)
        
        for chunk in chunks:
            words = [w for w, _ in chunk]
            labels = [l for _, l in chunk]
            
            if not is_valid_chunk(words, labels):
                continue
                
            rows.append({
                "id": f"kzp_tengri_{id_offset + len(rows):06d}",
                "input_text": ' '.join(words),
                "labels": ' '.join(labels),
            })
    
    return pd.DataFrame(rows)

urls = [
    "https://kaz.tengrinews.kz/kazakhstan_news/taksister-jolaushyilardyi-saktandyiruyi-kerek-pe-politsiya-372154/"
]

texts = []
for url in urls:
    try:
        text = scrape_tengri_article(url)
        if text:
            texts.append(text)
        time.sleep(0.5)  # чтобы не забанили
    except Exception as e:
        print(f"Ошибка {url}: {e}")

print(f"Собрано статей: {len(texts)}")

# 2. Конвертируем
df_tengri = texts_to_dataframe(texts)
print(f"Чанков из tengri: {len(df_tengri)}")
print(f"QUESTION в tengri: {df_tengri['labels'].str.contains('QUESTION').sum()}")

# 3. Объединяем с основным датасетом
df_main = pd.read_csv('train_full.csv')
df_combined = pd.concat([df_main, df_tengri], ignore_index=True)

# Переиндексируем id чтобы не было дублей
df_combined['id'] = [f"kzp_train_{i:06d}" for i in range(len(df_combined))]
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

df_combined.to_csv('train_combined.csv', index=False)
print(f"Итого строк: {len(df_combined)}")