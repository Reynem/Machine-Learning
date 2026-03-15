import pandas as pd
from datasets import load_dataset
from generator import process_text, split_into_chunks, is_valid_chunk

def build_kazqad_dataframe() -> pd.DataFrame:
    ds = load_dataset("issai/kazqad", "kazqad", token="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")
    
    rows = []
    
    for split in ['train', 'validation', 'test']:
        for item in ds[split]:
            
            # Вопросы → гарантированный QUESTION
            question = item['question'].strip()
            pairs = process_text(question)
            if pairs:
                words = [w for w, _ in pairs]
                labels = [l for _, l in pairs]
                if 4 <= len(words) <= 60 and labels[-1] == 'QUESTION':
                    rows.append({
                        "id": f"kzp_kazqad_q_{len(rows):06d}",
                        "input_text": ' '.join(words),
                        "labels": ' '.join(labels),
                    })
            
            # Контексты → PERIOD и COMMA
            context = item['context'].strip()
            pairs = process_text(context)
            if not pairs:
                continue
            chunks = split_into_chunks(pairs)
            for chunk in chunks:
                words = [w for w, _ in chunk]
                labels = [l for _, l in chunk]
                if is_valid_chunk(words, labels):
                    rows.append({
                        "id": f"kzp_kazqad_c_{len(rows):06d}",
                        "input_text": ' '.join(words),
                        "labels": ' '.join(labels),
                    })
    
    return pd.DataFrame(rows)


def combine_datasets(main_path: str, output_path: str) -> pd.DataFrame:
    print("Загрузка train_full.csv...")
    df_main = pd.read_csv(main_path)
    print(f"  train_full: {len(df_main):,} строк")
    
    print("\nЗагрузка KazQAD...")
    df_kazqad = build_kazqad_dataframe()
    print(f"  KazQAD: {len(df_kazqad):,} строк")
    
    # Объединяем
    df_combined = pd.concat([df_main, df_kazqad], ignore_index=True)
    
    # Переиндексируем
    df_combined['id'] = [f"kzp_train_{i:06d}" for i in range(len(df_combined))]
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_combined.to_csv(output_path, index=False)
    
    # Статистика
    print(f"\n{'='*50}")
    print(f"Итого строк: {len(df_combined):,}")
    
    all_labels = ' '.join(df_combined['labels']).split()
    from collections import Counter
    counts = Counter(all_labels)
    total = sum(counts.values())
    print(f"\nРаспределение классов:")
    for label in ['O', 'COMMA', 'PERIOD', 'QUESTION']:
        cnt = counts.get(label, 0)
        print(f"  {label:<10} {cnt:>8,}  ({cnt/total*100:.1f}%)")
    
    print(f"\nСохранено в: {output_path}")
    return df_combined


if __name__ == "__main__":
    combine_datasets(
        main_path="train_full.csv",
        output_path="train_combined.csv",
    )