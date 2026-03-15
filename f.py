import pandas as pd
import numpy as np
from collections import Counter
import argparse


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


df = load_data("train_combined.csv")

from collections import Counter

all_labels = ' '.join(df['labels']).split()
counts = Counter(all_labels)
total = sum(counts.values())

for label in ['O', 'COMMA', 'PERIOD', 'QUESTION']:
    cnt = counts.get(label, 0)
    print(f"{label:<10} {cnt:>8,}  ({cnt/total*100:.1f}%)")


all_labels = ' '.join(df['labels']).split()
from collections import Counter
print(Counter(all_labels))


def analyze(df: pd.DataFrame):
    print("=" * 60)
    print("РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
    print("=" * 60)
    print(f"\nВсего предложений: {len(df)}")
    
    all_labels = []
    word_counts = []
    for _, row in df.iterrows():
        labels = row["labels"].split()
        all_labels.extend(labels)
        word_counts.append(len(labels))

    label_counts = Counter(all_labels)
    total = sum(label_counts.values())

    print(f"\nВсего токенов (слов): {total}")
    print(f"\nРаспределение классов:")
    print(f"{'Класс':<12} {'Кол-во':>10} {'Доля':>10}")
    print("-" * 35)
    for label in ["O", "COMMA", "PERIOD", "QUESTION"]:
        cnt = label_counts.get(label, 0)
        print(f"{label:<12} {cnt:>10,} {cnt/total*100:>9.2f}%")

    print(f"\nДлина предложений (в словах):")
    wc = np.array(word_counts)
    print(f"  min={wc.min()}, max={wc.max()}, mean={wc.mean():.1f}, median={np.median(wc):.1f}")

    # Рекомендуемые веса для Weighted Cross-Entropy
    print(f"\nРекомендуемые веса классов (обратно пропорционально частоте):")
    label_order = ["O", "COMMA", "PERIOD", "QUESTION"]
    counts = np.array([label_counts.get(l, 1) for l in label_order], dtype=float)
    weights = total / (len(label_order) * counts)
    weights = weights / weights.min()  # нормируем
    for l, w in zip(label_order, weights):
        print(f"  {l:<12}: {w:.4f}")

    print("\n" + "=" * 60)


analyze(df)


import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# КОНСТАНТЫ
LABEL2ID = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
IGNORE_INDEX = -100


# ВОСПРОИЗВОДИМОСТЬ
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ЗАГРУЗКА И ПАРСИНГ ДАННЫХ
def load_csv_data(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Читает CSV с колонками: id, input_text, labels.
    Возвращает: (список слов по предложениям, список меток по предложениям).
    """
    df = pd.read_csv(path)
    sentences_words = []
    sentences_labels = []
    for _, row in df.iterrows():
        words = row["input_text"].split()
        labels = row["labels"].split()
        # Защита: если длины не совпадают — обрезаем
        min_len = min(len(words), len(labels))
        sentences_words.append(words[:min_len])
        sentences_labels.append(labels[:min_len])
    return sentences_words, sentences_labels


# DATASET
class PunctuationDataset(Dataset):
    """
    Токенизирует слова → подслова (subwords).
    Метка присваивается ТОЛЬКО первому подслову каждого слова.
    Остальным подсловам → IGNORE_INDEX (-100).
    """

    def __init__(
        self,
        sentences_words: List[List[str]],
        sentences_labels: Optional[List[List[str]]],
        tokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for idx, words in enumerate(sentences_words):
            labels = sentences_labels[idx] if sentences_labels else None
            self.samples.append((words, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, word_labels = self.samples[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        if word_labels is not None:
            label_ids = []
            word_ids = encoding.word_ids(batch_index=0)
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    # [CLS], [SEP], [PAD]
                    label_ids.append(IGNORE_INDEX)
                elif word_id != prev_word_id:
                    # Первое подслово → берём метку
                    lbl = word_labels[word_id] if word_id < len(word_labels) else "O"
                    label_ids.append(LABEL2ID.get(lbl, 0))
                else:
                    # Последующие подслова → игнорируем
                    label_ids.append(IGNORE_INDEX)
                prev_word_id = word_id

            labels_tensor = torch.tensor(label_ids, dtype=torch.long)
        else:
            labels_tensor = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
            "words": words,  # для восстановления при инференсе
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    result = {"input_ids": input_ids, "attention_mask": attention_mask}
    if batch[0]["labels"] is not None:
        result["labels"] = torch.stack([b["labels"] for b in batch])
    result["words"] = [b["words"] for b in batch]
    return result


# МЕТРИКИ
def compute_macro_f1(
    true_ids: List[int], pred_ids: List[int]
) -> float:
    """Macro F1 по классам COMMA, PERIOD, QUESTION (без O)."""
    labels_for_metric = [LABEL2ID["COMMA"], LABEL2ID["PERIOD"], LABEL2ID["QUESTION"]]
    f1 = f1_score(true_ids, pred_ids, labels=labels_for_metric, average="macro", zero_division=0)
    return f1


# WEIGHTED CROSS-ENTROPY
def compute_class_weights(
    sentences_labels: List[List[str]],
    device: torch.device,
    max_weight: float = 30.0,
) -> torch.Tensor:
    counts = np.zeros(len(LABEL2ID), dtype=float)
    for labels in sentences_labels:
        for lbl in labels:
            if lbl in LABEL2ID:
                counts[LABEL2ID[lbl]] += 1

    # Логарифмическое сглаживание
    weights = 1.0 + np.log(1 + counts.max() / np.maximum(counts, 1))
    
    # Ограничение сверху
    weights = np.clip(weights, a_min=1.0, a_max=max_weight)
    
    print(f"Веса классов: { {k: round(weights[v], 3) for k, v in LABEL2ID.items()} }")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ОБУЧЕНИЕ С ОПТИМИЗАЦИЕЙ (FP16)

from torch.amp import GradScaler, autocast


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device_type)
    print(f"Устройство: {device}")

    if args.data.endswith(".csv"):
        all_words, all_labels = load_csv_data(args.data)
    else:
        return

    train_w, val_w, train_l, val_l = train_test_split(
        all_words, all_labels, test_size=0.2, random_state=args.seed
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, 
        num_labels=len(LABEL2ID), 
        id2label=ID2LABEL, 
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    ).to(device)

    train_ds = PunctuationDataset(train_w, train_l, tokenizer, args.max_length)
    val_ds = PunctuationDataset(val_w, val_l, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    class_weights = compute_class_weights(train_l, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scaler = GradScaler()
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    best_f1 = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    
    # Таблица для логов эпох
    print("\n" + "="*80)
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val F1':^10} | {'Best F1':^10} | {'Saved':^6}")
    print("="*80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
        
        # Progress bar с более подробным описанием
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(args.device_type):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits.view(-1, len(LABEL2ID)), labels.view(-1)) / accum_steps

            scaler.scale(loss).backward()
            
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            
            # Обновление progress bar с текущим loss
            if (step + 1) % 10 == 0:  # каждые 10 шагов
                pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        
        # Валидация
        f1, report = evaluate(model, val_loader, device, criterion)
        
        # Проверка на лучший результат
        saved = ""
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            saved = "✓"
        
        # Лог эпохи
        print(f"Epoch {epoch:2d} | {avg_train_loss:^12.4f} | {f1:^10.4f} | {best_f1:^10.4f} | {saved:^6}")
        
        # Детальный report по классам для последней эпохи или лучшей
        if f1 == best_f1:
            print("\nClassification Report:")
            print(report)
            print("-"*80)
    
    print("="*80)
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Best model saved to: {best_model_path}")


# ВАЛИДАЦИЯ
def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if batch.get("labels") is not None else None

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1)

            if labels is not None:
                if criterion:
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    total_loss += loss.item()

                # Собираем предсказания (только не-IGNORE)
                mask = labels.view(-1) != IGNORE_INDEX
                all_true.extend(labels.view(-1)[mask].cpu().tolist())
                all_preds.extend(preds.view(-1)[mask].cpu().tolist())

    f1 = compute_macro_f1(all_true, all_preds)
    report = classification_report(
        all_true, all_preds,
        labels=[LABEL2ID["COMMA"], LABEL2ID["PERIOD"], LABEL2ID["QUESTION"]],
        target_names=["COMMA", "PERIOD", "QUESTION"],
        zero_division=0,
    )
    return f1, report


# ИНФЕРЕНС
def predict(
    model,
    tokenizer,
    sentences_words: List[List[str]],
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 256,
    window_size: int = 128,
    step_size: int = 64,
) -> Dict[int, str]:
    """
    Предсказание с агрегацией перекрывающихся окон.
    Если для одного слова есть два предсказания — берём не-O с наивысшим confidence.
    Возвращает словарь: {глобальный_индекс_слова -> метка}
    """
    model.eval()

    # Превращаем список предложений в плоский список слов + их глобальные индексы
    flat_words = []
    for sent in sentences_words:
        flat_words.extend(sent)

    N = len(flat_words)
    # word_scores[i] = {label_id: max_confidence}
    word_preds: Dict[int, Dict[int, float]] = {}

    # Скользящее окно
    starts = list(range(0, max(1, N - step_size + 1), step_size))
    if starts and starts[-1] + window_size < N:
        starts.append(N - window_size)

    # Батчим окна
    windows = [(s, min(s + window_size, N)) for s in starts]
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start: batch_start + batch_size]
        batch_words_list = [flat_words[s:e] for s, e in batch_windows]

        encodings = tokenizer(
            batch_words_list,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)  # (B, seq_len, num_labels)

        for b_idx, (win_start, win_end) in enumerate(batch_windows):
            word_ids = encodings.word_ids(batch_index=b_idx)
            seen = set()
            for tok_idx, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                global_idx = win_start + word_id
                if global_idx >= N:
                    continue

                tok_probs = probs[b_idx, tok_idx].cpu().tolist()
                pred_label = int(np.argmax(tok_probs))
                confidence = tok_probs[pred_label]

                if global_idx not in word_preds:
                    word_preds[global_idx] = (pred_label, confidence)
                else:
                    old_pred, old_conf = word_preds[global_idx]
                    # Приоритет: не-O перед O; при равных — наивысший confidence
                    if old_pred == 0 and pred_label != 0:
                        word_preds[global_idx] = (pred_label, confidence)
                    elif old_pred != 0 and pred_label == 0:
                        pass  # оставляем старое
                    elif confidence > old_conf:
                        word_preds[global_idx] = (pred_label, confidence)

    # Собираем финальные метки
    result = {}
    for i in range(N):
        if i in word_preds:
            result[i] = ID2LABEL[word_preds[i][0]]
        else:
            result[i] = "O"

    return result

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инференс на: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()

    if args.test.endswith(".csv"):
        df = pd.read_csv(args.test)
        all_texts = df["input_text"].fillna("").astype(str).tolist()
        ids = df["id"].tolist()
    else:
        with open(args.test, "r", encoding="utf-8") as f:
            all_texts = [line.strip() for line in f if line.strip()]
        ids = list(range(len(all_texts)))

    # Токенизируем предложения в слова
    all_words = [text.split() for text in all_texts]
    
    pred_map = predict(
        model, tokenizer, all_words, device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    # Формирование submission.csv в правильном формате
    rows = []
    for sent_id, words in zip(ids, all_words):
        # Каждое предложение — отдельный вызов, без смешивания контекста
        pred_map = predict(model, tokenizer, [words], device,
                          batch_size=args.batch_size,
                          max_length=args.max_length,
                          window_size=args.window_size,
                          step_size=args.step_size)
        
        labels_for_sent = [pred_map.get(i, "O") for i in range(len(words))]
        
        # Проверка длины перед сохранением
        assert len(labels_for_sent) == len(words), f"Mismatch on {sent_id}"
        
        rows.append({"id": sent_id, "labels": " ".join(labels_for_sent)})

    sub_df = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "submission.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    
    del model
    torch.cuda.empty_cache()
    
    print(f"Submission сохранён: {out_path} ({len(sub_df)} строк)")


# MAIN
def parse_args():
    p = argparse.ArgumentParser(description="Казахская пунктуация: обучение и инференс")
    p.add_argument("--data", default="train.csv", help="Путь к train CSV/TXT")
    p.add_argument("--test", default="test.csv", help="Путь к test CSV/TXT (для инференса)")
    p.add_argument("--model", default="xlm-roberta-base",
                   help="HuggingFace модель: xlm-roberta-base | kz-transformers/kaz-roberta-kw-base")
    p.add_argument("--checkpoint", default="./best_model", help="Путь к сохранённой модели (для инференса)")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--chunk_size", type=int, default=64, help="Размер окна (для TXT)")
    p.add_argument("--overlap", type=int, default=12, help="Перекрытие окон (для TXT)")
    p.add_argument("--window_size", type=int, default=128, help="Размер окна при инференсе")
    p.add_argument("--step_size", type=int, default=64, help="Шаг окна при инференсе")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--infer", action="store_true", help="Запустить инференс (не обучение)")
    return p.parse_args()


import types

args = types.SimpleNamespace(
    # Данные
    data="train_full.csv",
    test="test.csv",
    # Модель
    model="kz-transformers/kaz-roberta-conversational",
    checkpoint="./outputs/best_model",
    output_dir="./outputs",
    # Обучение
    epochs=5,
    batch_size=16,
    gradient_accumulation_steps=2,
    max_length=256,
    lr=2e-5,
    weight_decay=0.01,
    # Инференс
    window_size=128,
    step_size=64,
    seed=42,
    # Режим: False = обучение, True = инференс
    infer=False,
    device_type = "cuda"
)

if args.infer:
    run_inference(args)
else:
    train(args)


import torch
import gc

gc.collect()
torch.cuda.empty_cache()

print(f"Занято VRAM: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


args.infer = True
args.checkpoint = "./outputs/best_model"
args.test = "test.csv"

run_inference(args)


import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))  # например: NVIDIA GeForce RTX 3090
