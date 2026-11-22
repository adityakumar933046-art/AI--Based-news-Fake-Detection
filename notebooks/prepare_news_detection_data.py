
import pandas as pd
from pathlib import Path
from nltk.corpus import brown

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

NEWS_PATH = BASE_DIR / "dataset" / "cleaned_welfake.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "news_detection_dataset.csv"

# Brown Corpus non-news categories
CATEGORIES = [
    "adventure",
    "fiction",
    "humor",
    "mystery",
    "romance",
    "science_fiction"
]

MIN_WORDS = 200
MAX_WORDS = 500
TARGET_WORDS = 300

# Step 1: Prepare non-news chunks
non_news_records = []

for file_id in brown.fileids(categories=CATEGORIES):

    sentences = brown.sents(fileids=[file_id])

    current_sentences = []
    current_word_count = 0
    chunk_number = 0

    for sentence in sentences:

        sentence_text = " ".join(sentence)
        sentence_words = len(sentence)

        # Avoid exceeding the maximum when possible
        if (
            current_word_count >= MIN_WORDS
            and current_word_count + sentence_words > MAX_WORDS
        ):
            text = " ".join(current_sentences)

            non_news_records.append({
                "content": text,
                "label": 0,
                "source": "brown",
                "group_id": file_id,
                "chunk_id": chunk_number
            })

            chunk_number += 1
            current_sentences = []
            current_word_count = 0

        current_sentences.append(sentence_text)
        current_word_count += sentence_words

        # Save a chunk once it reaches the target size
        if current_word_count >= TARGET_WORDS:

            text = " ".join(current_sentences)

            non_news_records.append({
                "content": text,
                "label": 0,
                "source": "brown",
                "group_id": file_id,
                "chunk_id": chunk_number
            })

            chunk_number += 1
            current_sentences = []
            current_word_count = 0

    # Keep any sufficiently large remaining chunk
    if current_word_count >= MIN_WORDS:

        text = " ".join(current_sentences)

        non_news_records.append({
            "content": text,
            "label": 0,
            "source": "brown",
            "group_id": file_id,
            "chunk_id": chunk_number
        })


# Step 2: Load WELFake news articles
news_df = pd.read_csv(NEWS_PATH)

news_df = news_df.dropna(subset=["content", "label"]).copy()

news_df["word_count"] = (
    news_df["content"].astype(str).str.split().str.len()
)

# Select articles with comparable lengths
news_df = news_df[
    news_df["word_count"].between(MIN_WORDS, MAX_WORDS)
]

# Step 3: Create non-news DataFrame
non_news_df = pd.DataFrame(non_news_records)

non_news_df["word_count"] = (
    non_news_df["content"].str.split().str.len()
)

# Step 4: Balance the two classes
sample_count = min(len(non_news_df), len(news_df))

news_sample = news_df.sample(
    n=sample_count,
    random_state=42
).copy()

news_sample["label"] = 1
news_sample["source"] = "welfake"
news_sample["group_id"] = [
    f"news_{i}" for i in range(len(news_sample))
]
news_sample["chunk_id"] = 0

# Step 5: Combine both classes
final_df = pd.concat(
    [non_news_df, news_sample],
    ignore_index=True
)

# Shuffle records
final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Keep required columns
final_df = final_df[
    ["content", "label", "source", "group_id",
     "chunk_id", "word_count"]
]

# Step 6: Save dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

final_df.to_csv(OUTPUT_PATH, index=False)

# Step 7: Display dataset information
print("\nDataset preparation completed!")

print("\nDataset shape:", final_df.shape)

print("\nLabel distribution:")
print(final_df["label"].value_counts())

print("\nSource distribution:")
print(final_df["source"].value_counts())

print("\nWord count statistics:")
print(final_df["word_count"].describe())

print("\nUnique Brown documents:")
print(
    final_df[final_df["source"] == "brown"]["group_id"].nunique()
)

print("\nSaved to:", OUTPUT_PATH)
