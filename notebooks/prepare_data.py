
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "dataset" / "WELFake_Dataset.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "cleaned_welfake.csv"

# Load dataset
df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Handle missing values
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Combine title and article text
df["content"] = (
    df["title"].astype(str) + " " +
    df["text"].astype(str)
).str.strip()

# Remove records with empty content
df = df[df["content"] != ""]

# Remove duplicate articles based on combined content
df = df.drop_duplicates(subset=["title", "text"])

# Keep only required columns
df = df[["title", "text", "content", "label"]]

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nDataset preparation completed!")
print("Cleaned shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nMissing values:")
print(df.isnull().sum())
print("\nSaved to:", OUTPUT_PATH)
