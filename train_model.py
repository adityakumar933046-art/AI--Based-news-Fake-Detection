"""
Model Training and Evaluation Script for AI Fake News Detection System.

Loads real dataset, preprocesses text, and trains 2 ML models:
  1. Logistic Regression
  2. Multinomial Naive Bayes
Computes real evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix),
generates comparison plots, and saves artifacts into models/ and static/images/.
"""

import os
import sys
import time
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text
from src.evaluation import (
    evaluate_model,
    save_metrics,
    plot_model_comparison,
    plot_confusion_matrices
)


def load_and_prepare_dataset(data_dir: str, sample_size: int = None, random_state: int = 42) -> pd.DataFrame:
    """
    Load raw dataset from CSV files, handle missing values, duplicates,
    and format text + label columns.
    """
    combined_csv = os.path.join(data_dir, 'news_dataset.csv')
    fake_csv = os.path.join(data_dir, 'Fake.csv')
    true_csv = os.path.join(data_dir, 'True.csv')
    
    if os.path.exists(combined_csv) and os.path.getsize(combined_csv) > 1000:
        print(f"[Dataset] Loading existing combined dataset: {combined_csv}")
        df = pd.read_csv(combined_csv)
    elif os.path.exists(fake_csv) and os.path.exists(true_csv):
        print(f"[Dataset] Loading raw Fake.csv and True.csv...")
        fake_df = pd.read_csv(fake_csv)
        true_df = pd.read_csv(true_csv)
        
        fake_df['label'] = 0  # 0 = FAKE
        true_df['label'] = 1  # 1 = REAL
        
        fake_df['content'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
        true_df['content'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
        
        df = pd.concat([
            fake_df[['content', 'label']],
            true_df[['content', 'label']]
        ], ignore_index=True)
        
        print(f"[Dataset] Saving unified news_dataset.csv to {combined_csv}...")
        df.to_csv(combined_csv, index=False)
    else:
        raise FileNotFoundError(
            f"Dataset not found in {data_dir}! Please place 'Fake.csv' and 'True.csv' "
            f"or 'news_dataset.csv' inside '{data_dir}'."
        )

    print(f"[Dataset] Total raw records: {len(df):,}")
    
    # Check and remove nulls
    initial_len = len(df)
    df = df.dropna(subset=['content', 'label']).copy()
    if len(df) < initial_len:
        print(f"[Dataset] Removed {initial_len - len(df)} rows with missing values.")

    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['content']).copy()
    if len(df) < initial_len:
        print(f"[Dataset] Removed {initial_len - len(df)} duplicate records.")

    # Filter out empty records
    df = df[df['content'].str.strip().str.len() > 10].copy()

    # Optional balanced subsample
    if sample_size and sample_size < len(df):
        print(f"[Dataset] Sampling {sample_size:,} balanced records for training...")
        half = sample_size // 2
        df_fake = df[df['label'] == 0].sample(n=min(half, (df['label'] == 0).sum()), random_state=random_state)
        df_real = df[df['label'] == 1].sample(n=min(half, (df['label'] == 1).sum()), random_state=random_state)
        df = pd.concat([df_fake, df_real]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[Dataset] Final dataset shape: {df.shape}")
    print(f"[Dataset] Class distribution: Fake (0) = {(df['label'] == 0).sum():,}, Real (1) = {(df['label'] == 1).sum():,}")
    return df


def train_and_evaluate(sample_size: int = 16000, random_state: int = 42):
    """
    Main training workflow for Logistic Regression and Multinomial Naive Bayes.
    """
    start_time = time.time()
    data_dir = os.path.join(PROJECT_ROOT, 'dataset')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    static_img_dir = os.path.join(PROJECT_ROOT, 'static', 'images')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_img_dir, exist_ok=True)

    # 1. Load data
    df = load_and_prepare_dataset(data_dir, sample_size=sample_size, random_state=random_state)

    # 2. Preprocess text
    print("\n[Preprocessing] Applying NLP cleaning and tokenization...")
    t0 = time.time()
    df['cleaned_text'] = df['content'].apply(clean_text)
    df = df[df['cleaned_text'].str.strip().str.len() > 5].copy()
    print(f"[Preprocessing] Completed in {time.time() - t0:.2f} seconds. Valid records: {len(df):,}")

    # 3. Train/Test Split (stratified)
    X = df['cleaned_text']
    y = df['label'].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y
    )
    print(f"\n[Split] Training samples: {len(X_train_raw):,}, Testing samples: {len(X_test_raw):,}")

    # 4. Feature Extraction: TF-IDF
    print("\n[TF-IDF] Fitting vectorizer on training data...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"[TF-IDF] Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # 5. Define ML Models: 2 Models Only
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state
        ),
        'Multinomial Naive Bayes': MultinomialNB(
            alpha=0.1
        )
    }

    # 6. Train and Evaluate each model
    metrics_summary = {}

    print("\n[Training & Evaluation]")
    for name, model in models.items():
        print(f"  Training {name}...")
        t_model_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t_model_start
        print(f"  Trained {name} in {train_time:.2f}s. Evaluating on test set...")
        
        metrics = evaluate_model(name, model, X_test, y_test)
        metrics['training_time_seconds'] = round(train_time, 2)
        metrics_summary[name] = metrics
        
        print(f"    -> Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        print(f"    -> Precision: {metrics['precision'] * 100:.2f}%")
        print(f"    -> Recall:    {metrics['recall'] * 100:.2f}%")
        print(f"    -> F1-Score:  {metrics['f1_score'] * 100:.2f}%")

    # 7. Save Models and Vectorizer
    print("\n[Artifacts] Saving models and vectorizer...")
    joblib.dump(models['Logistic Regression'], os.path.join(models_dir, 'logistic_regression.pkl'))
    joblib.dump(models['Multinomial Naive Bayes'], os.path.join(models_dir, 'naive_bayes.pkl'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

    # Remove old svm.pkl if present
    svm_path = os.path.join(models_dir, 'svm.pkl')
    if os.path.exists(svm_path):
        os.remove(svm_path)
        print("  -> Removed obsolete svm.pkl")

    print("  -> Saved logistic_regression.pkl, naive_bayes.pkl, tfidf_vectorizer.pkl")

    # 8. Save Metrics and Generate Plots
    metrics_file = os.path.join(models_dir, 'evaluation_metrics.json')
    save_metrics(metrics_summary, metrics_file)
    print(f"  -> Saved real evaluation metrics to {metrics_file}")

    comparison_chart = os.path.join(static_img_dir, 'model_comparison.png')
    confusion_chart = os.path.join(static_img_dir, 'confusion_matrices.png')
    
    print("\n[Visualization] Generating evaluation charts with Matplotlib...")
    plot_model_comparison(metrics_summary, comparison_chart)
    plot_confusion_matrices(metrics_summary, confusion_chart)
    print(f"  -> Saved {comparison_chart}")
    print(f"  -> Saved {confusion_chart}")

    total_time = time.time() - start_time
    print(f"\n[Done] Pipeline successfully finished in {total_time:.2f} seconds!")
    return metrics_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fake News Detection ML Models")
    parser.add_argument('--sample', type=int, default=16000,
                        help="Number of records to sample for training (default: 16000, 0 for full dataset)")
    parser.add_argument('--full', action='store_true',
                        help="Use entire dataset without sampling")
    args = parser.parse_args()

    sample_size = None if args.full or args.sample == 0 else args.sample
    train_and_evaluate(sample_size=sample_size)
