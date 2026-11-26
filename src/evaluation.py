"""
Model Evaluation and Metrics Visualization Module.

Computes real evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
and produces publication-quality comparison charts using Matplotlib.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(name: str, model, X_test, y_test):
    """
    Evaluate a single model on test features and labels.
    Returns a dictionary of valid evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='binary', zero_division=0))
    rec = float(recall_score(y_test, y_pred, average='binary', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='binary', zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "test_samples": int(len(y_test))
    }


def save_metrics(metrics_dict: dict, file_path: str):
    """Save metrics dictionary to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)


def load_metrics(file_path: str) -> dict:
    """Load metrics dictionary from a JSON file."""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_model_comparison(metrics_dict: dict, save_path: str):
    """
    Generate and save a bar chart comparing Accuracy, Precision, Recall, and F1.
    """
    models = list(metrics_dict.keys())
    if not models:
        return
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    
    for i, (metric, label, color) in enumerate(zip(metrics_names, labels, colors)):
        values = [metrics_dict[m].get(metric, 0) for m in models]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, values, width, label=label, color=color, edgecolor='none', alpha=0.9, zorder=3)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e293b')
    
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=11, fontweight='semibold', color='#334155')
    ax.set_title('Machine Learning Models Performance Comparison', fontsize=13, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight='semibold', color='#1e293b')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.legend(loc='lower right', framealpha=0.9, facecolor='#ffffff', edgecolor='#e2e8f0')
    
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(metrics_dict: dict, save_path: str):
    """
    Generate and save confusion matrix heatmaps for all trained models.
    """
    models = list(metrics_dict.keys())
    if not models:
        return
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), dpi=150)
    if n_models == 1:
        axes = [axes]
    
    fig.patch.set_facecolor('#ffffff')
    
    class_labels = ['FAKE (0)', 'REAL (1)']
    
    for idx, name in enumerate(models):
        ax = axes[idx]
        cm = np.array(metrics_dict[name].get('confusion_matrix', [[0, 0], [0, 0]]))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=11, fontweight='bold', color='#0f172a', pad=10)
        
        tick_marks = np.arange(len(class_labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_labels, fontsize=9)
        
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "#0f172a",
                        fontweight="bold", fontsize=10)
                
        ax.set_ylabel('True Label', fontsize=10, fontweight='semibold', color='#334155')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='semibold', color='#334155')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
