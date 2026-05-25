"""Utility functions for classification models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any


def load_sample_data(test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Load sample classification dataset (Iris or similar).
    
    Args:
        test_size: Proportion of test set
        random_state: Random state for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=random_state
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Testing features
    
    Returns:
        Scaled X_train, X_test, and scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: np.ndarray = None, model_name: str = "") -> Dict[str, float]:
    """
    Evaluate classification model with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
        model_name: Name of the model for display
    
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # ROC-AUC for binary classification
    if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    if model_name:
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")
    
    return metrics


def perform_cross_validation(model, X: np.ndarray, y: np.ndarray, 
                            cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring metric
    
    Returns:
        Dictionary with CV scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    cv_results = {
        'Mean CV Score': scores.mean(),
        'Std CV Score': scores.std(),
        'Individual Scores': scores
    }
    
    return cv_results


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str = "", save_path: str = None):
    """
    Plot and display confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                  model_name: str = "", save_path: str = None):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Path to save the figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
