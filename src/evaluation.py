# src/evaluation.py
"""
Model evaluation, metrics, and analysis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate models and generate metrics."""
    
    def __init__(self, output_dir: str = 'results/'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = specificity
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = 'Model') -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name for title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Purchase', 'Purchase'],
                   yticklabels=['No Purchase', 'Purchase'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {filepath}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      model_name: str = 'Model') -> None:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Model name for title
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'{model_name}_roc_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {filepath}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    model_name: str = 'Model') -> None:
        """Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Model name for title
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'{model_name}_pr_curve.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curve saved to {filepath}")
    
    def plot_feature_importance(self, model, feature_names: list,
                               model_name: str = 'Model', top_n: int = 15) -> None:
        """Plot feature importance.
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            model_name: Model name for title
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'{model_name}_feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {filepath}")
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = 'Model') -> str:
        """Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name
            
        Returns:
            Classification report string
        """
        report = classification_report(
            y_true, y_pred,
            target_names=['No Purchase', 'Purchase']
        )
        
        print(f"\n{'='*80}")
        print(f"Classification Report - {model_name}")
        print(f"{'='*80}")
        print(report)
        
        return report
    
    def optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, float]:
        """Find optimal decision threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        thresholds = np.arange(0.1, 1.0, 0.01)
        best_threshold = 0.5
        best_value = 0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                value = f1_score(y_true, y_pred)
            elif metric == 'precision':
                value = precision_score(y_true, y_pred)
            elif metric == 'recall':
                value = recall_score(y_true, y_pred)
            else:
                value = f1_score(y_true, y_pred)
            
            if value > best_value:
                best_value = value
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} with {metric}={best_value:.4f}")
        return best_threshold, best_value
