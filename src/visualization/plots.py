import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_pred_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_misclassification_errors(y_true, y_pred, title="Misclassification Errors"):
    cm = confusion_matrix(y_true, y_pred)
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    
    plt.figure(figsize=(6,5))
    sns.barplot(x=["False Positives", "False Negatives"], y=[false_positives, false_negatives], palette="Blues")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()

def plot_probability_distribution(y_pred_prob, title="Probability Distribution"):
    plt.figure(figsize=(8,5))
    plt.hist(y_pred_prob, bins=20, color='blue', alpha=0.7)
    plt.xlabel("Predicted Probability (Positive Class)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

def plot_per_class_error_rate(y_true, y_pred, classes, title="Per-Class Error Rate"):
    cm = confusion_matrix(y_true, y_pred)
    class_errors = 1 - np.diag(cm) / np.sum(cm, axis=1)
    
    plt.figure(figsize=(6,4))
    sns.barplot(x=classes, y=class_errors, palette="Reds")
    plt.ylabel("Error Rate")
    plt.title(title)
    plt.show()
