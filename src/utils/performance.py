import torch

def f1_score(y_true, y_pred, num_classes=None):
    """
    Compute the macro-averaged F1 score for multi-class classification.

    Args:
        y_true: Ground truth labels (tensor of shape [num_samples])
        y_pred: Predicted labels (tensor of shape [num_samples])
        num_classes: Number of classes; inferred from the labels when omitted.
    Returns:
        f1: Macro-averaged F1 score (float)
    """
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    if num_classes is None:
        num_classes = int(max(y_true.max().item(), y_pred.max().item())) + 1

    per_class_f1 = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()

        precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        per_class_f1.append(2 * (precision * recall) / (precision + recall + 1e-8))

    return sum(per_class_f1) / len(per_class_f1)

def accuracy(y_true, y_pred):
    """
    Compute the accuracy for multi-class classification.
    Args:
        y_true: Ground truth labels (tensor of shape [batch_size])
        y_pred: Predicted labels (tensor of shape [batch_size])
    Returns:
        accuracy: Accuracy (float)
    """
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    accuracy = correct / total
    return accuracy