import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(r'C:\Users\user\Desktop\GRAPHFUSSION\graphfusion-challenge\model')
from model import attention_model  

def compute_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def evaluate_model(model, device):
    model.eval()
    inputs = torch.randn(10, 3, 224, 224).to(device)  # Dummy input data
    labels = torch.randint(0, 2, (10,)).to(device)  # Dummy labels

    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    all_predictions = predicted.cpu().numpy()
    all_labels = labels.cpu().numpy()

    metrics = compute_metrics(all_predictions, all_labels)
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attention_model
    model.to(device)

    metrics = evaluate_model(model, device)
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
