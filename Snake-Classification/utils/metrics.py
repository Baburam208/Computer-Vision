import torchmetrics
import torch


class MetricTracker:
    def __init__(self, num_classes):
        self.device = torch.device('cpu')  # Initialize on CPU
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def to(self, device):
        """Move all metrics to the specified device"""
        self.device = device
        self.accuracy = self.accuracy.to(device)
        self.f1 = self.f1.to(device)
        return self

    def update(self, preds, targets):
        self.accuracy.update(preds, targets)
        self.f1.update(preds, targets)

    def compute(self):
        return {
            "accuracy": self.accuracy.compute(),
            "f1": self.f1.compute()
        }

    def reset(self):
        self.accuracy.reset()
        self.f1.reset()
