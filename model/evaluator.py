import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class Evaluator:

    def __init__(self):
        self.y_true = []
        self.y_score = []

    def add_batch(self, logits, labels):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.y_score.extend(probs.tolist())
        self.y_true.extend(labels.cpu().numpy().tolist())

    def compute(self):

        y_pred = [1 if p > 0.5 else 0 for p in self.y_score]

        precision = precision_score(self.y_true, y_pred, zero_division=0)
        recall    = recall_score(self.y_true, y_pred, zero_division=0)
        f1        = f1_score(self.y_true, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(self.y_true, self.y_score)
        except:
            auc = 0.0

        return precision, recall, f1, auc
