import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def calc_metrics(testy, scores):
    precision, recall, _ = precision_recall_curve(testy, scores)
    roc_auc = roc_auc_score(testy, scores)
    prc_auc = auc(recall, precision)

    return {'auroc': roc_auc, 'auprc': prc_auc}


def aggregate_by_index(
        x: torch.Tensor,
        index: torch.Tensor) -> torch.Tensor:
    assert len(x) == len(index)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    weight = torch.zeros(len(index), len(x)).to(x.device)  # L, N
    weight[index, torch.arange(x.shape[0])] = 1
    label_count = weight.sum(dim=1)
    weight = torch.nn.functional.normalize(weight, p=1, dim=1)  # l1 normalization
    mean = torch.mm(weight, x)  # L, F
    index = torch.arange(mean.shape[0])[label_count > 0]
    return mean[index]
