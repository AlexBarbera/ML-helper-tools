import torch


class DeepEnergyLoss(torch.nn.Module):
    """
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
    """
    def __init__(self, alpha: float = 1.0, reduction: bool = True):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred_true, pred_false):
        loss_reg = self.alpha * (pred_true ** 2 + pred_false ** 2).mean()
        loss_div = pred_false.mean() - pred_true.mean()

        if self.reduction:
            return loss_reg + loss_div
        else:
            return loss_reg, loss_div
