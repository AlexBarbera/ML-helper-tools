import lightning
import torch

class LightningWrapper(lightning.LightningModule):
    def __init__(self, model, loss):
        super(LightningWrapper, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch

        pred = self.model(x)

        loss = self.loss(pred, y)

        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.model.parameters())], []
