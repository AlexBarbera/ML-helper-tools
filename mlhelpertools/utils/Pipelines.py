import os.path
from typing import Dict
import lightning
import torch

from models import LightningWrapper


def lightning_training_wrapper(model: torch.nn.Module, loss: torch.nn.Module, train_dl: torch.utils.data.DataLoader,
                               test_dl: torch.utils.data.DataLoader, train_params: Dict[str, str]):
    """
    Train a model with lightining API for standarization.

    :param model: Pytorch-like model that will be wrapped int a lightning model to train.
    :param loss: Pytorch loss module to use.
    :param train_dl: Dataloader to iterate in training.
    :param test_dl: Dataloader to iterate in validation.
    :param train_params: Dictionary with paramaters for training. Defaults are: ``{
        "log_dir": os.path.join(".", "logs"),
        "experiment_name": "my_experiment",
        "accelerator": "gpu"
    }``

    :return: Nothing
    """

    lmodel = LightningWrapper(model, loss)

    params = {
        "log_dir": os.path.join(".", "logs"),
        "experiment_name": "my_experiment",
        "accelerator": "gpu"
    }  # default params

    params = params | train_params

    callbacks = [
        lightning.pytorch.callbacks.ModelCheckpoint(
            params["log_dir"],
            save_last=True,
            every_n_epochs=1
        ),
        lightning.pytorch.callbacks.ModelSummary(),
        lightning.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
        lightning.pytorch.callbacks.TQDMProgressBar()
    ]

    logger = lightning.pytorch.loggers.TensorBoardLogger(params["log_dir"], name=params["experiment_name"])

    trainer = lightning.Trainer(callbacks=callbacks,
                                logger=logger,
                                accelerator=params["accelerator"])

    trainer.fit(lmodel, train_dataloaders=train_dl, val_dataloaders=test_dl)
