from typing import Dict

import lightning
from models import LightningWrapper


def lightning_training_wrapper(model, loss, train_dl, test_dl, train_params: Dict[str, str]):
    assert "log_dir" in train_params, "`log_dir` not found in `train_params`, found {}".format(train_params.keys())
    assert "experiment_name" in train_params, "`experiment_name` not found in `train_params`, found {}".format(
        train_params.keys())
    assert "accelerator" in train_params, "`accelerator` not found in `train_params`, found {}".format(
        train_params.keys())

    lmodel = LightningWrapper(model, loss)

    callbacks = [
        lightning.pytorch.callbacks.ModelCheckpoint(
            train_params["log_dir"],
            save_last=True,
            every_n_epochs=1
        ),
        lightning.pytorch.callbacks.ModelSummary(),
        lightning.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
        lightning.pytorch.callbacks.TQDMProgressBar()
    ]

    logger = lightning.pytorch.loggers.TensorBoardLogger(train_params["log_dir"], name=train_params["experiment_name"])

    trainer = lightning.Trainer(callbacks=callbacks,
                                logger=logger,
                                accelerator=train_params["accelerator"])

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)