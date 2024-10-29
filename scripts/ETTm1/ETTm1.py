import sys


import tomli

from model.Lip import (
    CLIP,
    TilpPreTrain,
    FutureUnknowEncoder,
    FutureKnowEncoder,
    PastEncoder,
    LiP,
)
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from commons.auto_model import AutoModel


def main(seed):
    pl.seed_everything(seed)
    with open("./etth.toml", "rb") as f:
        config = tomli.load(f)
    future_enc = FutureUnknowEncoder(**config["data"] | config["model"]["clip"])
    static_enc = FutureKnowEncoder(**config["data"] | config["model"]["clip"])
    model = CLIP(static_enc, future_enc, output_dict=True)

    pl_model = TilpPreTrain(
        model,
        data_cls="ETTm1",
        data_path="../../data/ETTm1_diy.csv",
        **(config["data"] | config["hyperparams"] | config["model"]["clip"]),
    )
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath="./", monitor="val_loss", mode="min", verbose=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        callbacks=[
            ckpt,
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ],
    )

    trainer.fit(pl_model, pl_model.data_pl)
    pl_model = TilpPreTrain.load_from_checkpoint(ckpt.best_model_path, model=model)
    pl_model.eval()
    past_enc = PastEncoder(**config["model"]["clip"] | config["data"])
    feature_enc = pl_model.clip.feature_enc
    feature_enc.requires_grad_ = False
    model = LiP(past_enc, feature_enc, **config["data"] | config["model"]["clip"])

    pl_model = AutoModel(
        model,
        data_cls="ETTm1",
        data_path="../../data/ETTm1_diy.csv",
        **(config["data"] | config["hyperparams"] | config["model"]["clip"]),
    )
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath="./", monitor="val_loss", mode="min", verbose=True
    )
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        max_epochs=30,
        callbacks=[
            ckpt,
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ],
    )
    trainer.fit(pl_model, pl_model.data_pl)
    pl_model = AutoModel.load_from_checkpoint(ckpt.best_model_path, model=model)
    trainer.test(pl_model, pl_model.data_pl)


if __name__ == "__main__":
    seed = 2021
    main(seed)
