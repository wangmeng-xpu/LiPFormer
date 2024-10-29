import time
import pytorch_lightning as pl
import torch
from commons.load_data import (
    ETThDataModule,
    ETTmDataModule,
    WeatherDataModule,
    ElectricityDataModule,
    TrafficDataModule,
    CycleDataModule,
    EPDataModule,
)
import torch.nn.functional as F
from commons.metric import mae, mape, mse, rmse
import numpy as np

model_dict = {
    "ETTh1": ETThDataModule,
    "ETTh2": ETThDataModule,
    "ETTm1": ETTmDataModule,
    "ETTm2": ETTmDataModule,
    "electricity": ElectricityDataModule,
    "cycle": CycleDataModule,
    "traffic": TrafficDataModule,
    "weather": WeatherDataModule,
    "EP": EPDataModule,
}


class AutoModel(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        data_cls: str,
        data_path: str,
        pdf_path: str = "./",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.seq_len = kwargs["seq_len"]
        self.pred_len = kwargs["pred_len"]
        self.label_len = kwargs["label_len"]
        self.lr = kwargs["lr"]
        self.weight_decay = kwargs["weight_decay"]
        self.data_pl = model_dict[data_cls](
            data_path=data_path,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            batch_size=kwargs["batch_size"],
            scale=kwargs["scale"],
            uni=kwargs["uni"],
        )
        self.model = model
        self.pdf_str = (
            pdf_path
            + self.model.__class__.__name__
            + str(time.strftime("%Y年%m月%d日%H时%M分%S秒", time.localtime()))
            + ".pdf"
        )

        self.inverse = kwargs["inverse"]

    def forward(self, x):
        self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def training_step(self, train_batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = train_batch
        dec_input = torch.zeros_like(batch_y[:, -self.pred_len :, :])
        dec_input = torch.cat([batch_y[:, : self.label_len, :], dec_input], dim=1)
        outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        f_dim = 0
        outputs = outputs[:, -self.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.pred_len :, f_dim:]
        loss = F.smooth_l1_loss(outputs, batch_y, beta=0.1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = valid_batch
        dec_input = torch.zeros_like(batch_y[:, -self.pred_len :, :])
        dec_input = torch.cat([batch_y[:, : self.label_len, :], dec_input], dim=1)
        outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)
        f_dim = 0
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        outputs = outputs[:, -self.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.pred_len :, f_dim:]
        loss = F.l1_loss(outputs, batch_y)
        self.log("val_loss", loss)

    # def on_test_epoch_start(self):
    #     self.pdf = PdfPages(self.pdf_str)

    def test_step(self, test_batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = test_batch
        dec_input = torch.zeros_like(batch_y[:, -self.pred_len :, :])
        dec_input = torch.cat([batch_y[:, : self.label_len, :], dec_input], dim=1)
        outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        f_dim = 0
        outputs = outputs[:, -self.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.pred_len :, f_dim:]
        if outputs.size(-1) == 1:
            outputs = outputs.repeat(1, 1, batch_y.size(-1))
        if self.inverse:
            outputs_true = self.data_pl.inverse_transform(
                data=outputs.reshape(-1, outputs.size(-1)).cpu().detach().numpy()
            ).reshape(outputs.size())
            batch_y_true = self.data_pl.inverse_transform(
                data=batch_y.reshape(-1, batch_y.size(-1)).cpu().detach().numpy()
            ).reshape(batch_y.size())
            outputs_true = np.maximum(outputs_true, 0)
            batch_x = self.data_pl.inverse_transform(
                data=batch_x.reshape(-1, batch_x.size(-1)).cpu().detach().numpy()
            ).reshape(batch_x.size())
        else:
            outputs_true = outputs.cpu().detach().numpy()
            batch_y_true = batch_y.cpu().detach().numpy()
            batch_x = batch_x.cpu().detach().numpy()
        self.log(
            "test_rmse", rmse(batch_y_true[:, :, f_dim:], outputs_true[:, :, f_dim:])
        )
        self.log(
            "test_mae", mae(batch_y_true[:, :, f_dim:], outputs_true[:, :, f_dim:])
        )
        self.log(
            "test_mape", mape(batch_y_true[:, :, f_dim:], outputs_true[:, :, f_dim:])
        )
        self.log(
            "test_mse", mse(batch_y_true[:, :, f_dim:], outputs_true[:, :, f_dim:])
        )
        # l_fn = evaluate.load("mase")
        # mase  = torch.tensor([ l_fn.compute(predictions=outputs[i,:,f_dim],references=batch_y[i,:,f_dim],training=batch_x[i,:,f_dim],periodicity=24*4)["mase"] for i in range(outputs.shape[0])]).mean()
        # self.log("test_mase", mase)
        # print(batch_y_true.shape, outputs_true.shape)
        # for i in range(batch_y_true.shape[0]):
        #     plt.figure()
        #     plt.plot(batch_y_true[i, :, 0], label="true")
        #     plt.plot(outputs_true[i, :, 0], label="pred")
        #     plt.legend(["true", "pred"])
        #     self.pdf.savefig()
        #     plt.close()

    # def on_test_end(self) -> None:
    #     self.pdf.close()
