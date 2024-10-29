import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import pytorch_lightning as pl
from commons.load_data import (
    ETThDataModule,
    ETTmDataModule,
    WeatherDataModule,
    ElectricityDataModule,
    TrafficDataModule,
    CycleDataModule,
    EPDataModule,
)

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


class CLIPLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.prev_nim_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.prev_nim_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                raise NotImplementedError
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_nim_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, past_features, targets, logit_scale):
        # print(logit_scale.shape,targets.shape,past_features.shape)
        if self.world_size > 1:
            raise NotImplementedError
        else:
            logit_per_past = logit_scale * past_features @ targets.T
            logit_per_target = logit_scale * targets @ past_features.T
        return logit_per_past, logit_per_target

    def forward(self, past_features, targets, logit_scale, output_dict=False):
        device = past_features.device
        logit_per_past, logit_per_target = self.get_logits(
            past_features, targets, logit_scale
        )
        labels = self.get_ground_truth(device, past_features.shape[0])
        total_loss = (
            F.cross_entropy(logit_per_past, labels)
            + F.cross_entropy(logit_per_target, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss


class LiP(nn.Module):
    def __init__(self, past_enc, feature_enc, **kwargs):
        super().__init__()
        self.past_enc = past_enc
        self.feature_enc = feature_enc
        self.pred_len = kwargs["pred_len"]
        self.d_model = kwargs["d_model"]
        self.future_channels = kwargs["future_channels"]
        # self.past_enc.requires_grad_ = False
        # self.feature_enc.requires_grad_ = False
        # self.linear = nn.Linear(self.d_model, self.pred_len * self.future_channels)
        # self.linear1 = nn.Linear(self.d_model, self.pred_len)
        # self.tau = nn.Parameter(torch.tensor(0.5))
        # self.tau = 0.001
        self.drop = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(self.pred_len, self.pred_len)

    def forward(self, seq_x, seq_x_future, _, seq_x_static):
        past_dy = torch.cat([seq_x, seq_x_future[:, : -self.pred_len, :]], dim=-1)
        past_stat = seq_x_static[:, : -self.pred_len, :]
        static_dy = seq_x_future[:, -self.pred_len :, :]
        static_stat = seq_x_static[:, -self.pred_len :, :]
        # seq_last = past_dy[:, -1:, :].detach()
        # past_dy = past_dy-seq_last
        past = self.past_enc(past_dy, past_stat)
        x = self.feature_enc(static_dy, static_stat)
        x = F.gelu(x)
        x = self.linear2(x)
        x = x.reshape(x.shape[0], self.pred_len, -1)
        x = self.drop(x)
        x = x + past
        return x


class TilpPreTrain(pl.LightningModule):
    def __init__(
        self, model: nn.Module, data_cls: str, data_path: str, **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.lr = kwargs["lr"]
        self.seq_len = kwargs["seq_len"]
        self.pred_len: int = kwargs["pred_len"]
        self.label_len = kwargs["label_len"]
        self.weight_decay = kwargs["weight_decay"]
        self.clip = model
        self.accum_freq = kwargs["accum_freq"]
        self.inverse = kwargs["inverse"]
        self.data_pl = model_dict[data_cls](
            data_path=data_path,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            batch_size=kwargs["batch_size"],
            scale=kwargs["scale"],
            time_enc=kwargs["time_enc"],
            uni=kwargs["uni"],
        )
        self.batch_size = kwargs["batch_size"]
        self.scaler = None
        self.is_past: bool = kwargs["is_past"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.clip.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def forward(self, x):
        return self.clip(x)

    def on_train_epoch_start(self):
        if self.accum_freq > 1:
            self.pasts, self.targets, self.features = [], [], {}
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # idx_accum = batch_idx // self.accum_freq
        # step = self.batch_size // self.accum_freq
        loss = CLIPLoss()
        seq_x, seq_y, seq_x_future, seq_x_static = batch
        # torch.cat([i[0],i[2][:,:-96,:]],dim=-1),i[3][:,:-96,:],i[1][:,-96:,:]
        if self.accum_freq == 1:
            if self.is_past:
                x = torch.cat([seq_x, seq_x_future[:, : -self.pred_len, :]], dim=-1)
                model_out = self.clip(
                    x,
                    seq_x_static[:, : -self.pred_len, :],
                    seq_y[:, -self.pred_len :, :],
                )
            else:
                model_out = self.clip(
                    seq_x_future[:, -self.pred_len :, :],
                    seq_x_static[:, -self.pred_len :, :],
                    seq_y[:, -self.pred_len :, :],
                )
            # logits_scale = model_out["logits_scale"]
            losses = loss(**model_out, output_dict=True)
            total_loss = sum(losses.values())
            self.log("train_loss", total_loss)
            return total_loss if self.scaler is None else NotImplementedError
        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        loss = CLIPLoss()
        seq_x, seq_y, seq_x_future, seq_x_static = batch
        if self.is_past:
            x = torch.cat([seq_x, seq_x_future[:, : -self.pred_len, :]], dim=-1)
            model_out = self.clip(
                x, seq_x_static[:, : -self.pred_len, :], seq_y[:, -self.pred_len :, :]
            )
        else:
            model_out = self.clip(
                seq_x_future[:, -self.pred_len :, :],
                seq_x_static[:, -self.pred_len :, :],
                seq_y[:, -self.pred_len :, :],
            )
        losses = loss(**model_out, output_dict=True)
        total_loss = sum(losses.values())
        self.log("val_loss", total_loss)
        return total_loss


class PastEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.enc_in = kwargs["enc_in"]
        self.d_model = kwargs["d_model"]
        self.future_channels = kwargs["future_channels"]
        # self.lucky = nn.Embedding(self.enc_in, self.d_model // 2)
        self.seq_len = kwargs["seq_len"]
        self.pred_len = kwargs["pred_len"]
        self.patch_len = kwargs["patch_len"]
        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.attn0 = nn.MultiheadAttention(
            self.seq_len // self.patch_len, 1, batch_first=True
        )
        self.attn1 = nn.MultiheadAttention(
            self.d_model, 4, batch_first=True, dropout=kwargs["dropout"]
        )
        self.dropout0 = nn.Dropout(kwargs["dropout"])
        self.dec_in = nn.Parameter(
            torch.randn(self.pred_len // self.patch_len, 1, self.d_model)
        )
        self.dropout1 = nn.Dropout(kwargs["dropout"])

        self.linear = nn.Linear(self.d_model, self.patch_len)
        self.linear2 = nn.Linear(self.seq_len // self.patch_len, self.d_model)
        self.linear3 = nn.Linear(self.d_model, self.pred_len // self.patch_len)
        self.abl = nn.Linear(
            self.seq_len // self.patch_len, self.seq_len // self.patch_len
        )

    def forward(self, x, x_mark):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        B, L, C = x.shape
        N = self.seq_len // self.patch_len

        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)
        xx = xw.transpose(1, 2)
        xd = self.attn0(xx, xx, xx)[0] + xx

        xd = xd.transpose(1, 2)

        xd = self.linear_patch(xd)
        xd = F.gelu(xd)

        enc_out = self.attn1(xd, xd, xd)[0] + xd
        enc_out = self.linear2(enc_out.transpose(1, 2)).transpose(1, 2)
        enc_out = F.gelu(enc_out)
        enc_out = self.linear3(enc_out.transpose(1, 2)).transpose(1, 2)
        yw = self.linear(enc_out).reshape(B, C, -1).permute(0, 2, 1)
        y = yw + seq_last
        return y[:, :, : self.future_channels]


class FutureUnknowEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.channels = kwargs["future_channels"]
        self.pred_len = kwargs["pred_len"]
        self.hidden_layers = kwargs["d_model"]
        self.linear = nn.Linear(self.channels, self.hidden_layers)
        self.attn = nn.MultiheadAttention(
            self.hidden_layers, 4, batch_first=True, dropout=kwargs["dropout"]
        )
        self.proj = nn.Linear(self.hidden_layers, 1)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.attn(x, x, x)[0] + x
        x = self.proj(x).squeeze(-1)
        return x


class FutureKnowEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.channels = kwargs["channels"]
        self.embed_dim = kwargs["embed_dim"]
        self.hidden_layers = kwargs["d_model"]
        self.future_static_nums: list[int] = kwargs["future_static_nums"]
        self.seq_len = kwargs["seq_len"]
        self.pred_len = kwargs["pred_len"]
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.static_embeding = nn.ModuleList()
        for num_embeddings in self.future_static_nums:
            self.static_embeding.append(nn.Embedding(num_embeddings, self.embed_dim))
        # self.linear_dy = nn.Linear(1,self.embed_dim)
        # self.linear_dim = nn.Linear(
        #     self.embed_dim*(self.channels+len(self.future_static_nums)),
        #     self.hidden_layers)
        self.linear = nn.Linear(
            self.channels + len(self.future_static_nums) * self.embed_dim,
            self.hidden_layers,
        )
        # self.linear_seq = nn.Linear(self.seq_len, self.pred_len)
        self.attn = nn.MultiheadAttention(
            self.hidden_layers, 4, batch_first=True, dropout=kwargs["dropout"]
        )
        self.proj = nn.Linear(self.hidden_layers, 1)

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor):
        # print(x_dynamic.shape)
        x_static = [
            self.static_embeding[i](x_static[:, :, i])
            for i in range(len(self.future_static_nums))
        ]
        x_static = torch.stack(x_static, dim=2)

        x_static = x_static.reshape(x_dynamic.shape[0], x_dynamic.shape[1], -1)

        x_dynamic = x_dynamic

        x = torch.cat([x_dynamic, x_static], dim=-1)
        x = x.reshape(x.shape[0], self.pred_len, -1)
        x = self.linear(x)
        x = self.attn(x, x, x)[0] + x
        x = self.proj(x).squeeze(-1)
        return x


class CLIP(nn.Module):
    def __init__(
        self, feature_encoder: nn.Module, future_encoder: nn.Module, output_dict=False
    ) -> None:
        super().__init__()
        self.output_dict = output_dict
        self.feature_enc = feature_encoder
        self.future_enc = future_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_feature(self, x, static, normailze=False):
        features = self.feature_enc(x, static)
        return F.normalize(features, dim=-1) if normailze else features

    def encoder_future(self, x, normailze=False):
        futures = self.future_enc(x)
        return F.normalize(futures, dim=-1) if normailze else futures

    def forward(
        self,
        feature: torch.Tensor | None = None,
        feature_static: torch.Tensor | None = None,
        future: torch.Tensor | None = None,
    ):
        features = (
            self.encode_feature(feature, feature_static, normailze=True)
            if feature is not None
            else None
        )
        futures = (
            self.encoder_future(future, normailze=True) if future is not None else None
        )
        if self.output_dict:
            return {
                "past_features": features,
                "targets": futures,
                "logit_scale": self.logit_scale.exp(),
            }
        else:
            return features, futures, self.logit_scale.exp()
