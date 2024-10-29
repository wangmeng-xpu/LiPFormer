import logging

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class AllDataSet(Dataset):
    def __init__(self, size, data_pred, data_future, data_static):
        self.seq_len, self.label_len, self.pred_len = size
        self.data_pred = data_pred
        self.data_future = data_future
        self.data_static = data_static

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_pred[s_begin:s_end]
        seq_y = self.data_pred[r_begin:r_end]
        seq_x_future = self.data_future[s_begin : s_end + self.pred_len]
        seq_x_static = self.data_static[s_begin : s_end + self.pred_len]
        return seq_x, seq_y, seq_x_future, seq_x_static

    def __len__(self):
        return len(self.data_pred) - self.seq_len - self.pred_len + 1


class ETThDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        time_enc: bool = False,
        batch_size: int = 32,
        uni=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_path = data_path
        self.batch_size = batch_size
        self.uni = uni
        self.scale = scale
        self.time_enc = time_enc
        self.scaler = StandardScaler()
        self.uni = uni
        self.train_ds, self.valid_ds, self.test_ds = tuple(self._read_data())

    def _read_data(self):
        df_raw = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        df_raw = df_raw.astype(dtype=np.float32)
        if self.uni:
            df_raw = df_raw[["OT"]]
        else:
            df_raw = df_raw.iloc[:, :-4]
        df_static = pd.read_csv(
            self.data_path,
            usecols=["date", "month", "day", "weekday", "hour"],
            index_col=0,
            parse_dates=True,
        )
        df_static = df_static.astype(dtype=np.int32)
        border_list = (
            [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len],
            [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        )
        borders = list(zip(border_list[0], border_list[1]))
        df_data = df_raw

        logging.info("读取数据集: %s ,数据大小: %s", self.data_path, df_data.shape)
        if self.scale:
            train_data = df_data[border_list[0][0] : border_list[1][0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_static = df_static.values
        for idx in range(3):
            border = borders[idx]
            df_stamp = df_data.index[border[0] : border[1]].to_frame()
            print(df_stamp)
            data_dynamic = data[border[0] : border[1]]
            data_static = df_static[border[0] : border[1]]
            if self.uni:
                yield AllDataSet(
                (self.seq_len, self.label_len, self.pred_len),
                data_dynamic[:, :1],
                data_dynamic[:, 1:],
                data_static,
            )
            else:
                yield AllDataSet(
                    (self.seq_len, self.label_len, self.pred_len),
                    data_dynamic[:, :7],
                    data_dynamic[:, 7:],
                    data_static,
                )

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        return data * self.scaler.scale_[0] + self.scaler.mean_[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )


class ETTmDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        time_enc: bool = False,
        batch_size: int = 32,
        uni=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_path = data_path
        self.batch_size = batch_size
        self.scale = scale
        self.time_enc = time_enc
        self.scaler = StandardScaler()
        self.uni=uni
        self.train_ds, self.valid_ds, self.test_ds = tuple(self._read_data())

    def _read_data(self):
        if self.uni:
            df_raw = pd.read_csv(
                self.data_path, usecols=["OT"], index_col=0, parse_dates=True
            )
        else:
            df_raw = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        df_raw = df_raw.astype(dtype=np.float32)
        df_raw = df_raw.iloc[:, :-4]
        df_static = pd.read_csv(
            self.data_path,
            usecols=["date", "month", "day", "weekday", "hour"],
            index_col=0,
            parse_dates=True,
        )
        df_static = df_static.astype(dtype=np.int32)
        border_list = (
            [
                0,
                12 * 30 * 24 * 4 - self.seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            ],
            [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ],
        )
        borders = list(zip(border_list[0], border_list[1]))
        df_data = df_raw

        
        if self.scale:
            train_data = df_data[border_list[0][0] : border_list[1][0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_static = df_static.values
        for idx in range(3):
            border = borders[idx]
            df_stamp = df_data.index[border[0] : border[1]].to_frame()
            print(df_stamp)
            data_dynamic = data[border[0] : border[1]]
            data_static = df_static[border[0] : border[1]]
            yield AllDataSet(
                (self.seq_len, self.label_len, self.pred_len),
                data_dynamic[:, :7],
                data_dynamic[:, 7:],
                data_static,
            )

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        return data * self.scaler.scale_[0] + self.scaler.mean_[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )


class WeatherDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        time_enc: bool = False,
        batch_size: int = 32,
        uni=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_path = data_path
        self.batch_size = batch_size
        self.scale = scale
        self.time_enc = time_enc
        self.scaler = StandardScaler()
        self.uni = uni
        self.train_ds, self.valid_ds, self.test_ds = tuple(self._read_data())
        

    def _read_data(self):
        if self.uni:
            df_raw = pd.read_csv(
                self.data_path, usecols=["OT"], index_col=0, parse_dates=True
            )
        else:
            df_raw = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        df_raw = df_raw.astype(dtype=np.float32)
        df_raw = df_raw.iloc[:, :-4]
        df_static = pd.read_csv(
            self.data_path,
            usecols=["date", "month", "day", "weekday", "hour"],
            index_col=0,
            parse_dates=True,
        )
        df_static = df_static.astype(dtype=np.int32)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = len(df_raw) - num_test - num_train
        border_list = (
            [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
            [num_train, num_train + num_valid, len(df_raw)],
        )
        borders = list(zip(border_list[0], border_list[1]))
        df_data = df_raw

        
        if self.scale:
            train_data = df_data[border_list[0][0] : border_list[1][0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_static = df_static.values
        for idx in range(3):
            border = borders[idx]
            df_stamp = df_data.index[border[0] : border[1]].to_frame()
            print(df_stamp)
            data_dynamic = data[border[0] : border[1]]
            data_static = df_static[border[0] : border[1]]
            yield AllDataSet(
                (self.seq_len, self.label_len, self.pred_len),
                data_dynamic[:, :21],
                data_dynamic[:, 21:],
                data_static,
            )

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        return data * self.scaler.scale_[0] + self.scaler.mean_[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )


class ElectricityDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        time_enc: bool = False,
        batch_size: int = 32,
        uni=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_path = data_path
        self.batch_size = batch_size
        self.scale = scale
        self.time_enc = time_enc
        self.scaler = StandardScaler()
        self.uni = uni
        self.train_ds, self.valid_ds, self.test_ds = tuple(self._read_data())
        

    def _read_data(self):
        if self.uni:
            df_raw = pd.read_csv(
                self.data_path, usecols=["OT"], index_col=0, parse_dates=True
            )
        else:
            df_raw = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        df_raw = df_raw.astype(dtype=np.float32)
        df_raw = df_raw.iloc[:, :-4]
        df_static = pd.read_csv(
            self.data_path,
            usecols=["date", "month", "day", "weekday", "hour"],
            index_col=0,
            parse_dates=True,
        )
        df_static = df_static.astype(dtype=np.int32)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = len(df_raw) - num_test - num_train
        border_list = (
            [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
            [num_train, num_train + num_valid, len(df_raw)],
        )
        borders = list(zip(border_list[0], border_list[1]))
        df_data = df_raw

        if self.scale:
            train_data = df_data[border_list[0][0] : border_list[1][0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_static = df_static.values
        for idx in range(3):
            border = borders[idx]
            df_stamp = df_data.index[border[0] : border[1]].to_frame()
            print(df_stamp)
            data_dynamic = data[border[0] : border[1]]
            data_static = df_static[border[0] : border[1]]
            yield AllDataSet(
                (self.seq_len, self.label_len, self.pred_len),
                data_dynamic[:, :321],
                data_dynamic[:, 321:],
                data_static,
            )

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        return data * self.scaler.scale_[0] + self.scaler.mean_[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )


class TrafficDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        time_enc: bool = False,
        batch_size: int = 32,
        uni=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_path = data_path
        self.batch_size = batch_size
        self.scale = scale
        self.time_enc = time_enc
        self.scaler = StandardScaler()
        self.uni = uni
        self.train_ds, self.valid_ds, self.test_ds = tuple(self._read_data())
        

    def _read_data(self):
        if self.uni:
            df_raw = pd.read_csv(
                self.data_path, usecols=["OT"], index_col=0, parse_dates=True
            )
        else:
            df_raw = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        df_raw = df_raw.astype(dtype=np.float32)
        df_raw = df_raw.iloc[:, :-4]
        df_static = pd.read_csv(
            self.data_path,
            usecols=["date", "month", "day", "weekday", "hour"],
            index_col=0,
            parse_dates=True,
        )
        df_static = df_static.astype(dtype=np.int32)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = len(df_raw) - num_test - num_train
        border_list = (
            [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
            [num_train, num_train + num_valid, len(df_raw)],
        )
        borders = list(zip(border_list[0], border_list[1]))
        df_data = df_raw

        if self.scale:
            train_data = df_data[border_list[0][0] : border_list[1][0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_static = df_static.values
        for idx in range(3):
            border = borders[idx]
            df_stamp = df_data.index[border[0] : border[1]].to_frame()
            print(df_stamp)
            data_dynamic = data[border[0] : border[1]]
            data_static = df_static[border[0] : border[1]]
            yield AllDataSet(
                (self.seq_len, self.label_len, self.pred_len),
                data_dynamic[:, :862],
                data_dynamic[:, 862:],
                data_static,
            )

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        return data * self.scaler.scale_[0] + self.scaler.mean_[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )


class EPDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        time_enc: bool = False,
        batch_size: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_path = data_path
        self.batch_size = batch_size
        self.scale = scale
        self.time_enc = time_enc
        self.scaler = StandardScaler()
        self.train_ds, self.valid_ds, self.test_ds = tuple(self._read_data())

    def _read_data(self):
        df_raw = pd.read_csv(
            self.data_path,
            usecols=[
                "date_time",
                "日前价格",
                "实时价格",
                "统调负荷预测值(MW)",
                "外送预测值(MW)",
                "风光总加预测值(MW)",
                "风电预测值(MW)",
                "光伏预测值(MW)",
                "大同最高温度",
                "大同最低温度",
                "大同风力等级",
                "晋城最高温度",
                "晋城最低温度",
                "晋城风力等级",
                "晋中最高温度",
                "晋中最低温度",
                "晋中风力等级",
                "临汾最高温度",
                "临汾最低温度",
                "临汾风力等级",
                "吕梁最高温度",
                "吕梁最低温度",
                "吕梁风力等级",
                "朔州最高温度",
                "朔州最低温度",
                "朔州风力等级",
                "太原最高温度",
                "太原最低温度",
                "太原风力等级",
                "阳泉最高温度",
                "阳泉最低温度",
                "阳泉风力等级",
                "沂州最高温度",
                "沂州最低温度",
                "沂州风力等级",
                "运城最高温度",
                "运城最低温度",
                "运城风力等级",
                "长治最高温度",
                "长治最低温度",
                "长治风力等级",
            ],
            index_col=0,
            parse_dates=True,
        )
        df_raw = df_raw.astype(dtype=np.float32)
        df_raw = df_raw.loc[
            :,
            [
                "日前价格",
                "实时价格",
                "统调负荷预测值(MW)",
                "外送预测值(MW)",
                "风光总加预测值(MW)",
                "风电预测值(MW)",
                "光伏预测值(MW)",
                "大同最高温度",
                "大同最低温度",
                "大同风力等级",
                "晋城最高温度",
                "晋城最低温度",
                "晋城风力等级",
                "晋中最高温度",
                "晋中最低温度",
                "晋中风力等级",
                "临汾最高温度",
                "临汾最低温度",
                "临汾风力等级",
                "吕梁最高温度",
                "吕梁最低温度",
                "吕梁风力等级",
                "朔州最高温度",
                "朔州最低温度",
                "朔州风力等级",
                "太原最高温度",
                "太原最低温度",
                "太原风力等级",
                "阳泉最高温度",
                "阳泉最低温度",
                "阳泉风力等级",
                "沂州最高温度",
                "沂州最低温度",
                "沂州风力等级",
                "运城最高温度",
                "运城最低温度",
                "运城风力等级",
                "长治最高温度",
                "长治最低温度",
                "长治风力等级",
            ],
        ]
        df_static = pd.read_csv(
            self.data_path,
            usecols=[
                "date_time",
                "is_holiday",
                "大同风向",
                "大同天气状况",
                "晋城风向",
                "晋城天气状况",
                "晋中风向",
                "晋中天气状况",
                "临汾风向",
                "临汾天气状况",
                "吕梁风向",
                "吕梁天气状况",
                "朔州风向",
                "朔州天气状况",
                "太原风向",
                "太原天气状况",
                "阳泉风向",
                "阳泉天气状况",
                "沂州风向",
                "沂州天气状况",
                "运城风向",
                "运城天气状况",
                "长治风向",
                "长治天气状况",
            ],
            index_col=0,
            parse_dates=True,
        )
        df_static = df_static.astype(dtype=np.int32)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = len(df_raw) - num_test - num_train
        border_list = (
            [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
            [num_train, num_train + num_valid, len(df_raw)],
        )
        borders = list(zip(border_list[0], border_list[1]))
        df_data = df_raw

        
        if self.scale:
            train_data = df_data[border_list[0][0] : border_list[1][0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_static = df_static.values
        for idx in range(3):
            border = borders[idx]
            df_stamp = df_data.index[border[0] : border[1]].to_frame()
            print(df_stamp)
            data_dynamic = data[border[0] : border[1]]
            data_static = df_static[border[0] : border[1]]
            yield AllDataSet(
                (self.seq_len, self.label_len, self.pred_len),
                data_dynamic[:, :2],
                data_dynamic[:, 2:],
                data_static,
            )

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        return data * self.scaler.scale_[0] + self.scaler.mean_[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )


class CycleDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        time_enc: bool = False,
        batch_size: int = 32,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_path = data_path
        self.batch_size = batch_size
        self.scale = scale
        self.time_enc = time_enc
        self.scaler = StandardScaler()
        self.train_ds, self.valid_ds, self.test_ds = tuple(self._read_data())

    def _read_data(self):
        df_raw = pd.read_csv(
            self.data_path,
            usecols=[
                "date",
                "count",
                "max_temp",
                "mean_temp",
                "min_temp",
                "max_dew",
                "mean_dew",
                "min_dew",
                "max_hum",
                "mean_hum",
                "min_hum",
                "max_sea",
                "mean_sea",
                "min_sea",
                "max_vis",
                "mean_vis",
                "min_vis",
                "max_wind",
                "mean_wind",
                "max_gust",
                "precip",
                "cloud",
                "wind_dir",
            ],
            index_col=0,
            parse_dates=True,
        )
        df_raw = df_raw.astype(dtype=np.float32)

        df_raw = df_raw.loc[
            :,
            [
                "count",
                "max_temp",
                "mean_temp",
                "min_temp",
                "max_dew",
                "mean_dew",
                "min_dew",
                "max_hum",
                "mean_hum",
                "min_hum",
                "max_sea",
                "mean_sea",
                "min_sea",
                "max_vis",
                "mean_vis",
                "min_vis",
                "max_wind",
                "mean_wind",
                "max_gust",
                "precip",
                "cloud",
                "wind_dir",
            ],
        ]
        df_static = pd.read_csv(
            self.data_path,
            usecols=["date", "weekend"],
            index_col=0,
            parse_dates=True,
        )
        df_static = df_static.astype(dtype=np.int32)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = len(df_raw) - num_test - num_train
        border_list = (
            [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len],
            [num_train, num_train + num_valid, len(df_raw)],
        )
        borders = list(zip(border_list[0], border_list[1]))
        df_data = df_raw

        
        if self.scale:
            train_data = df_data[border_list[0][0] : border_list[1][0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_static = df_static.values
        for idx in range(3):
            border = borders[idx]
            df_stamp = df_data.index[border[0] : border[1]].to_frame()
            print(df_stamp)
            data_dynamic = data[border[0] : border[1]]
            data_static = df_static[border[0] : border[1]]
            yield AllDataSet(
                (self.seq_len, self.label_len, self.pred_len),
                data_dynamic[:, :1],
                data_dynamic[:, 1:],
                data_static,
            )

    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        return data * self.scaler.scale_[0] + self.scaler.mean_[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )
