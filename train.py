import argparse
import os
import time
import pandas as pd
from typing import Tuple, List

from numpy import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import psutil


class MyDataLoader:
    def __init__(self, data_dir: str) -> None:
        os.makedirs(data_dir, exist_ok=True)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        cpu_count = psutil.cpu_count(logical=True)
        self.num_workers = (
            cpu_count // 2 if type(cpu_count) is int and cpu_count > 2 else 1
        )

    def get_data_loader(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """
        取得train_loader和test_loader

        Args:
            batch_size: 一個批次的大小

        Returns:
            train_loader: 訓練用的dataloader
            test_loader: 驗證用的dataloader
        """
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader, test_loader


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> List:
    """
    訓練輸入的模型一個epoch

    Args:
        model: 要訓練的model
        optimizer: 訓練用的優化器
        train_loader: 訓練用的資料
        device: 訓練使用的裝置

    Returns:
        times: 每個batch的訓練時間
    """
    times = []
    model.to(device)
    total_time = 0.0
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for inputs, targets in train_loader:
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_time += time.time() - start_time
        times.append(total_time)
    return times


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    訓練輸入的模型驗證

    Args:
        model: 要驗證的model
        test_loader: 驗證用的資料
        device: 驗證使用的裝置

    Returns:
        acc: 模型的準確率
    """
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = correct / total
    return acc


class Trainer:
    def __init__(self, data_dir: str, num_experiment: int, output_path: str) -> None:
        """
        初始化Trainer

        Args:
            data_dir: 用以訓練的資料
            num_experiment: 試驗的次數
            output_path: 輸出資料的存放位置
        """
        self.data_dir = data_dir
        self.num_experiment = num_experiment
        self.output_path = output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq_max = psutil.cpu_freq().max
        cpu_freq_min = psutil.cpu_freq().min
        mem = psutil.virtual_memory()
        total_mem = mem.total / (1024**3)
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        else:
            gpu = "None"
        self.hardware = {
            "cpu_count": cpu_count,
            "cpu_freq_max": cpu_freq_max,
            "cpu_freq_min": cpu_freq_min,
            "mem": total_mem,
            "gpu": gpu,
        }

    def _save_data(self, data) -> None:
        os.makedirs(name=self.output_path, exist_ok=True)
        csv_file = os.path.join(self.output_path, "results.csv")

        df = pd.DataFrame(data)
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_file, mode="w", header=True, index=False)

    def run(self) -> None:
        """
        呼叫Trainer訓練，產生數據
        """
        my_data_loader = MyDataLoader(data_dir=self.data_dir)
        with tqdm(
            range(self.num_experiment), unit="experiment", ncols=100, leave=False
        ) as pbar:
            for _ in pbar:
                hyper = {
                    "lr": random.uniform(0.0001, 1),
                    "momentum": random.uniform(0.0001, 1),
                    "model_type": random.choice(
                        [
                            "resnet-18",
                            "resnet-34",
                            "resnet-50",
                        ]
                    ),
                    "batch_size": int(
                        random.choice(
                            [
                                128,
                                256,
                                512,
                                1024,
                            ]
                        )
                    ),
                }
                pbar.set_description("Training " + hyper["model_type"])
                if hyper["model_type"] == "resnet-18":
                    model = models.resnet18()
                elif hyper["model_type"] == "resnet-34":
                    model = models.resnet34()
                else:
                    model = models.resnet50()
                optimizer = torch.optim.SGD(
                    params=model.parameters(),
                    lr=hyper["lr"],
                    momentum=hyper["momentum"],
                )
                train_loader, test_loader = my_data_loader.get_data_loader(
                    hyper["batch_size"]
                )
                datas = []
                with tqdm(range(20), unit="epoch", ncols=80, leave=False) as epoch_bar:
                    for _ in epoch_bar:
                        times = train(
                            model=model,
                            optimizer=optimizer,
                            train_loader=train_loader,
                            device=self.device,
                        )
                        acc = test(
                            model=model, test_loader=test_loader, device=self.device
                        )
                        for time in times:
                            data = {}
                            data["time"] = time
                            data["acc"] = acc
                            data.update(hyper)
                            data.update(self.hardware)
                            datas.append(data)
                        self._save_data(data=datas)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("experiment", type=int, help="輸入試驗次數")
    args = parse.parse_args()
    trainer = Trainer("./data", args.experiment, "./output")
    trainer.run()
