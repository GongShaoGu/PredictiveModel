import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import tomlkit
from numpy import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from functools import partial


def get_data_loader(
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """
    取得train_loader和test_loader

    Args:
        model_type: 模型的類別
        batch_size: 一個批次的大小

    Returns:
        train_loader: 訓練用的dataloader
        test_loader: 驗證用的dataloader
    """
    data_dir = "/home/ray_cluster/Documents/workspace/tune_population_based/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    my_get_loader = partial(DataLoader, batch_size=batch_size, shuffle=True)

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader, test_loader = (
        my_get_loader(train_dataset),
        my_get_loader(test_dataset),
    )

    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """
    訓練輸入的模型一個epoch

    Args:
        model: 要訓練的model
        optimizer: 訓練用的優化器
        train_loader: 訓練用的資料
        device: 訓練使用的裝置

    Returns:
        avg_loss: epoch的平均loss
        avg_time: 每個batch的平均訓練時間
    """
    model.to(device)
    total_loss = 0.0
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

        total_loss += loss.item()
        total_time += time.time() - start_time

    avg_loss = total_loss / len(train_loader)
    avg_time = total_time / len(train_loader)
    return avg_loss, avg_time


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
    def __init__(self, num_experiment: int, output_path: str) -> None:
        self.num_experiment = num_experiment
        self.output_path = output_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.toml_dict = tomlkit.table()
        self.toml_dict["hyper"] = tomlkit.array()
        self.toml_dict["data"] = tomlkit.array()

    def run(self) -> None:
        train_loader, test_loader = get_data_loader(batch_size=512)
        with tqdm(
            range(self.num_experiment), unit="experiment", ncols=100, leave=False
        ) as pbar:
            for _ in pbar:
                hyper = {
                    "lr": random.uniform(0.001, 1),
                    "momentum": random.uniform(0.001, 1),
                    "model_type": random.choice(
                        [
                            "resnet-18",
                            "resnet-34",
                            "resnet-50",
                            "resnet-101",
                            "resnet-152",
                        ]
                    ),
                }
                pbar.set_description("Training " + hyper["model_type"])
                if hyper["model_type"] == "resnet-18":
                    model = torchvision.models.resnet18()
                elif hyper["model_type"] == "resnet-34":
                    model = torchvision.models.resnet34()
                elif hyper["model_type"] == "resnet-50":
                    model = torchvision.models.resnet50()
                elif hyper["model_type"] == "resnet-101":
                    model = torchvision.models.resnet101()
                else:
                    model = torchvision.models.resnet152()
                optimizer = torch.optim.SGD(
                    params=model.parameters(),
                    lr=hyper["lr"],
                    momentum=hyper["momentum"],
                )
                data = {
                    "times": [],
                    "accs": [],
                }
                with tqdm(range(50), unit="epoch", ncols=80, leave=False) as epoch_bar:
                    for _ in epoch_bar:
                        _, t = train_epoch(
                            model=model,
                            optimizer=optimizer,
                            train_loader=train_loader,
                            device=self.device,
                        )
                        acc = test(
                            model=model, test_loader=test_loader, device=self.device
                        )
                        data["times"].append(t)
                        data["accs"].append(acc)
                self.toml_dict["hyper"].append(hyper)
                self.toml_dict["data"].append(data)

    def save_data(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(self.toml_dict))


if __name__ == "__main__":
    trainer = Trainer(100, "./data/results.toml")
    trainer.run()
    trainer.save_data()
