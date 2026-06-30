# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    train.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了模型的训练流程
#               ★★★请在空白处填写适当的语句，将模型训练流程补充完整★★★
# -----------------------------------------------------------------------

import os
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import CustomNet

# Ensure current working directory is the project root
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def train_loop(epoch, dataloader, model, loss_fn, optimizer, device):
    """定义训练流程。
    :param epoch: 定义训练的总轮次
    :param dataloader: 数据加载器
    :param model: 模型，需在model.py文件中定义好
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 训练设备，即使用哪一块CPU、GPU进行训练
    """
    # 将模型置为训练模式
    model.train()

    # START----------------------------------------------------------
    size = len(dataloader.dataset)
    for t in range(epoch):
        total_loss = 0.0
        correct_num = 0

        for batch, sample in enumerate(dataloader):
            X = sample['image'].to(device)
            y = sample['label'].to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct_num += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % 30 == 0:
                current = min((batch + 1) * len(X), size)
                print(f"Epoch {t + 1:03d}/{epoch} | batch {batch:03d} | loss {loss.item():.6f} | {current}/{size}")

        avg_loss = total_loss / size
        accuracy = correct_num / size
        print(f"Epoch {t + 1:03d} finished | train loss: {avg_loss:.6f} | train accuracy: {accuracy:.2%}")

    # END------------------------------------------------------------

    # 保存模型
    torch.save(model, './models/model_0-5test_own.pkl')

if __name__ == "__main__":
    # ==================== 超参数 ====================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3       # 从头训练用 1e-3，继续训练用 1e-4 ~ 5e-5
    EPOCH = 15

    # ==================== 模型加载模式（三选一） ====================
    #   None = 从头训练
    #   .pkl 路径 = 加载后继续训练 / 迁移学习
    RESUME_PATH = None            # 继续训练：加载模型继续跑
    PRETRAINED_PATH = None        # 迁移学习：加载权重，用新数据微调

    # ---------------------------------------------------------------
    # 设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 加载 / 创建模型
    if RESUME_PATH:
        # 继续训练已保存的模型
        print(f"[继续训练] 加载: {RESUME_PATH}")
        checkpoint = torch.load(RESUME_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model = CustomNet()
            model.load_state_dict(checkpoint["model_state"])
            print(f"  恢复自 epoch {checkpoint.get('epoch', '?')}, "
                  f"准确率 {checkpoint.get('accuracy', '?')}")
        else:
            model = checkpoint
        model.to(device)
        LEARNING_RATE = 5e-5   # 继续训练用更小学习率，精调
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    elif PRETRAINED_PATH:
        # 迁移学习：借预训练模型的权重，在新数据上微调
        print(f"[迁移学习] 预训练模型: {PRETRAINED_PATH}")
        checkpoint = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model = CustomNet()
            model.load_state_dict(checkpoint["model_state"])
            print(f"  加载 state_dict, 原准确率 {checkpoint.get('accuracy', '?')}")
        else:
            model = checkpoint
        model.to(device)
        LEARNING_RATE = 5e-5   # 微调用小学习率
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    else:
        # 从头训练
        print("[从头训练] 初始化 CustomNet")
        model = CustomNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"  lr={LEARNING_RATE}  |  epochs={EPOCH}  |  batch={BATCH_SIZE}")

    # 数据加载器（Windows 下 num_workers 必须为 0，否则 spawn 模式会卡死）
    train_dataset = CustomDataset('./images/train.txt', '', ToTensor)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
    )

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 开始训练
    train_loop(EPOCH, train_dataloader, model, loss_fn, optimizer, device)
