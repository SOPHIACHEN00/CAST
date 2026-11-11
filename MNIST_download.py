from torchvision import datasets, transforms
from pathlib import Path

DATA_ROOT = Path("data/raw")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  #MNIST 全训练集的像素均值； 标准差--> 均值0 方差1，降低无关噪音
])

train_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=transform
)

# 如果你用 multiprocessing / Ray / joblib

# 第一次下载请单进程做：

# # run once
# datasets.MNIST("data/raw", train=True, download=True)
# datasets.MNIST("data/raw", train=False, download=True)


# 之后所有脚本统一用：

# download=False

# 否则多个 worker 同时下数据 → 偶发 race condition。