from torchvision import datasets, transforms
from pathlib import Path

DATA_ROOT = Path("data/raw")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standardize using the MNIST training-set pixel mean and standard deviation.
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

# When using multiprocessing, Ray, or joblib:

# Download the dataset once in a single process:

# # run once
# datasets.MNIST("data/raw", train=True, download=True)
# datasets.MNIST("data/raw", train=False, download=True)


# Use the following setting in subsequent workers:

# download=False

# Concurrent downloads from multiple workers can cause intermittent race conditions.
