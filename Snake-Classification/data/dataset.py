# Dataset and data loading code
from torchvision import datasets
from torch.utils.data import DataLoader
from .transforms import get_train_transforms, get_val_transforms


def get_datasets(train_path, val_path):
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

    return train_dataset, val_dataset


def get_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True
    )

    return train_loader, val_loader
