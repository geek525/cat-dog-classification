import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import os

DEFAULT_DATA_DIR = 'Cat_Dog_data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])


def get_train_val_loaders(data_dir: str = DEFAULT_DATA_DIR,
                          batch_size: int = 32,
                          val_ratio: float = 0.2):
    train_dir = os.path.join(data_dir, 'train')

    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    full_val_dataset = datasets.ImageFolder(root=train_dir, transform=eval_transforms)

    dataset_len = len(full_train_dataset)
    val_len = int(val_ratio * dataset_len)
    train_len = dataset_len - val_len

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    val_indices = val_subset.indices
    val_subset_transformed = Subset(full_val_dataset, val_indices)

    val_loader = DataLoader(
        val_subset_transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader


def get_test_loader(data_dir: str = DEFAULT_DATA_DIR,
                    batch_size: int = 32):
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    return test_loader