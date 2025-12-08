"""
Contains functionality for creating PyTorch DataLoader's for image classification data.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

num_workers = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = num_workers
):
  """
  Creates training and testing DataLoaders.
  Takes in a training and testing directory path and turns them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader

  Returns:
  A tuple of (train_dataloader, test_dataloader, class_names).
  Where class_names is a list of the target classes.

  Example usage:
    train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir, test_dir=path/to/test_dir, transform=some_transform, batch_size=32, num_workers=4)
  """

  print('Creating Datasets........')
  train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
  test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
  print(f'Train and Test datasets created')

  print('Creating DataLoaders........')
  train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
  print(f'Train and Test DataLoaders created')

  class_names = train_dataset.classes

  return train_dataloader, test_dataloader, class_names
