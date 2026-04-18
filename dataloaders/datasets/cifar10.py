import torch
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Lambda

# standard constants for CIFAR10 normalization
CIFAR10_STD = torch.tensor([0.2023, 0.1994, 0.2010])


class sCIFAR10Dataset():
    def __init__(
        self,
        root: str = "./data/cifar10",
        download: bool = True,
        val_size: int = 5000,
        **kwargs
    ):
        
        self.base = CIFAR10(root=root, train=True, download=download, transform=ToTensor(), **kwargs)
        self.test = CIFAR10(root=root, train=False, download=download, transform=ToTensor(), **kwargs)
        
        self.val_size = val_size
        self.train_size = len(self.base) - val_size
        
        # State tracking to prevent double-normalization
        self._is_normalized = False

    def process(self):
        if self._is_normalized:
            return

        def process_sample(img):
            x = img.permute(1, 2, 0) # (32, 32, 3)
            x = x / CIFAR10_STD # Normalize - non-negative values   
            x = x.reshape(-1, 3) # (1024, 3)
            return x

        # Append custom logic to the existing transform pipeline
        transform_pipeline = Compose([
            self.base.transform, 
            Lambda(process_sample)
        ])

        self.base.transform = transform_pipeline
        self.test.transform = transform_pipeline
        self._is_normalized = True

    def create_dataset(self):
        # 4. Create subsets of train, valid, test based on val_size
        train, valid = random_split(self.base, [self.train_size, self.val_size])
        
        # 5. Output dataset dictionary and number of classes
        datasets = {
            "train": train,
            "valid": valid,
            "test": self.test
        }
        num_classes = 10
        
        return datasets, num_classes
