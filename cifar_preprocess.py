import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define image preprocessing steps:
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Download and load the CIFAR-10 training dataset.
# If already downloaded in './data', it won't download again.
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example: Iterate over one batch and inspect shapes
images, labels = next(iter(dataloader))
print("Batch image tensor shape:", images.shape)  # Expected: [32, 3, 224, 224]
