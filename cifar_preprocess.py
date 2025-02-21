import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Define image preprocessing steps:
transform = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize images to 224x224
    transforms.RandomHorizontalFlip(),           # Data augmentation: random horizontal flip
    transforms.ToTensor(),                       
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization using ImageNet values
                         std=[0.229, 0.224, 0.225])
])

# Load the CIFAR-10 training dataset (won't re-download if already available)
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the pre-trained ResNet-18 model and modify it to output embeddings
model = models.resnet18(pretrained=True)
model.fc = nn.Identity()  # Replace the final classification layer with an identity mapping
model.eval()              # Set the model to evaluation mode

# Fetch one batch of images
images, labels = next(iter(dataloader))
print("Batch image tensor shape:", images.shape)  # Expected: [32, 3, 224, 224]

# Extract embeddings from the images
with torch.no_grad():
    embeddings = model(images)
print("Embeddings shape:", embeddings.shape)  # Expected: [32, 512] for ResNet-18
