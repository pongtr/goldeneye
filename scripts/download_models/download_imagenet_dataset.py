import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define transformations for the validation dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create an ImageNet DataLoader for the validation dataset
imagenet_val_data = datasets.ImageNet(root='~/data', split='val', transform=transform, download=True)
val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=32, shuffle=False)
