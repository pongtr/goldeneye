import torch
import torchvision.models as models

# Download ResNet-50 pre-trained on ImageNet
resnet = models.resnet50(pretrained=True)

# Save the model with .pt extension
torch.save(resnet.state_dict(), '../../src/othermodels/state_dicts/resnet50_model.pt')