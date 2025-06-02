
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze the early layers (optional)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last block and FC
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Modify the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
