import torch
import torch.nn as nn
from torchvision import models
class CorrelationPredictor(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        """
        Args:
            pretrained (bool): If True, uses pre-trained weights
            freeze_backbone (bool): If True, freezes the backbone layers
        """
        super(CorrelationPredictor, self).__init__()
        
        # Load pre-trained ResNet model (smaller ResNet18 for faster training)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ResNet18's fc layer input features is 512
        in_features = self.backbone.fc.in_features
        
        # Replace with a regression head
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Output a single value for correlation
        )
    
    def forward(self, x):
        x = self.backbone(x)
        # Squeeze to remove extra dimension and match target
        return x.squeeze()