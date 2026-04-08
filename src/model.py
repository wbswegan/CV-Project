import torch
import torch.nn as nn
from torchvision import models

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SkinCancerResNet(nn.Module):
    def __init__(self, num_classes=7, use_attention=True, pretrained=False):
        super(SkinCancerResNet, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = backbone.avgpool
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = SEBlock(512)

        num_ftrs = backbone.fc.in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def get_model(num_classes, pretrained=False):
    return SkinCancerResNet(num_classes=num_classes, use_attention=True, pretrained=pretrained)
