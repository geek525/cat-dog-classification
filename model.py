import torch.nn as nn
from torchvision import models
from collections import OrderedDict

def create_model(num_classes: int = 2):

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 1024)),
        ('relu1', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(1024, 500)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(500, num_classes)),
    ]))

    model.fc = classifier
    return model