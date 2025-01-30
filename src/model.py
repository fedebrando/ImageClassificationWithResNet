
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self._resnet = models.resnet34(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self._resnet.fc.out_features = num_classes
        self._resnet.add_module('Softmax', nn.Softmax(self._resnet.fc.out_features))

    def forward(self, x):
        return self._resnet(x)
