
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self._resnet = models.resnet18(weights=None)
        self._resnet.fc.out_features = num_classes
        self._resnet.add_module('Softmax', nn.Softmax(self._resnet.fc.out_features))

    def forward(self, x):
        return self._resnet(x)
