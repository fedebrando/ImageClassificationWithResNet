
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        # Depth choice
        match args.depth:
            case '18':
                self._resnet = models.resnet18(
                    weights=(models.ResNet18_Weights.DEFAULT if args.pretrained else None)
                )
            case '34':
                self._resnet = models.resnet34(
                    weights=(models.ResNet34_Weights.DEFAULT if args.pretrained else None)
                )
            case '50':
                self._resnet = models.resnet50(
                    weights=(models.ResNet50_Weights.DEFAULT if args.pretrained else None)
                )
            case '101':
                self._resnet = models.resnet101(
                    weights=(models.ResNet101_Weights.DEFAULT if args.pretrained else None)
                )
            case '152':
                self._resnet = models.resnet152(
                    weights=(models.ResNet152_Weights.DEFAULT if args.pretrained else None)
                )
        
        # Change the last layer and add Softmax according to the task (200 disjoint classes)
        self._resnet.fc.out_features = num_classes
        self._resnet.add_module('Softmax', nn.Softmax(self._resnet.fc.out_features))

    def forward(self, x):
        return self._resnet(x)
