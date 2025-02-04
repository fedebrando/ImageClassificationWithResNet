
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    '''
    Deep neural network with architecture like a ResNet
    '''
    def __init__(self, args, n_classes):
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
        
        # Normalization layers
        if not args.use_norm:
            self._remove_norm_layers(self._resnet)
        
        # Change fully-connected layer
        if n_classes > 1:
            self._resnet.fc = nn.Sequential(
                nn.Linear(self._resnet.fc.in_features, n_classes),
                nn.Softmax(dim=1) # multi-class classification with disjoint classes
            )
        else:
            self._resnet.fc = nn.Linear(self._resnet.fc.in_features, n_classes)

        # Freezing specified modules
        if args.freeze:
            self._freeze_specified_modules(args.freeze)

    def forward(self, x):
        '''
        Inference
        '''
        return self._resnet(x)
    
    def _remove_norm_layers(self, module):
        '''
        Replaces BatchNorm2d modules with the Identity ones
        '''
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
            else:
                self._remove_norm_layers(child)

    def _freeze_specified_modules(self, module_names):
        '''
        Freezes all parameters which names contains at least one of the received names
        '''
        for name, param in self._resnet.named_parameters():
            if any(arg_name in name for arg_name in module_names):
                param.requires_grad = False
