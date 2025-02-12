
import os
import torchvision.transforms as transforms
import torchvision.models as models
import torch
from types import SimpleNamespace

from dataset import TinyImageNet
from model import Net

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Transforms according to non-pretrained model (ResNet-18)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821))
    ])

    # Validation set and relative loader
    valset = TinyImageNet(os.path.join('..', 'data', 'tiny-imagenet-200'), transform=transform, data_subset='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)

    # Model
    args = SimpleNamespace(depth='18', pretrained=False, use_norm=True, freeze=[])
    resnet18 = Net(args, 1)
    resnet18.to(device)
    resnet18.load_state_dict(torch.load(os.path.join('..', 'models', 'model_A.pth'), weights_only=True))

    # Evaluation on validation set
    resnet18.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = resnet18.forward(inputs)
            predicted = (outputs.sigmoid() > 0.5).int()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print accuracy (10% positives, 90% negatives)
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on {len(valset)} validation images: {accuracy:.2f} %')


if __name__ == '__main__': # Entry point
    main()
