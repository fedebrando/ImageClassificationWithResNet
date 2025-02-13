
import os
import torchvision.transforms as transforms
import torch
from types import SimpleNamespace

from dataset import TinyImageNet
from model import Net

def main():
    # Device selection (GPU preference)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Transforms according to non-pretrained model (ResNet-18)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821))
    ])

    # Validation set and relative loader
    valset = TinyImageNet(os.path.join('..', 'data', 'tiny-imagenet-200'), transform=transform, data_subset='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)

    # Label assigned to "Lemon" classes
    lemon_label = valset.class_to_label('n07749582')

    # Model
    args = SimpleNamespace(depth='18', pretrained=False, use_norm=True, freeze=[])
    resnet18 = Net(args, 1)
    resnet18.to(device)
    resnet18.load_state_dict(torch.load(os.path.join('..', 'models', 'model_A.pth'), weights_only=True))

    # Evaluation on validation set
    resnet18.eval()
    total_p, correct_p = 0, 0 # for positives
    total_n, correct_n = 0, 0 # for negatives
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = resnet18.forward(inputs)
            predicted = (outputs.sigmoid() > 0.5).int()

            p_mask = labels == lemon_label
            n_mask = labels != lemon_label

            total_p += p_mask.sum().item()
            correct_p += predicted[p_mask].sum().item()

            total_n += n_mask.sum().item()
            correct_n += (predicted[n_mask] == 0).sum().item()

    # Print stats
    accuracy = 100 * (correct_p + correct_n) / (total_p + total_n)
    balanced_accuracy = 100 * (correct_p / total_p + correct_n / total_n) / 2
    print(f'Correct positive classifications: {correct_p}/{total_p}')
    print(f'Correct negative classifications: {correct_n}/{total_n}')
    print(f'Accuracy of the network on the {len(valset)} validation images: {accuracy:.2f} %')
    print(f'Balanced accuracy of the network on the {len(valset)} validation images: {balanced_accuracy:.2f} %')

if __name__ == '__main__': # Entry point
    main()
