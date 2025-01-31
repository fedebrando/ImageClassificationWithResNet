
import torch
import torchvision.transforms as transforms
import argparse
from torch.utils.tensorboard import SummaryWriter
import os

from solver import Solver
from dataset import TinyImageNet

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default='1', help='name of current run')
    parser.add_argument('--model_name', type=str, default='1', help='name of the model to be saved/loaded')

    parser.add_argument('--depth', type=str, default='18', choices=['18', '34', '50', '101', '152'], help='depth of the ResNet model')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model (DEFAULT weights)')

    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer used for training')
    parser.add_argument('--use_norm', action='store_true', help='use normalization layers in model')
    parser.add_argument('--feat', type=int, default=16, help='number of features in model')
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=None,
        help='use early stopping to prevent overfitting setting max non-improvements number on validation'
    )

    parser.add_argument('--dataset_path', type=str, default=os.path.join('..', 'data', 'tiny-imagenet-200'), help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join('..', 'models'), help='path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')

    return parser.parse_args()

def main(args):
    writer = SummaryWriter('../runs/' + 'run_{}'.format(args.run_name))

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load train ds
    trainset = TinyImageNet(data_dir=args.dataset_path, transform=transform, subset='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # Load validation ds
    valset = TinyImageNet(data_dir=args.dataset_path, transform=transform, subset='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # Define solver class
    solver = Solver(
        train_loader=trainloader,
        val_loader=valloader,
        device=device,
        writer=writer,
        args=args
    )

    # TRAIN model
    solver.train()

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
    