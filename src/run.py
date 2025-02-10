
import torch
import torchvision.transforms as transforms
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from random import sample

from solver import Solver
from dataset import TinyImageNet

DATA_AUGMENTATION = [
    transforms.RandomHorizontalFlip(p=0.5)
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #transforms.RandomRotation(15),
    #transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
]

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default='unknown', help='name of current run')
    parser.add_argument('--model_name', type=str, default='unknown', help='name of the model to be saved/loaded')

    parser.add_argument('--depth', type=str, default='18', choices=['18', '34', '50', '101', '152'], help='depth of the ResNet model')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model (DEFAULT weights)')
    parser.add_argument('--freeze', type=str, nargs='+', default=[], help='train model freezing subset of layers by name')

    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in a batch')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses and validate model every that number of iterations')
    parser.add_argument('--class_accuracy', action='store_true', help='print also accuracy for each class')
    parser.add_argument('--resize_imgs', action='store_true', help='resize input images according to ImageNet dataset (224x224)')
    parser.add_argument('--data_augmentation', action='store_true', help='add data augmentation')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer used for training')
    parser.add_argument('--use_norm', action='store_true', help='use normalization layers in model')
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=None,
        help='use early stopping to prevent overfitting setting max non-improvements number on validation'
    )

    parser.add_argument('--dataset_path', type=str, default=os.path.join('..', 'data', 'tiny-imagenet-200'), help='path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join('..', 'models'), help='path were to save the trained model')
    parser.add_argument('--classes_subset', type=str, nargs='+', default=None, help='train (and validate) model with a subset of classes')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')

    return parser.parse_args()

def check_rand_classes_arg(classes_subset: list[str]):
    '''
    Modify received list adding (eventually) a specified number of random classes
    '''
    num_rand_classes = 0
    if classes_subset:
        for i, arg in enumerate(classes_subset):
            if arg.startswith('rand') and (n_str := arg[len('rand'):]).isnumeric():
                num_rand_classes = int(n_str)
                classes_subset.pop(i)
                break
        if num_rand_classes > 0:
            other_class_ids = []
            with open(os.path.join('..', 'data', 'tiny-imagenet-200', 'wnids.txt'), 'r') as f:
                for line in f:
                    class_id = line[:-1] # remove '\n'
                    if class_id not in classes_subset:
                        other_class_ids.append(class_id)
            classes_subset.extend(sample(other_class_ids, min(len(other_class_ids), num_rand_classes)))

def main(args):
    writer = SummaryWriter('../runs/' + 'run_{}'.format(args.run_name))

    # Define transforms                                                                 # these are computed by tinyimagenet_stats.py
    mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if args.pretrained else ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_lst = [   # general transformation for each data subset
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    if args.resize_imgs:
        transform_lst.insert(0, transforms.Resize(224)) # resize images from 64x64 to 224x224 (ImageNet standard scale)
    train_transform_lst = transform_lst.copy() # specific transformations for training (if there is data augmentation)
    transform = transforms.Compose(transform_lst)

    # Data augmentation
    if args.data_augmentation:
        train_transform_lst = DATA_AUGMENTATION + train_transform_lst
    train_transform = transforms.Compose(train_transform_lst)
    
    # Check and manage 'rand' presence in classes_subset
    check_rand_classes_arg(args.classes_subset)

    # Load training set
    trainset = TinyImageNet(data_dir=args.dataset_path, transform=train_transform, data_subset='train', classes_subset=args.classes_subset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Load validation set
    valset = TinyImageNet(data_dir=args.dataset_path, transform=transform, data_subset='val', classes_subset=args.classes_subset)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Device (GPU preference)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Define solver class
    solver = Solver(
        train_loader=trainloader,
        val_loader=valloader,
        device=device,
        writer=writer,
        args=args,
        data_augm_description=str(DATA_AUGMENTATION)
    )

    # TRAIN model
    solver.train()

if __name__ == '__main__': # Entry point
    args = get_args()
    print(args)
    main(args)
    