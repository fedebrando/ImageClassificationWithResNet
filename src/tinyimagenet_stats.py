
'''
Mean and Std for Tiny ImageNet training subset
'''

import torch
import torchvision.transforms as transforms
from dataset import TinyImageNet
import os

BATCH_SIZE = 500
WORKERS = 4

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sum = torch.zeros(3).to(device)
    sum_squared = torch.zeros(3).to(device)
    trainset = TinyImageNet(os.path.join('..', 'data', 'tiny-imagenet-200'), transform=transforms.ToTensor(), data_subset='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

    # Compute sum and sum of squared for each channel
    for i, data in enumerate(trainloader):
        inputs, _ = data
        inputs = inputs.to(device)
        sum += inputs.sum(dim=[0, 2, 3]) # sum over all dimensions but first (the 3 channels)
        sum_squared += (inputs ** 2).sum(dim=[0, 2, 3])
        print(f'Batch {i} analized') # heartbeat log
    
    # Compute mean and std for each channel
    single_img_pixels_in_ch = trainset[0][0].size(1) * trainset[0][0].size(2) # number of pixels in a single image channel
    all_img_pixels_in_ch = single_img_pixels_in_ch * len(trainset) # sum of all single-channel pixels for all train images
    mean = sum / all_img_pixels_in_ch
    var = sum_squared / all_img_pixels_in_ch - mean ** 2 # variance = mean_of_squared - square_of_mean
    std = var ** 0.5

    # Print mean and std found
    print('mean =', mean)
    print('std =', std)

if __name__ == '__main__':
    main()
