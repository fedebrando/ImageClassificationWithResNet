
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Literal

class TinyImageNet(Dataset):
    _N_TRAIN_IMAGES_PER_CLASS = 500
    _N_TEST_IMAGES = 10000

    def __init__(self, data_dir, transform=None, subset: Literal['train', 'val', 'test']='train'):
        self._data_dir = data_dir       # root directory (tiny-imagenet-200)
        self._transform = transform
        self._subset = subset           # training, validation or test set
        self._classes = self._load_classes()
        if subset == 'val':
            self._val_labels = self._load_val_labels()
        
    def __len__(self):
        match self._subset:
            case 'train':
                return len(self._classes) * self._N_TRAIN_IMAGES_PER_CLASS
            case 'val':
                return len(self._val_labels)
            case 'test':
                return self._N_TEST_IMAGES
    
    def __getitem__(self, idx):
        match self._subset:
            case 'train':
                class_idx = idx // self._N_TRAIN_IMAGES_PER_CLASS
                class_offset = idx % self._N_TRAIN_IMAGES_PER_CLASS
                image = Image.open(
                    os.path.join(
                        self._data_dir,
                        'train',
                        self._classes[class_idx],
                        'images',
                        f'{self._classes[class_idx]}_{class_offset}.JPEG'
                    )
                ).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image, class_idx
            case 'val':
                image = Image.open(os.path.join(self._data_dir, 'val', 'images', f'val_{idx}.JPEG')).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image, self._val_labels[idx]
            case 'test':
                image = Image.open(os.path.join(self._data_dir, 'test', 'images', f'test_{idx}.JPEG')).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image

    # It returns the number of classes of Tiny ImageNet Dataset (storing this information is a skill of this class)
    def num_classes(self):
        return len(self._classes)

    # It reads wnids.txt file and returns a list with all class-codes
    def _load_classes(self) -> list[str]:
        classes = []
        with open(os.path.join(self._data_dir, 'wnids.txt'), 'r') as f:
            for line in f:
                classes.append(line[:-1])
        return classes
    
    # It reads val_annotations.txt and returns a list of indexes which correspond to validation labels
    def _load_val_labels(self) -> list[int]:
        labels = []
        with open(os.path.join(self._data_dir, 'val', 'val_annotations.txt'), 'r') as f:
            for line in f:
                class_code = line.split()[1] # the second column
                labels.append(self._classes.index(class_code))
        return labels
