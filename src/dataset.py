
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Literal

class TinyImageNet(Dataset):
    _N_TRAIN_IMAGES_PER_CLASS = 500
    _N_TEST_IMAGES = 10000

    def __init__(self, data_dir, transform=None, subset: Literal['train', 'val', 'test']='train', training_classes: list[str]=None):
        self._data_dir = data_dir       # root directory (tiny-imagenet-200)
        self._transform = transform
        self._subset = subset           # training, validation or test set
        self._classes = self._load_classes()
        if subset != 'test':
            self._training_classes = [c for c in set(training_classes) if c in self._classes] if training_classes else self._classes # to avoid error on input or duplicates
            if not self._training_classes:
                self._training_classes = self._classes
            self._training_labels = [self._classes.index(c) for c in self._training_classes] if training_classes else [i for i in range(len(self._classes))]
        self._classes_descriptions = self._load_descriptions()
        if subset == 'val':
            val_labels = self._load_val_labels()
            self._val_indexes_labels = [(i, l) for i, l in enumerate(val_labels) if l in self._training_labels] # list[(img_index, label)]
        
    def __len__(self):
        match self._subset:
            case 'train':
                return len(self._training_classes) * self._N_TRAIN_IMAGES_PER_CLASS
            case 'val':
                return len(self._val_indexes_labels)
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
                        self._training_classes[class_idx],
                        'images',
                        f'{self._training_classes[class_idx]}_{class_offset}.JPEG'
                    )
                ).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image, self._training_labels[class_idx]
            case 'val':
                i, l = self._val_indexes_labels[idx]
                image = Image.open(os.path.join(self._data_dir, 'val', 'images', f'val_{i}.JPEG')).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image, l
            case 'test':
                image = Image.open(os.path.join(self._data_dir, 'test', 'images', f'test_{idx}.JPEG')).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image

    # It returns True if a subset of training class is used, False otherwise (None if this dataset is not for training)
    def training_with_subset(self) -> bool | None:
        return not (self._classes is self._training_classes) if self._subset == 'train' else None

    # It returns the list of couples (training class, training label) if this is a training set, None otherwise
    def training_classes_indexes(self) -> list[tuple[str, int]] | None:
        return zip(self._training_classes, self._training_labels) if self._subset == 'train' else None

    # It returns the number of classes of Tiny ImageNet Dataset (storing this information is a skill of this class)
    def num_classes(self) -> int:
        return len(self._classes)
    
    # It returns a copy of the train labels list
    def train_labels(self) -> int:
        return self._training_labels.copy()
    
    # It returns the description of the received class (class ID)
    def class_description(self, class_id: str) -> str:
        return self._classes_descriptions[class_id]

    # It returns the description of the received label
    def label_description(self, label: int) -> str:
        return self.class_description(self._classes[label])

    # It returns the classes/descriptions dict (it contains more classes, but to avoid temporal complexity it's ok)
    def _load_descriptions(self) -> dict[str, str]:
        classes_descriptions = {}
        with open(os.path.join(self._data_dir, 'words.txt'), 'r') as f:
            for line in f:
                splitted = line[:-1].split()
                classes_descriptions[splitted[0]] = ' '.join(splitted[1:]).split(',')[0]
        return classes_descriptions

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
