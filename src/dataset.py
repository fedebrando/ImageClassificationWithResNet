
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Literal, Iterable
from random import sample

class TinyImageNet(Dataset):
    '''
    Tiny ImageNet dataset
    '''
    _N_TINYIMAGENET_CLASSES = 200
    _N_TRAIN_IMAGES_PER_CLASS = 500
    _N_TEST_IMAGES = 10000

    def __init__(self, data_dir, transform=None, data_subset: Literal['train', 'val', 'test']='train', classes_subset: list[str] = None):
        self._data_dir = data_dir       # root directory (tiny-imagenet-200)
        self._transform = transform
        self._data_subset = data_subset   # training, validation or test set
        self._class_ids = self._load_class_ids(classes_subset=classes_subset)
        if self._data_subset == 'val':
            self._val_img_indexes_labels = self._load_val_img_indexes_labels(classes_subset=classes_subset)
        self._class_ids_descriptions = self._load_descriptions(classes_subset=classes_subset)
        
    def __len__(self):
        match self._data_subset:
            case 'train':
                return len(self._class_ids) * self._N_TRAIN_IMAGES_PER_CLASS
            case 'val':
                return len(self._val_img_indexes_labels)
            case 'test':
                return self._N_TEST_IMAGES
    
    def __getitem__(self, idx):
        match self._data_subset:
            case 'train':
                class_idx = idx // self._N_TRAIN_IMAGES_PER_CLASS
                class_offset = idx % self._N_TRAIN_IMAGES_PER_CLASS
                image = Image.open(
                    os.path.join(
                        self._data_dir,
                        'train',
                        self._class_ids[class_idx],
                        'images',
                        f'{self._class_ids[class_idx]}_{class_offset}.JPEG'
                    )
                ).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image, (class_idx if self.n_classes() > 1 else 1.0) # float in else statement for BCEWithLogitLoss
            case 'val':
                img_idx, label = self._val_img_indexes_labels[idx]
                image = Image.open(os.path.join(self._data_dir, 'val', 'images', f'val_{img_idx}.JPEG')).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image, (label if self.n_classes() > 1 else float(label)) # float(1) in else statement for BCEWithLogitLoss
            case 'test':
                image = Image.open(os.path.join(self._data_dir, 'test', 'images', f'test_{idx}.JPEG')).convert('RGB')
                if self._transform:
                    image = self._transform(image)
                return image

    def classes_subset_enabled(self) -> bool:
        '''
        Returns True if a classes subset is used, False otherwise
        '''
        return len(self._class_ids) < self._N_TINYIMAGENET_CLASSES

    def n_classes(self) -> int:
        '''
        Returns the number of classes used in training and validation stages
        '''
        return len(self._class_ids)
    
    def range_labels(self) -> Iterable[int]:
        '''
        Returns an iterable object on all existing labels
        '''
        return range(self.n_classes()) if self.n_classes() > 1 else range(1, 2)
    
    def class_description(self, class_id: str) -> str:
        '''
        Returns the description of the received class id
        '''
        return self._class_ids_descriptions[class_id]

    def label_description(self, label: int) -> str:
        '''
        Returns the description of the received label
        '''
        return self.class_description(self._class_ids[label + (0 if self.n_classes() > 1 else -1)])
    
    def class_to_label(self, class_id: str) -> int:
        '''
        Returns the label related to received class id
        '''
        return self._class_ids.index(class_id)

    def _load_descriptions(self, classes_subset: list[str] | None) -> dict[str, str]:
        '''
        Returns a dict where keys are the class ids and value are related descriptions
        '''
        classes_descriptions = {}
        with open(os.path.join(self._data_dir, 'words.txt'), 'r') as f:
            for line in f:
                [class_id, *description] = line[:-1].split()
                if (not classes_subset) or (classes_subset and class_id in classes_subset): # if not classes_subset, I decide to load all classes description to avoid control computational cost
                    classes_descriptions[class_id] = ' '.join(description).split(',')[0] # only the first description
                    if classes_subset and len(classes_descriptions) == len(classes_subset):
                        break
        return classes_descriptions

    def _load_class_ids(self, classes_subset: list[str] | None) -> list[str]:
        '''
        Reads wnids.txt file and returns a list with class-codes which are also in received classes_subset if it is truthy,
        all class ids otherwise
        '''
        # Check rand presence
        n = 0
        if classes_subset:
            for c in classes_subset:
                if c.startswith('rand') and (n_str := c[len('rand'):]).isnumeric():
                    n = int(n_str) # get suffix (the number of classes)
                    break

        # Reading file wnids.txt
        class_ids = []
        other_class_ids = []
        with open(os.path.join(self._data_dir, 'wnids.txt'), 'r') as f:
            for line in f:
                class_id = line[:-1]
                if (not classes_subset) or (classes_subset and class_id in classes_subset):
                    class_ids.append(class_id)
                elif n > 0:
                    other_class_ids.append(class_id)
        
        # Add (eventually) random classes to adjust classes_subset for next
        random_classes = sample(other_class_ids, min(n, len(other_class_ids)))
        if classes_subset:
            classes_subset.extend(random_classes)

        return class_ids + random_classes

    def _load_val_img_indexes_labels(self, classes_subset: list[str] | None) -> list[int]:
        '''
        Reads val_annotations.txt and returns a list of couples (image_index, label) for only class ids in classes_subset if it is truthy,
        for all class ids otherwise
        '''
        indexes_labels = []
        with open(os.path.join(self._data_dir, 'val', 'val_annotations.txt'), 'r') as f:
            for img_idx, line in enumerate(f):
                class_id = line.split()[1] # the second column
                if (not classes_subset) or (classes_subset and class_id in classes_subset):
                    indexes_labels.append((img_idx, self._class_ids.index(class_id) if self.n_classes() > 1 else 1))
        return indexes_labels
