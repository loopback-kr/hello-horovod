import os
from os.path import expanduser, join, basename, splitext, isfile
import torch.nn as nn
import shutil
from zipfile import ZipFile
from filelock import FileLock
from torchvision import models, transforms, datasets
from requests import get

def get_model(num_classes):
    # Finetuning the convnet
    # Load a pretrained model and reset final fully connected layer.
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_transforms():
    data_transforms = {
        # Data augmentation and normalization for training.
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        # Just normalization for validation
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    return data_transforms

def load_dataset(dataset_dir='datasets', data_transforms=get_transforms(), force_download=False):
    url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
    archive_path = join(dataset_dir, basename(url))

    if force_download or not isfile(archive_path):
        with open(archive_path, 'wb') as f:
            f.write(get(url).content)
    
    # extract archive file
    with ZipFile(archive_path, 'r') as z:
        z.extractall(dataset_dir)
    
    with FileLock(expanduser("~/.torch_lock")):
        dsets = {x: datasets.ImageFolder(join(dataset_dir, splitext(basename(url))[0], x), transform=data_transforms[x]) for x in ['train', 'val']}
    return dsets