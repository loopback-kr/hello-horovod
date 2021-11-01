import torch.nn as nn
from requests import get
from requests.auth import HTTPBasicAuth
from zipfile import ZipFile
from filelock import FileLock
from torchvision import datasets, transforms, models
from os.path import join, basename, isfile, isdir, expanduser, splitext

DATASET_URL = {
    'hymenoptera' : 'https://download.pytorch.org/tutorial/hymenoptera_data.zip',
}

def get_model(model, num_class):
    if model == 'MNIST':
        from models.mnist import Net
        return Net(num_class)
    elif model == 'resnet18_fine':
        # Finetuning the convnet
        # Load a pretrained model and reset final fully connected layer.
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        # Here the size of each output can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, num_class)
        return model
    else:
        raise NotImplementedError('Unknown model.')

def get_transforms(dataset):
    
    if dataset == 'MNIST':
        return {
            'train':transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
        }
    elif dataset == 'hymenoptera' or dataset == 'RSNA_COVID_512' or dataset == 'RSNA_COVID_1024':
        return {
            # Data augmentation and normalization for training
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # Just normalization for validation
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        raise NotImplementedError('Unknown dataset.')


def load_dataset(dataset_name, dataset_dir='datasets'):
    data_transforms = get_transforms(dataset_name)

    if dataset_name in datasets.__all__:
        DSet = eval('datasets.' + dataset_name)
        return {x: DSet(dataset_dir, train=True if x == 'train' else False, download=True, transform=data_transforms[x]) for x in ['train', 'val']}
    else:
        if isinstance(DATASET_URL[dataset_name], list):
            url = DATASET_URL[dataset_name][0]
        else:
            url = DATASET_URL[dataset_name]
            auth=HTTPBasicAuth(DATASET_URL[dataset_name][1], DATASET_URL[dataset_name][2])

        archive_path = join(dataset_dir, basename(url))
        
        if not isfile(archive_path):
            print('Download dataset...')
            with open(archive_path, 'wb') as f:
                f.write(get(url, auth=auth if auth is not None else None).content)
        
        if not isdir(join('datasets', dataset_name)):
            print('Extract dataset archive...')
            with ZipFile(archive_path, 'r') as z:
                z.extractall(dataset_dir)
        
        with FileLock(expanduser("~/.torch_lock")):
            dsets = {x: datasets.ImageFolder(join(dataset_dir, dataset_name, x), transform=data_transforms[x]) for x in ['train', 'val']}
        return dsets