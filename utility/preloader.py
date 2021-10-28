import torch.nn as nn
from torchvision import datasets, transforms, models

def get_model(model, num_class):
    if model == 'MNIST':
        from models.mnist import Net
        return Net(num_class)
    elif model == 'resnet18':
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
    else:
        raise NotImplementedError('Unknown dataset.')


def load_dataset(dataset, dataset_dir='datasets'):
    data_transforms = get_transforms(dataset)

    if dataset in datasets.__all__:
        DSet = eval('datasets.' + dataset)
        return {x: DSet(dataset_dir, train=True if x == 'train' else False, download=True, transform=data_transforms[x]) for x in ['train', 'val']}
    else:
        pass