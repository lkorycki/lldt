import torch
import torchvision
from torchsummary import summary

from data.tensor_set import extract_features
from learners.models.resnext import create_cifar_resnext
import data.data_collection as data_col


def extract(last):
    print('Running')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # cifar_resnext29.pth.tar -> https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/resnext.py

    extractor = create_cifar_resnext('pytorch_models/cifar_resnext29.pth.tar')
    extractor.classifier = torch.nn.Identity()
    extractor.eval().to(device)
    summary(extractor.to(device), (3, 32, 32))
    dataset = data_col.get('SVHN-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/svhn10-2-train.pt', device=device)
    dataset = data_col.get('SVHN-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/svhn10-2-test.pt', device=device)

    extractor = create_cifar_resnext('pytorch_models/cifar_resnext29.pth.tar')
    extractor.classifier = torch.nn.Identity()
    extractor.eval().to(device)
    summary(extractor.to(device), (3, 32, 32))
    dataset = data_col.get('CIFAR20C-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar20c-train.pt', device=device)
    dataset = data_col.get('CIFAR20C-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/cifar20c-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    fc1 = extractor.fc
    extractor.fc = torch.nn.Sequential(fc1, torch.nn.Linear(1000, 256), torch.nn.ReLU(), torch.nn.Linear(256, 20))
    extractor.load_state_dict(torch.load('pytorch_models/imgnet20a-2f.pth'))
    if not last: extractor.fc = torch.nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('IMAGENET20A-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet20a-train.pt', device=device)
    dataset = data_col.get('IMAGENET20A-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet20a-test.pt', device=device)

    extractor = torchvision.models.resnet18(pretrained=True)
    fc1 = extractor.fc
    extractor.fc = torch.nn.Sequential(fc1, torch.nn.Linear(1000, 256), torch.nn.ReLU(), torch.nn.Linear(256, 20))
    extractor.load_state_dict(torch.load('pytorch_models/imgnet20b-2f.pth'))
    if not last: extractor.fc = torch.nn.Identity()
    extractor.eval().to(device)
    dataset = data_col.get('IMAGENET20B-TRAIN')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet20b-train.pt', device=device)
    dataset = data_col.get('IMAGENET20B-TEST')
    extract_features(dataset, extractor, f'pytorch_data/extracted/imagenet20b-test.pt', device=device)

