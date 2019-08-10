
import pandas as pd
import torch
import numpy as np
from torchvision import datasets, transforms, models, utils
from torch import nn, optim
from collections import OrderedDict
import torchvision
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_directory', type=str, help='Path to dataset ')
parser.add_argument('--save_dir', type=str, help='Save trained model checkpoint to path')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU mode')


args, _ = parser.parse_known_args()
train_image_datasets = ''

def prep_data(folder_dir):
    data_dir = folder_dir +'/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop (224), transforms.RandomHorizontalFlip (), transforms.ToTensor (), transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valid_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                 transforms.CenterCrop (224),
                                                 transforms.ToTensor (),
                                                 transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                 transforms.CenterCrop (224),
                                                 transforms.ToTensor (),
                                                 transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
        # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder (train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder (valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder (test_dir, transform = test_data_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)
    return train_loader, valid_loader, test_loader

def train(epochs=9, learning_rate=0.001, device='cpu', checkpoint='', arch='vgg19', hidden_units=4096):
    train_loader, valid_loader, test_loader = prep_data(args.data_directory)
    if args.arch:
        arch = args.arch    
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)
    
    if args.hidden_units:
        hidden_units = args.hidden_units
    
    classifier = nn.Sequential  (OrderedDict ([
                                ('fc1', nn.Linear (25088, hidden_units)),
                                ('relu1', nn.ReLU ()),
                                ('dropout1', nn.Dropout (p = 0.5)),
                                ('fc2', nn.Linear (hidden_units, 102)),
                                ('output', nn.LogSoftmax (dim =1))
                               ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()


    if args.epochs:
        epochs = args.epochs    
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.gpu:
        device = 'cuda:0'
        model.cuda()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate )
    for param in model.parameters(): 
        param.requires_grad = False
    steps = 0
    print_every = 40
    for e in range(epochs):
        cur_loss = 0
        for i, (inputs, labels) in enumerate(train_loader) :
            steps += 1
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cur_loss += loss.item()
            if steps % print_every == 0:
                valid_loss, accuracy = validate(model, valid_loader, criterion, device)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(cur_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss),
                      "Accuracy: {:.3f}%".format(accuracy))
                cur_loss = 0
    if args.save_dir:
        checkpoint = args.save_dir 
        model.class_to_idx = train_image_datasets.class_to_idx
        checkpoint= {'arch': 'vgg19',
                    'state_dict': model.state_dict(), 
                    'class_to_idx': model.class_to_idx,
                    'input_size': 25088,
                    'output_size': 102,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'optimizer': optimizer.state_dict(),
                    'classifier' : classifier}
        torch.save(checkpoint_dict, checkpoint)
        
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Network accuracy: %d %%' % (100 * correct / total))
    return model

def validate(model, data, criterion, device):
    model.to(device)
    model.eval()
    loss = 0
    accuracy = 0
    for i, (inputs, labels) in enumerate(data):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return (loss/len(data)), (accuracy/len(data))



args, _ = parser.parse_known_args()
if __name__ == '__main__':
    train()
    