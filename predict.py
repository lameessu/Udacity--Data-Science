import pandas as pd
import torch
import numpy as np
from torchvision import datasets, transforms, models, utils
from torch import nn, optim
from collections import OrderedDict
import torchvision
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to image ')
parser.add_argument('checkpoint', type=str, help='Save trained model checkpoint to path')
parser.add_argument('--top_k', type=int, help='Top K most likely classes')
parser.add_argument('--category_names', type=str, help='category names mapping file')
parser.add_argument('--gpu', action='store_true', help='Use GPU mode')


args, _ = parser.parse_known_args()

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    learning_rate = checkpoint['learning_rate']
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    off = 0.5*(256-224)
    img = Image.open(image)##.resize((256,256)).crop(off,off,256-off,256-off)
    img_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    return img_transform(img)


def predict(topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    if args.gpu:
       device = 'cuda:0'
       model.cuda()
    else:
        device = 'cpu'
    model = load_checkpoint(args.checkpoint)
    
    if args.top_k:
        topk = top_k
    img = process_image(args.path).unsqueeze(0)
    
    output = torch.exp(model.forward(img)).topk(topk)
    
    probs = output[0][0].tolist()
    labels = output[1][0].tolist()
    

    ## get indecies 
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    #for i in labels:
    #    print (i)
    
    ##classes = []
    classes = [idx_to_class[i] for i in labels]
    print('Prediction',probs)
    print('Class',classes)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        labels = [cat_to_name[i] for i in classes]
        print('Label',classes)
        
        
    
    return probs, classes


if __name__ == '__main__':
    p , c = predict()
