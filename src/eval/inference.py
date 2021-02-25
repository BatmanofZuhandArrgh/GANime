import os
import glob as glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import hiddenlayer as hl
from tqdm import tqdm

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import hiddenlayer as hl
import matplotlib.pyplot as plt
from PIL import Image
torch.manual_seed(0)

def denorm(img_tensors, stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    return img_tensors * stats[1][0] + stats[0][0]

#Image2image inference
def inferA2B(model, input_image, device = 'cuda'):
    test_output = Image.open(input_image).convert('RGB')

    torch_output = transforms.functional.resize(test_output, 64)
    torch_output_cuda = transforms.ToTensor()(torch_output).unsqueeze_(0).float().to(device)
        
    anime_output = model(torch_output_cuda).cpu()
    anime_output = torch.squeeze(anime_output,0) 
    anime_output = transforms.ToPILImage()(anime_output)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(torch_output)
    ax[1].imshow(anime_output)
    plt.show()
    
    return anime_output

def inferA2B2A(modelAB, modelBA, input_image, device = 'cuda'):
    output_image = inferA2B(modelAB, input_image)
    output_image_cuda = transforms.ToTensor()(output_image).unsqueeze_(0).float().to(device)
    
    translated_image = modelBA(output_image_cuda).cpu()
    translated_image = torch.squeeze(translated_image,0) 
    translated_image = transforms.ToPILImage()(translated_image)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(output_image)
    ax[1].imshow(translated_image)
    plt.show()
    return output_image, translated_image