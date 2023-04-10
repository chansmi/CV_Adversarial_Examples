#Bennett Brain
#Computer Vision project 5
#Part 2, examining the network

import torch
import matplotlib.pyplot as plt
import Part1
import torchvision
import numpy
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os

model = Part1.MyNetwork()
model.load_state_dict(torch.load(os.getcwd() + '/results/model.pth'))
model.eval() #set it to eval mode

weights1 = model.conv1.weight


batch_size_test = 1
test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/.torch/', train=False, download=True, # change from '/files/'
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (data, targets) = next(examples)

with torch.no_grad():
    # This plots the weights themselves
    fig = plt.figure() #plot the weights
    for i in range(10):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(weights1[i][0], interpolation='none')
        plt.title("Filter " + str(i))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    plt.close()

    #This generates the filtered image next to the weights
    im = data[0][0]
    im = im.detach().numpy()

    fig2 = plt.figure()
    for i in range(10):
        knl = weights1[i][0].detach().numpy()
        filIm = cv2.filter2D(src=im,ddepth=-1,kernel=knl)
        
        plt.subplot(5,4,2*i+1)
        plt.tight_layout()
        plt.imshow(weights1[i][0], cmap= "gray", interpolation='none')
        plt.title("Filter " + str(i))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(5,4,2*i+2)
        plt.tight_layout()
        plt.imshow(filIm, cmap = "gray", interpolation='none')
        plt.title("Filter " + str(i))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    plt.close()



