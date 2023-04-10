#Bennett Brain
#Computer Vision project 5
#Part 1.G, running the network on new data

import torch
import matplotlib.pyplot as plt
import Part1
import numpy
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os

parent_dir = os.path.realpath(os.path.dirname(__file__))
os.chdir(parent_dir)


imlist = list() #For just 10 images, an image list should be ok instead of a data loader
dataArr = list() #same for tensors

#load up the model
model = Part1.MyNetwork()
model.load_state_dict(torch.load(os.getcwd() + '/results/model.pth'))
model.eval() #set it to eval mode

# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

for i in range(0,10): #note: images are already 28x28, I did that resizing manually
    img_gray=cv2.imread(os.getcwd() + "/newData/" + str(i) + ".jpeg") #load the data
	
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY) #convert to proper greyscale
    imagem = cv2.bitwise_not(img_gray) #invert

    mask = cv2.inRange(imagem, 100,255) #make the background properly black using a mask instead of just dark-grey

    imagem[mask==0] =[0] #set background pixels to fully black


    # Convert the image to PyTorch tensor
    imTens = transform(imagem)

    imlist.append(imagem)
    dataArr.append(imTens)

pred = list()
for i in range(0,10): #loop through and get predictions
    out = model(dataArr[i])
    anOutput = out.detach().numpy()
    print("Model's output vals: ")
    print(anOutput)
    maxInd = numpy.argmax(anOutput)
    print("Index of max label: " + str(maxInd))
    print("Correct Label: {}".format(i))
    i = i+1
    print("")
    pred.append(maxInd)

fig = plt.figure() #plot the picture against the prediction
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(imlist[i], cmap='gray', interpolation='none')
    plt.title("Prediction: " + str(pred[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()
plt.close()
