# Bennett Brain and Chandler Smith
# Adversarial Transforms
# credit: this tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import sys
import torch
import torchvision
import matplotlib.pyplot as plt

def TopLeftPixels(data,ind):
    data[ind][0][0][0] = 2.5
    data[ind][0][0][1] = 2.5
    data[ind][0][1][0] = 2.5
    data[ind][0][1][1] = 2.5

def cornerBunches(data,ind):
    data[ind][0][0][0] = 2.5
    data[ind][0][0][1] = 2.5
    data[ind][0][1][0] = 2.5
    data[ind][0][1][1] = 2.5

    data[ind][0][23][0] = 2.5
    data[ind][0][23][1] = 2.5
    data[ind][0][24][0] = 2.5
    data[ind][0][24][1] = 2.5

    data[ind][0][0][23] = 2.5
    data[ind][0][0][24] = 2.5
    data[ind][0][1][23] = 2.5
    data[ind][0][1][24] = 2.5

    data[ind][0][23][23] = 2.5
    data[ind][0][23][24] = 2.5
    data[ind][0][24][23] = 2.5
    data[ind][0][24][24] = 2.5

def WholeImageLine(data,ind):
    for k in range(0,28):
      data[ind][0][k][k] = 2.5 #this seems to correspond to white in the sample data