# Chandler Smith
# FGSM Attack for a white box.py model

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


# defines the level or perturbation - higher = more visible
epsilons = [0, .05, .1, .15, .2, .25, .3]
# reference to the pretrained model
pretrained_model = "model_whitebox.pth"

# perturbed image = image + epsilon * sign(gradient of the loss)
def fgsm(image, epsilon, data_gradient):
        #sign of the gradient of the loss
        sdg = data_gradient.sign()
        perturbed_image = image+epsilon*sdg
        # clip to maintain range of data
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
