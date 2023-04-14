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

# PGD attack
def pgd(network, image, target, epsilon, data_gradient, alpha, num_iterations):
    perturbed_image = image.clone().detach()
    for _ in range(num_iterations):
        perturbed_image.requires_grad = True
        output = network(perturbed_image)
        loss = F.nll_loss(output, target)
        network.zero_grad()
        loss.backward()

        data_gradient = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_gradient.sign()
        perturbation = torch.clamp(perturbed_image - image, -epsilon, epsilon)
        perturbed_image = torch.clamp(image + perturbation, 0, 1).detach()

    return perturbed_image
