#Bennett Brain
#Computer Vision project 5
#Part 1.F, loading the network in a separate file and running on small subsets of the test data

import torch
import torchvision
import matplotlib.pyplot as plt
import Part1
import numpy

import os

parent_dir = os.path.realpath(os.path.dirname(__file__))
os.chdir(parent_dir)

#load up the model
model = Part1.MyNetwork()
model.load_state_dict(torch.load(os.getcwd() + '/results/model.pth'))
model.eval() #set it to eval mode

batch_size_test = 10

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/.torch/', train=False, download=True, #'/files/' for windows and '~/.torch/' for mac
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (data, targets) = next(examples)

output = model(data)
outArr = output.detach().numpy() #turn the output tensor into an array

predictions = list()

i = 0
for anOutput in outArr:
    anOutput = numpy.around(anOutput, 2)
    
    print("Model's output vals: ")
    print(anOutput)
    maxInd = numpy.argmax(anOutput)
    print("Index of max label: " + str(maxInd))
    print("Correct Label: {}".format(targets[i]))
    i = i+1
    print("")

    predictions.append(maxInd)


fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: " + str(predictions[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()
plt.close()

