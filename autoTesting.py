# Bennett Brain
# CV project 5- MNISTFASHION Machine Learning
# part 4 auto testing
# Cite: MNISTFASHION labels, https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide

''' //////////// Testing Doubling the number of epochs ////////////'''

# import statements
import sys
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchviz import make_dot

# class definitions
class MyNetwork(nn.Module):
    #Initialize the layers
    def __init__(self, convSize, dropRate, nConv):
        super(MyNetwork, self).__init__()
        self.nConv = nConv
        self.conv1 = nn.Conv2d(1, 10, kernel_size=convSize, padding= 'same') #first convolution layer, goes from 1 channel (gscl) to 10 channels, padding size "same" used to make automation possible
        self.conv2 = nn.Conv2d(10, 20, kernel_size=convSize, padding = 'same') #second convolution layer, goes from 10 channels (output of first) to 20 channels
        self.conv3 = nn.Conv2d(20, 40, kernel_size=convSize, padding = 'same') #etc, up to 5 layers
        self.conv4 = nn.Conv2d(40, 80, kernel_size=convSize, padding = 'same')
        self.conv5 = nn.Conv2d(80, 160, kernel_size=convSize, padding = 'same')
        self.conv_drop = nn.Dropout2d(p = dropRate) #Dropout layer, 50% is default


        self.fc1 = nn.Linear(1960, 50) #Fully Connected linear layer from one conv, comes from 1960 entries and goes to 50

        self.fc2 = nn.Linear (980, 50) #Fully Connected linear layer from two conv

        self.fc3 = nn.Linear (360, 50) #Fully Connected linear layer from three conv

        self.fc4 = nn.Linear (720, 50) #Fully Connected linear layer from four conv (bigger because no more maxpool)

        self.fc5 = nn.Linear (1440, 50) #Fully Connected linear layer from four conv (bigger because no more maxpool)

        self.fcfinal = nn.Linear(50, 10) #final FC linear, goes from the resulting 50 down to 10 (output)

    # computes a forward pass for the network
    def forward(self, x):

        nConv = self.nConv

        if (nConv == 1):
            x = F.relu(F.max_pool2d(self.conv_drop(self.conv1(x)), 2)) #Take first conv layer, maxpool from there
            x = x.view(-1, 1960)
            x = F.relu(self.fc1(x))
        elif (nConv == 2):
            x = F.relu(F.max_pool2d(self.conv1(x), 2)) #Take first conv layer, maxpool from there
            x = F.relu(F.max_pool2d(self.conv_drop(self.conv2(x)), 2))
            x = x.view(-1, 980)
            x = F.relu(self.fc2(x))

        elif (nConv == 3):
            x = F.relu(F.max_pool2d(self.conv1(x), 2)) #Take first conv layer, maxpool from there
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2))

            x = x.view(-1, 360)
            x = F.relu(self.fc3(x))

        elif (nConv == 4):
            x = F.relu(F.max_pool2d(self.conv1(x), 2)) #Take first conv layer, maxpool from there
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(self.conv3(x)) #No more maxpools or else our data becomes too small
            x = F.relu(F.max_pool2d(self.conv_drop(self.conv4(x)), 2))

            x = x.view(-1, 720)
            x = F.relu(self.fc4(x))
        elif (nConv == 5):
            x = F.relu(F.max_pool2d(self.conv1(x), 2)) #Take first conv layer, maxpool from there
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(self.conv3(x)) #No more maxpools or else our data becomes too small
            x = F.relu(self.conv4(x))
            x = F.relu(F.max_pool2d(self.conv_drop(self.conv5(x)), 2))

            x = x.view(-1, 1440)
            x = F.relu(self.fc5(x))

        
        x = self.fcfinal(x)
        return F.log_softmax(x, dim=1) #Softmax on the second FC layer (output)

# useful functions with a comment for each function
def train_network( epoch, network, optimizer, train_losses, train_counter, log_interval, train_loader, model_name ):

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            # torch.save(network.state_dict(), os.getcwd() + '/results/' + model_name + '.pth') #Dynamic Model Name
            # torch.save(optimizer.state_dict(), os.getcwd() + '/results/' + model_name + 'optimizer.pth')
    return

# useful functions with a comment for each function
def test_network(network, test_losses, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# main function (yes, it needs a comment too)
def main(argv):
    # Change to the directory of this script
    parent_dir = os.path.realpath(os.path.dirname(__file__))
    os.chdir(parent_dir)

    # MNIST dictionary from Kaggle
    fashion_labels = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot',
    }

    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    model_name = 'model'

    #number of convolution layers, must be in range 1-5
    nC = 1
    #drop rate, must be in range 0 to 1
    dR = .75
    #convolution size, must be an odd number between 3 and 7
    cS = 3

    random_seed = 532 #pick a number but be consistent about it
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        #Updated file path for mac
        torchvision.datasets.FashionMNIST('~/.torch/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        #Updated file path for mac
        torchvision.datasets.FashionMNIST('~/.torch/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    '''fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(fashion_labels[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    plt.close()
    '''
    network = MyNetwork(convSize=cS,dropRate=dR, nConv=nC)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


    test_network(network,test_losses,test_loader)
    for epoch in range(1, n_epochs + 1):
        train_network(epoch,network,optimizer,train_losses,train_counter,log_interval,train_loader,model_name)
        
        test_network(network,test_losses,test_loader)

    # fig2 = plt.figure()
    # plt.plot(train_counter, train_losses, color='blue')
    # plt.scatter(test_counter, test_losses, color='red')
    # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.xlabel('number of training examples seen')
    # plt.ylabel('negative log likelihood loss')
    # plt.show()
    # plt.close()

    return

    

if __name__ == "__main__":
    main(sys.argv)