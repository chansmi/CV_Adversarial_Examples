# Bennett Brain and Chandler Smith
# Credit: Referenced the tutorial at https://nextjournal.com/gkoehler/pytorch-mnist 
# Credit: used the following for network visualization: https://github.com/szagoruyko/pytorchviz 
# Necessary installs: Torchvision, Pytorch, torchviz, and graphviz


# Black-box attacks, where the adversary is only aware of the images coming in and the target labels, not anything else.  
# Run at training time to attempt to poorly-train the network.


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
import adversarial_transforms as at


# class definitions
class MyNetwork(nn.Module):
    #Initialize the layers
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #first convolution layer, goes from 1 channel (gscl) to 10 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) #second convolution layer, goes from 10 channels (output of first) to 20 channels, 5x5 kernel
        self.conv2_drop = nn.Dropout2d() #Dropout layer, 50% is default
        self.fc1 = nn.Linear(320, 50) #Fully Connected linear layer, comes from 320 entries and goes to 50
        self.fc2 = nn.Linear(50, 10) #Second FC linear layer, comes from 50 entries and goes to the final 10

    # computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #Take first conv layer, maxpool from there
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) #Take second conv and then the dropout from that, maxpool, and then activate with ReLu
        x = x.view(-1, 320) #Swap into a linear setup
        x = F.relu(self.fc1(x)) #ReLu on the first fully connected layer
        #x = F.dropout(x, training=self.training) Was in tutorial, but canvas instructions don't ask for a second dropout layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) #Softmax on the second FC layer (output)

# The main training function, which contains the adversarial attack at the start of each batch.
def train_network( epoch, network, optimizer, train_losses, train_counter, log_interval, train_loader, model_name, adv_targets ):

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        #The adversarial attack occurs here
        for i in range(len(target)):
            #specific attacks on given digits
            # if (target[i] in adv_targets):
            #     #at.TopLeftPixels(data,i)
            #     #at.diagImageLine(data,i)
            #     #at.leftImageLine(data,i)
            #     #at.cornerBunches(data,i)
            #     at.projectPattern(data,i)
            
            #random attacks
            randInt = np.random.randint(2) #the number in randint() is inverse frequency of samples; i.e. 4 => 1/4 of samples
            if (randInt == 0): #1/n chance
                #at.TopLeftPixels(data,i)
                #at.diagImageLine(data,i)
                at.leftImageLine(data,i)
                #at.cornerBunches(data,i)
                #at.projectPattern(data,i)

        #The rest of the training is the same as in project 5
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
            torch.save(network.state_dict(), os.getcwd() + '/results/' + model_name + '.pth') #Dynamic Model Name
            torch.save(optimizer.state_dict(), os.getcwd() + '/results/' + model_name + 'optimizer.pth')
    return

# useful functions with a comment for each function
def test_network(network, test_losses, test_loader):

    testHist = [0]*10
    testHistMax = [0]*10
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

            for i in range(len(target)):
                testHistMax[target[i]] += 1
                if pred[i] == target[i]:
                    testHist[target[i]] += 1
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return (testHist, testHistMax)
# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv
    # main function code
    # Change to the directory of this script
    parent_dir = os.path.realpath(os.path.dirname(__file__))
    os.chdir(parent_dir)

    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    model_name = 'model_blackbox'

    adv_targets = [1,7] #the target digits to attack

    random_seed = 532 #pick a number but be consistent about it
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        #Updated file path for mac
        torchvision.datasets.MNIST('~/.torch/', train=True, download=True, # change from '/files/'
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        #Updated file path for mac
        torchvision.datasets.MNIST('~/.torch/', train=False, download=True,  # change from '/files/'
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)


    # This mimics the adversarial attack so that we can visualize what they look like.
    for i in range(len(example_targets)):
            if (example_targets[i] in adv_targets):
                #at.TopLeftPixels(example_data,i)
                #at.diagImageLine(example_data,i)
                #at.leftImageLine(example_data,i)
                #at.cornerBunches(example_data,i)
                at.projectPattern(example_data,i)

    print(example_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    plt.close()

    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


    #visualize network
    # model = MyNetwork()
    # yhat = model(example_data) # Give dummy batch to forward(), got this from the earlier plot.
    # make_dot(yhat, params=dict(list(model.named_parameters()))).render("MNIST_CNN", format="png")

    test_network(network,test_losses,test_loader)
    for epoch in range(1, n_epochs + 1):
        train_network(epoch,network,optimizer,train_losses,train_counter,log_interval,train_loader,model_name, adv_targets)
        
        (testHist, testHistMax) = test_network(network,test_losses,test_loader)

    testHistPct = [0]*10
    for i in range (0,10):
        testHistPct[i] = 100 * testHist[i]/testHistMax[i] 

    fig2 = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    plt.close()


    #delete this after
    barWidth = .25
    
    br1 = np.arange(len(testHist))
    br2 = [x + barWidth for x in br1]   
    fig3 = plt.figure()
    plt.bar(br1, testHist, color ='c', width = barWidth,
        edgecolor ='grey', label ='Number Correct')
    plt.bar(br2, testHistMax, color ='b', width = barWidth,
        edgecolor ='grey', label ='Total Examples')
    plt.xticks([r + barWidth for r in range(len(testHist))],
        ['0','1','2','3','4','5','6','7','8','9'])
    plt.legend(['Number Correct', 'Total Examples'], loc='upper right')
    plt.xlabel('Digit')
    plt.ylabel('number of times it appears')
    plt.show()
    plt.close()

    fig4 = plt.figure()
    plt.bar(br1, testHistPct, color ='b', width = barWidth*3,
        edgecolor ='grey')
    plt.xticks([r + barWidth for r in range(len(testHist))],
        ['0','1','2','3','4','5','6','7','8','9'])
    plt.xlabel('Digit')
    plt.ylabel('Percent Accuracy')
    plt.show()
    plt.close()
    

    return

    

if __name__ == "__main__":
    main(sys.argv)