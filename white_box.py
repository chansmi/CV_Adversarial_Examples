# Bennett Brain and Chandler Smith
# Credit: Referenced the tutorial at https://nextjournal.com/gkoehler/pytorch-mnist 
# Credit: used the following for network visualization: https://github.com/szagoruyko/pytorchviz 
# Necessary installs: Torchvision, Pytorch, torchviz, and graphviz
# White-box attacks, where the adversary is aware of network structure/weights/etc

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
from fgsm import fgsm
from pgd import pgd


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
            torch.save(network.state_dict(), os.getcwd() + '/results/' + model_name + '.pth') #Dynamic Model Name
            torch.save(optimizer.state_dict(), os.getcwd() + '/results/' + model_name + 'optimizer.pth')
    return



# --- FGSM TEST NETWORK --- #
def fgsm_test_network(network, test_losses, test_loader, epsilon):
    network.eval()
    test_loss = 0
    correct = 0
    perturbed_data_list = []
    true_labels_list = []

    for data, target in test_loader:
        data.requires_grad = True
        output = network(data)
        loss = F.nll_loss(output, target)
        network.zero_grad()
        loss.backward()

        # Apply FGSM attack
        data_grad = data.grad.data
        perturbed_data = fgsm(data, epsilon, data_grad)

        # for FGSM Visualization
        for i in range(perturbed_data.size(0)):
            perturbed_data_list.append(perturbed_data[i])
            true_labels_list.append(target[i].unsqueeze(0))

        # Test network with perturbed data
        with torch.no_grad():
            output = network(perturbed_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset), perturbed_data_list, true_labels_list

# --- PGD TEST NETWORK --- #
def pgd_test_network(network, test_loader, epsilon, alpha, num_iterations):
    network.eval()
    test_loss = 0
    correct = 0
    perturbed_data_list = []
    true_labels_list = []
    test_losses = []  # Define test_losses within the function

    for data, target in test_loader:
        data.requires_grad = True
        output = network(data)
        loss = F.nll_loss(output, target)
        network.zero_grad()
        loss.backward()

        # Apply PGD attack
        data_grad = data.grad.data
        perturbed_data = pgd(network, data, target, epsilon, data_grad, alpha, num_iterations)

        # for PGD Visualization
        for i in range(perturbed_data.size(0)):
            perturbed_data_list.append(perturbed_data[i])
            true_labels_list.append(target[i].unsqueeze(0))

        # Test network with perturbed data
        with torch.no_grad():
            output = network(perturbed_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset), perturbed_data_list, true_labels_list



# --- MAIN FUNCTION --- #
# main function (yes, it needs a comment too)
def main(argv):
    # Change to the directory of this script
    parent_dir = os.path.realpath(os.path.dirname(__file__))
    os.chdir(parent_dir)

    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    model_name = 'model_whitebox'
    # for visualization purposes
    epsilon_losses = []
    epsilon_accuracies = []
    # pgd variables
    alpha = 0.01
    num_iterations = 40

    random_seed = 532 #pick a number but be consistent about it
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/.torch/', train=True, download=True, # change from '/files/'
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/.torch/', train=False, download=True,  # change from '/files/'
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    # uncomment for ground truth
    '''fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()
    plt.close()
    '''

    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    n_epochs = 5

    

    # Train the network for n_epochs --- UNCOMMENT IF NETWORK NOT ALREADY TRAINED
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, optimizer, train_losses, train_counter, log_interval, train_loader, model_name)

    # --- TOGGLE NETWORK HERE --- 
    # attack_type = 'pgd'  # 'fgsm' or 'pgd'
    attack_type = 'fgsm'

    if attack_type == 'fgsm':
        test_function = fgsm_test_network
    elif attack_type == 'pgd':
        test_function = pgd_test_network
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    def plot_images(perturbed_data_list, true_labels_list, epsilon, num_images_to_plot=6):
        fig, axes = plt.subplots(1, num_images_to_plot, figsize=(12, 2))
        for i in range(num_images_to_plot):
            img = perturbed_data_list[i].squeeze().detach().numpy()
            true_label = true_labels_list[i].item()

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"True Label: {true_label}")
            axes[i].axis('off')

        plt.suptitle(f"Epsilon = {epsilon}")
        plt.tight_layout()
        plt.show()

    # Test the trained network using FGSM or PGD attacks with different epsilon values
    epsilons = [0, .05, .1, .15, .2, .25, .3, 0.5, 0.75]
    for epsilon in epsilons:
        print(f"Testing with epsilon = {epsilon}")
        if attack_type == 'fgsm':
            test_loss, test_accuracy, perturbed_data_list, true_labels_list = test_function(network, test_losses, test_loader, epsilon)
        elif attack_type == 'pgd':
            test_loss, test_accuracy, perturbed_data_list, true_labels_list = test_function(network, test_loader, epsilon, alpha, num_iterations)
        epsilon_losses.append(test_loss)
        epsilon_accuracies.append(test_accuracy)

        # Display the perturbed images after each epsilon iteration
        plot_images(perturbed_data_list, true_labels_list, epsilon)


    # Display the perturbed images after each epsilon iteration
    #plot_images(perturbed_data_list, true_labels_list, epsilon)

    # Display the perturbed images for the last epsilon value
    num_images_to_plot = 6
    fig, axes = plt.subplots(1, num_images_to_plot, figsize=(12, 2))

    for i in range(num_images_to_plot):
        img = perturbed_data_list[i].squeeze().detach().numpy()
        true_label = true_labels_list[i].item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True Label: {true_label}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Print FGSM accuracy results results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(epsilons, epsilon_losses, marker='o')
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('Test Loss vs Epsilon')
    fig.suptitle(f'{attack_type.upper()} Attack Performance')

    ax2.plot(epsilons, epsilon_accuracies, marker='o')
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy vs Epsilon')

    plt.tight_layout()
    plt.show()

    return

    

if __name__ == "__main__":
    main(sys.argv)