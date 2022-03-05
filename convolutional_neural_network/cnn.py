from multiprocessing.dummy import freeze_support
from turtle import forward
from matplotlib.transforms import Transform
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import traceback as tb

freeze_support()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 50
batch_size = 4
learning_rate = 0.001

# Dataset of PIL images in range [0, 1]
# Transform to Tensors in normalized range [-1, 1]
transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def imshow(image):
    image = image / 2 + 0.5
    npimage = image.numpy()
    plt.imshow(np.transpose(npimage, (1, 2, 0)))
    plt.show()
    plt.close()

# get random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))

class ConvNet(nn.Module):
    '''
        CNN Layers: 
        * Input -> 
        * (Convolution + Relu -> Pooling -> Convolution + Relu -> Pooling) [Feature Learning] ->
        * (Flatten -> Fully Connected -> Softmax) [Classification]
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.n_classes = len(classes)      
        # 3 color channels, 6 output channels, 5 kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # Input layers = output layers from conv1
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # Fully connected Layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_classes) # 10 different classes

        # (W - F + 2P ) / S + 1
    def forward(self, x):
        # activation function applied on first convolution layer, then pooling applied to the result
        x = self.pool(F.relu(self.conv1(x))) # n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x))) # n, 16, 5, 5
        # batches = max batces
        # x = x.view(batch_size, 16 * 5 * 5) # n, 400
        x = torch.flatten(x, 1) # flatten all
        # apply activation function on fully connected layer
        x = F.relu(self.fc1(x)) # n, 120
        x = F.relu(self.fc2(x)) # N, 84
        x = self.fc3(x) # n, 10
        return x


def main():
    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # original shape: [4, 3, 32, 32] = 4, 3, 1024
            # 3 inputs, 6 outputs, 5 kernel

            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999: # print every 2000 mini batches
                print(f'[{epoch + 1} / {num_epochs:3d}, {i + 1:5d}] loss: {running_loss / 2000: .3f}')
                running_loss = 0.0

    print('Finished Training')
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect correct and total predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    #print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        tb.print_exc()
        sys.exit(0)