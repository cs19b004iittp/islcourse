import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
import numpy as np
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class cs19b004NN(nn.Module):
  pass
  # ... your code ...
  # ... write init and forward functions appropriately ...
  def __init__(self, input, width, height, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input=input, output=10, kernel_size=(2, 2), padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            input=10, output=20, kernel_size=(2, 2), padding='same')
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=width*height*20, out_features=classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  model = None

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
    classes=0
    channels=0
    width = 0
    height = 0
    for (X, y) in train_data_loader:
        channels = X.shape[1]
        width = X.shape[2]
        height = X.shape[3]
        classes = torch.max(y).item()-torch.min(y).item()+1
        break

    model = cs19b004NN(channels, width, height,classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    # training
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        correct = 0
        for batch, (X, y) in enumerate(train_data_loader):
            X, y = X.to(device), y.to(device)
            ypred = model(X)
            loss = loss_function(ypred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            correct += (ypred.argmax(1) == y).type(torch.float).sum().item()
        print("Epoch:", epoch, "loss:", train_loss)
        print("Epoch", epoch, "accuracy:",
              correct/len(train_data_loader.dataset))

    print('Returning model... (rollnumber: cs19b004)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = None

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  # In addition,
  # Refer to config dict, where learning rate is given, 
  # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
  # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
  # You need to create 2d convoution layers as per specification above in each element
  # You need to add a proper fully connected layer as the last layer
  
  # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
  # HINT: Flatten function can also be used if required
  return model
  
  
  print ('Returning model... (rollnumber: cs19b004)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  
  print ('Returning metrics... (rollnumber: cs19b004)')
  
  return accuracy_val, precision_val, recall_val, f1score_val
