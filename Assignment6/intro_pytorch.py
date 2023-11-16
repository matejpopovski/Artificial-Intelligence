import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# import numpy as np


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test = datasets.FashionMNIST('./data', train=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(train, batch_size=64)
    if not training:
        data_loader = torch.utils.data.DataLoader(test, batch_size=64)
    return data_loader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
   
    for epoch in range(T):
        running_loss = 0.0
        corrects = 0
        all_data = 0
        model.train()
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step() 
            running_loss += loss.item() * 64
            _, predicted = torch.max(predictions.data, 1)
            all_data += labels.size(0)
            corrects += (predicted == labels).sum().item()

        accuracy = 100 * corrects / all_data
        loss = running_loss / all_data
        print(f'Train Epoch: {epoch} Accuracy: {corrects}/{all_data}({accuracy:.2f}%) Loss: {loss:.3f}')

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    running_loss = 0.0
    corrects = 0
    all_data = 0

    with torch.no_grad():
        for data, labels in test_loader:
            predictions = model(data)
            loss = criterion(predictions, labels)
            running_loss += loss.item() * 64
            _, predicted = torch.max(predictions.data, 1)
            all_data += labels.size(0)
            corrects += (predicted == labels).sum().item()

    loss = running_loss / all_data
    accuracy = 100 * corrects / all_data
    if show_loss:
        print(f'Average loss: {loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')
    

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    out = model(test_images)
    probabilities = F.softmax(out, dim=1)
    best_3 = torch.topk(probabilities[index], 3)[0]
    best_3 = best_3.detach().numpy()
    best_3_idx = torch.topk(probabilities[index], 3)
    best_3_idx = best_3_idx.indices.detach().numpy()
    for i in range(3): 
        print(f'{classes[best_3_idx[i]]}: {best_3[i] * 100:.2f}%')


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    train_loader = get_data_loader()
    test_loader = get_data_loader(training=False)
    
    model = build_model()
    
    criterion = nn.CrossEntropyLoss()
    
    train_model(model, train_loader, criterion, 5)
    
    evaluate_model(model, test_loader, criterion, show_loss = True)
    
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)
