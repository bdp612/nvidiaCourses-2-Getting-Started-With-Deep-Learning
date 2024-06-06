# Import necessary libraries
from random import randint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms.v2 as transforms

# Bool for turning demo on/off
demo = True

# Training function definition
def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # Move data to GPU if available
        output = model(x)  # Forward pass
        optimizer.zero_grad()  # Clear previous gradients
        batch_loss = loss_function(output, y)  # Compute loss
        batch_loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        loss += batch_loss.item()  # Accumulate loss
        accuracy += get_batch_accuracy(output, y, train_N)  # Accumulate accuracy
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# Validation function definition
def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():  # Disable gradient calculation for validation
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)  # Move data to GPU if available
            output = model(x)  # Forward pass

            loss += loss_function(output, y).item()  # Accumulate loss
            accuracy += get_batch_accuracy(output, y, valid_N)  # Accumulate accuracy
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# Function to calculate batch accuracy
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)  # Get predictions
    correct = pred.eq(y.view_as(pred)).sum().item()  # Count correct predictions
    return correct / N  # Return accuracy

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training and validation datasets
train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)

# Define list of transformations
transformations = transforms.Compose([transforms.ToTensor()])

# Demo
if demo:
    # Gather a random image from the training set for later testing
    x_0, y_0 = train_set[randint(0, len(train_set) - 1)]
    x_0.show()

    # Convert the random image to tensor and move it to GPU
    x_0_tensor = transformations(x_0)
    x_0_gpu = x_0_tensor.cuda()
    x_0_tensor.to(device).device

# Prepare the datasets with the list of transformation instructions
train_set.transform = transformations
valid_set.transform = transformations

# Define data loaders with batch size and shuffling
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

# Define the neural network layers
input_size = 1 * 28 * 28
n_classes = 10
layers = [
    nn.Flatten(),  # Flatten the input
    nn.Linear(input_size, 512),  # Input layer, reducing to 512
    nn.ReLU(),  # Activation function
    nn.Linear(512, 512),  # Hidden layer
    nn.ReLU(),  # Activation function
    nn.Linear(512, n_classes)  # Output layer, reducing to n_classes
]

# Create the model and move it to the device
model = nn.Sequential(*layers)
model.to(device)
next(model.parameters()).device  # Ensure model is on the correct device
model = torch.compile(model)  # Compile the model for optimized execution

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Get dataset sizes
train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

# Training loop for a specified number of epochs
epochs = 5
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()  # Train the model
    validate()  # Validate the model

# Demo
if demo:
    # Make a prediction on a single example
    prediction = model(x_0_gpu)
    print("The prediction is: " + str(prediction.argmax(dim=1, keepdim=True).item()))
    print("The actual answer is: " + str(y_0))
