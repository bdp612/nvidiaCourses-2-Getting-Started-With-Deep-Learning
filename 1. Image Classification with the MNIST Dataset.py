# Load Libraries
from random import randint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

# Train Function
def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

# Validation Function
def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# Load Train and Valid Datasets
train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)
train_set_length = len(train_set)
x_0, y_0 = train_set[randint(0, train_set_length - 1)]
x_0.show()

# Convert Into Tensors
trans = transforms.Compose([transforms.ToTensor()])
x_0_tensor = trans(x_0)

# Move To GPU
x_0_gpu = x_0_tensor.cuda()
x_0_tensor.to(device).device

# Back to PIL
image = F.to_pil_image(x_0_tensor)
plt.imshow(image, cmap='gray')

# Transforms & Dataloaders
trans = transforms.Compose([transforms.ToTensor()])
train_set.transform = trans
valid_set.transform = trans
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

# Creating the Model -- Flatten
layers = []
layers
test_matrix = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)
nn.Flatten()(test_matrix)
batch_test_matrix = test_matrix[None, :]
nn.Flatten()(batch_test_matrix)
layers = [
    nn.Flatten()
]

# Input Layer
input_size = 1 * 28 * 28
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
]

# Hidden Layer
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU()  # Activation for hidden
]

# Output Layer
n_classes = 10
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(),  # Activation for hidden
    nn.Linear(512, n_classes)  # Output
]

# Compile The Model
model = nn.Sequential(*layers)
model.to(device)
next(model.parameters()).device
model = torch.compile(model)

# Train The Model
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

# Get Lens
train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

# Training Loop
epochs = 5
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

# Test
prediction = model(x_0_gpu)
print("The prediction is: " + str(prediction.argmax(dim=1, keepdim=True).item()))
print("The actual answer is: " + str(y_0))

