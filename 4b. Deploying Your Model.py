import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import MyConvBlock

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')

# SOLUTION
def predict_letter(file_path):
    show_image(file_path)
    image = tv_io.read_image(file_path, tv_io.ImageReadMode.GRAY)
    image = preprocess_trans(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    prediction = output.argmax(dim=1).item()
    # convert prediction to letter
    predicted_letter = alphabet[prediction]
    return predicted_letter

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load('model.pth', map_location=device)
next(model.parameters()).device

# To greyscale and resize
IMG_WIDTH = 28
IMG_HEIGHT = 28
preprocess_trans = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0, 1]
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.Grayscale()  # From Color to Gray
])

# prediction test
image = tv_io.read_image('data/asl_images/b.png', tv_io.ImageReadMode.GRAY)
processed_image = preprocess_trans(image)
batched_image = processed_image.unsqueeze(0)
batched_image_gpu = batched_image.to(device)
output = model(batched_image_gpu)
prediction = output.argmax(dim=1).item()
alphabet = "abcdefghiklmnopqrstuvwxy"
alphabet[prediction]
predict_letter("data/asl_images/b.png")
predict_letter("data/asl_images/a.png")



