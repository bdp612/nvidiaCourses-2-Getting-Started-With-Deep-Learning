import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io
import glob
from PIL import Image
import utils
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.imgs = []
        self.labels = []

        for l_idx, label in enumerate(DATA_LABELS):
            data_paths = glob.glob(data_dir + label + '/*.png', recursive=True)
            for path in data_paths:
                img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
                self.imgs.append(pre_trans(img).to(device))
                self.labels.append(torch.tensor(l_idx).to(device))

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weights
weights = VGG16_Weights.FIXME
vgg_model = vgg16(weights=weights)

# Freeze base model
vgg_model.requires_grad_(FIXME)
next(iter(vgg_model.parameters())).requires_grad

# Add layers to model
vgg_model.classifier[0:3]
N_CLASSES = FIXME
my_model = nn.Sequential(
    vgg_model.features,
    vgg_model.avgpool,
    nn.Flatten(),
    vgg_model.classifier[0:3],
    nn.Linear(4096, 500),
    nn.ReLU(),
    nn.Linear(500, N_CLASSES)
)

# Compile
loss_function = nn.FIXME()
optimizer = Adam(my_model.parameters())
my_model = torch.compile(my_model.to(device))

# Data transforms
pre_trans = weights.transforms()
IMG_WIDTH, IMG_HEIGHT = (224, 224)
random_trans = transforms.Compose([
    FIXME
])

# Load Dataset
n = FIXME
DATA_LABELS = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]
train_path = "data/fruits/train/"
train_data = MyDataset(train_path)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)
valid_path = "data/fruits/valid/"
valid_data = MyDataset(valid_path)
valid_loader = DataLoader(valid_data, batch_size=n, shuffle=False)
valid_N = len(valid_loader.dataset)

# Training
epochs = 10
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
    utils.validate(my_model, valid_loader, valid_N, loss_function)

# Unfreeze the base model
vgg_model.requires_grad_(FIXME)
optimizer = Adam(my_model.parameters(), lr=.0001)

# Fine Tuning
epochs = 1
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
    utils.validate(my_model, valid_loader, valid_N, loss_function)

# Evaluation
utils.validate(my_model, valid_loader, valid_N, loss_function)

