import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

## Data location
data_dir = Path('./input')

## Training
train_dir = data_dir / 'train'

train_pneumonia = [['input/train/PNEUMONIA/' + i, 1]  for i in os.listdir(train_dir / 'PNEUMONIA') if i.endswith(".jpeg")]
train_healthy   = [['input/train/NORMAL/' + i, 0]     for i in os.listdir(train_dir / 'NORMAL') if i.endswith(".jpeg")]
train_pneumonia.extend(train_healthy)

df_train = pd.DataFrame(train_pneumonia, columns = ['img', 'label'], index=None)
# Shuffle the data
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

## Validation
val_dir = data_dir / 'val'

val_pneumonia = [['input/val/PNEUMONIA/' + i, 1]  for i in os.listdir(val_dir / 'PNEUMONIA') if i.endswith(".jpeg")]
val_healthy   = [['input/val/NORMAL/' + i, 0]     for i in os.listdir(val_dir / 'NORMAL') if i.endswith(".jpeg")]
val_pneumonia.extend(val_healthy)

df_val = pd.DataFrame(val_pneumonia, columns = ['img', 'label'], index=None)
# Shuffle the data
df_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True)

## Test
test_dir = data_dir / 'test'

test_pneumonia = [['input/test/PNEUMONIA/' + i, 1]  for i in os.listdir(test_dir / 'PNEUMONIA') if i.endswith(".jpeg")]
test_healthy   = [['input/test/NORMAL/' + i, 0]     for i in os.listdir(test_dir / 'NORMAL') if i.endswith(".jpeg")]
test_pneumonia.extend(test_healthy)

df_test = pd.DataFrame(test_pneumonia, columns = ['img', 'label'], index=None)
# Shuffle the data
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

train_counts = df_train['label'].value_counts()
val_counts = df_val['label'].value_counts()
test_counts = df_test['label'].value_counts()
print(train_counts, val_counts, test_counts)


device = torch.device('cuda')

class CNN(nn.Module):
    """ CNN class 
        Attributes:
        convolution (nn.Sequential): the cnn
        hidden_size (int): Size of the hidden layers
        pixel_size (int): Size of the pixel space
        output_size (int): Size of the output space
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.convolution = nn.Sequential(
            # original input size: 200x200
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 100 x 100 

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # size: 50 x 50
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * (200//16) * (200//16), 256), # 64 x 50 x 50
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.convolution(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

EPOCHS = 10
BATCH_SIZE = 32
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Assuming the image location is in the first column
        label = self.data.iloc[idx, 1]     # Assuming the label is in the second column
        
        image = Image.open(img_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

cnn = CNN().to(device)
# Define transformations (resize, convert to tensor)
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
])

# Create custom dataset
train = CustomDataset(dataframe=df_train, transform=transform)
val = CustomDataset(dataframe=df_val, transform=transform)
test = CustomDataset(dataframe=df_test, transform=transform)

# Create data loader
train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn.parameters())

print( 'heyooooo')
## TRAINING LOOP
for epoch in range(EPOCHS):
    # TRAINING PHASE
    cnn.train() # set training mode
    train_loss = 0.0
    train_acc = 0.0
    
    for inputs, labels in train_loader:
        inputs.to(device)
        labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = cnn(inputs)
        # loss calculation
        loss = criterion(outputs, labels.unsqueeze(1).float())
        # backward pass
        loss.backward()
        # weight adjustment
        optimizer.step()
        
        # accuracy
        pred = outputs.round()
        correct = pred.eq(labels.view_as(pred)).sum().item()

        # metrics
        train_acc += correct / labels.numel()
        train_loss += loss.item()

    ## VALIDATION PHASE
    cnn.eval() # set evalidation mode
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad(): # to prevent the gradient computation, as we are not doing anything with it during validation
        for inputs, labels in val_loader:
            inputs.to(device)
            labels.to(device)
             # forward pass
            outputs = cnn(inputs)
            # loss calculation
            loss = criterion(outputs, labels.unsqueeze(1).float())
            # no backward pass and optimizer step because no training

            # accuracy
            pred = outputs.round()
            correct = pred.eq(labels.view_as(pred)).sum().item()

            # metrics
            val_acc += correct / labels.numel()
            val_loss += loss.item()

    # epoch metrics
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = val_acc / len(val_loader)
    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = train_acc / len(train_loader)
    print(f"Epoch {epoch + 1} | Train: Loss: {epoch_train_loss:.4f},  Acc: {epoch_train_acc:.4f} | Val: Loss: {epoch_val_loss:.4f},  Acc: {epoch_val_acc:.4f}")
    
# TESTING
cnn.eval()
test_loss = 0.0
test_acc = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs.to(device)
        labels.to(device)
        
        # forward pass
        outputs = cnn(inputs)
        # loss calculation
        loss = criterion(outputs, labels.unsqueeze(1).float())
        # no backward pass and optimizer step because no training

        # accuracy
        pred = outputs.round()
        correct = pred.eq(labels.view_as(pred)).sum().item()

        # metrics
        test_acc += correct / labels.numel()
        test_loss += loss.item()
avg_loss = test_loss / len(test_loader)
avg_acc = test_acc / len(test_loader)
print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {avg_acc:.4f}" )
