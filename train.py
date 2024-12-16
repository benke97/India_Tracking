from models import UNet
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import HDF5Dataset
import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import gc
import h5py
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_file = 'dataset_3/set280624.h5'
batch_size = 64  

# Determine total number of images
with h5py.File(data_file, 'r') as h5data:
    num_images = h5data['images'].shape[0]

# Split indices for train and validation
indices = np.arange(num_images)
np.random.shuffle(indices)
train_indices = indices[:int(0.8 * len(indices))]
val_indices = indices[int(0.8 * len(indices)):]

# Create datasets
train_dataset = HDF5Dataset(data_file, train_indices, preload_fraction=1)
val_dataset = HDF5Dataset(data_file, val_indices, preload_fraction=1)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = UNet(1,1).to(device)


optimizer = optim.Adam(model.parameters(), lr=0.0008)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def criterion(output, gt, epoch):

    scale_map = torch.ones_like(gt)
    if epoch < 10:
        scale_map[gt == -1] = 10
        scale_map[gt == 1] = 10

    loss = torch.mean((output - gt)**2 * scale_map)
    return loss

num_epochs = 100
best_loss = 1000
train_losses = []
val_losses = []

torch.cuda.empty_cache()
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    gc.collect()

    loader_train = tqdm.tqdm(train_loader, total=len(train_loader))

    train_loss = 0
    for image, gt in loader_train:
        image = image.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, gt, epoch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss)
    

    torch.cuda.empty_cache()
    gc.collect()
    
    
    loader_val = tqdm.tqdm(val_loader, total=len(val_loader))
    
    val_loss = 0
    with torch.no_grad():
        for image, gt in loader_val:
            image = image.to(device)
            gt = gt.to(device)

            output = model(image)
            loss = criterion(output, gt, epoch) 
            val_loss += loss.item()
        val_losses.append(val_loss)

    if val_loss/len(loader_val) < best_loss:
        best_loss = val_loss/len(loader_val)
        torch.save(model.state_dict(), "model_10k.pth")
        if False:
            plt.subplot(1,3,1)
            plt.imshow(image[0].squeeze().cpu().detach().numpy(), vmin=-1, vmax=1)
            plt.subplot(1,3,2)
            plt.imshow(gt[0].squeeze().cpu().detach().numpy(), vmin=-1, vmax=1)
            plt.subplot(1,3,3)
            plt.imshow(output[0].squeeze().cpu().detach().numpy(), vmin=-1, vmax=1)
            plt.show()
            plt.close()
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(loader_train)}, Validation Loss: {val_loss/len(loader_val)}")

plt.figure(figsize=(10,5))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.legend()
plt.show()


model = UNet(1,1).to(device)
model.load_state_dict(torch.load("model_10k.pth"))
model.eval()

loader_val = tqdm.tqdm(val_loader, total=len(val_loader))

for image, gt in loader_val:
    image = image.to(device)
    gt = gt.to(device)

    output = model(image)
    plt.subplot(1,3,1)
    plt.imshow(image[0].squeeze().cpu().detach().numpy(), vmin=-1, vmax=1)
    plt.subplot(1,3,2)
    plt.imshow(gt[0].squeeze().cpu().detach().numpy(), vmin=-1, vmax=1)
    plt.subplot(1,3,3)
    plt.imshow(output[0].squeeze().cpu().detach().numpy(), vmin=-1, vmax=1)
    plt.show()
    plt.close()
    break


