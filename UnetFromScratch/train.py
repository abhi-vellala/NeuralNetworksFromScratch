from unet import Unet
import json
import os
from dataloader import BuildData, LoadData
# from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
from torch.nn import BCEWithLogitsLoss
import time


with open("model_configs.json") as config_file:
        model_configs = json.load(config_file)

images_path = "./data/images"
masks_path = "./data/masks/left_lung/"

image_mask_mapping = {}
for file in os.listdir(images_path):
    if os.path.exists(os.path.join(masks_path, file.split(".")[0]+".gif")):
        key = os.path.join(images_path, file)
        value = os.path.join(masks_path, file.split(".")[0]+".gif")
        image_mask_mapping[key] = value

d = [(i,m) for i,m in image_mask_mapping.items()]
np.random.shuffle(d)
train_size = int(len(d)*0.8)
train, valid = dict(d[:train_size]), dict(d[train_size:])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
datasets = {phase[0]: BuildData(phase[1], mode="train") for phase in [('train',train), ('valid', valid)]}

epochs = 2
num_classes = 1
batch_size = 4

model = Unet(model_configs, num_classes=num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
lossFunc = BCEWithLogitsLoss()


dataload = LoadData(datasets=datasets,batch_size=batch_size, shuffle=True, num_workers=0).load()

plt.ion()
figure1, ax1 = plt.subplots()
ax1.set_title("Loss Vs Batch Size Updating...")

figure2, ax2 = plt.subplots()
ax2.set_title("Loss Vs Epochs Updating...")

losses = []
bs = []
phase = "train"
bs_count = batch_size
for epoch in range(epochs):
    running_loss = 0
    for images, mask in dataload[phase]:
        images = images.to(device)
        mask = mask.to(device)
        pred = model(images)
        loss = lossFunc(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        bs.append(bs_count)
        ax1.scatter(bs, losses, marker="o", c="red")
        ax1.plot(bs, losses, label="Loss", c="black")
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss vs Batch Size")
        figure1.canvas.draw()
        figure1.canvas.flush_events()
        time.sleep(0.1)
        # plt.show(block=False)
        print(f"Epoch: {epoch+1}/{epochs}, Batch: {bs_count}/{len(dataload[phase])} Loss: {loss.item()}")
        bs_count += batch_size
        running_loss += loss.item()
        
    ax2.scatter(range(epoch+1), running_loss, marker="o", c="red")
    ax2.plot(range(epoch+1), running_loss, label="Loss", c="black")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Epochs")
    figure2.canvas.draw()
    figure2.canvas.flush_events()
    time.sleep(0.1)

