import os
import torch
import pandas as pd
import torch.optim as optim
import json
from dataloader import DataBuilder
from alexnet import AlexNet
import matplotlib.pyplot as plt



train_images_path = "./data/rsna-pneumonia-detection-challenge_data/stage_2_train_images"
df = pd.read_csv("./data/rsna-pneumonia-detection-challenge_data/stage_2_train_labels.csv")
t1 = df.loc[df['Target'] == 1].iloc[:100]
t0 = df.loc[df['Target'] == 0].iloc[:100]
df = pd.concat([t0,t1], axis=0)
print(f"Total shape of data: {df.shape}")
df = df.sample(frac=1).reset_index(drop=True)
dataBuilder = DataBuilder(train_images_path, df, trainval_split=0.8,transform=True,resize=(227,227))
datasets = dataBuilder.prepare()
print(f"Shape of training data: {len(datasets['train'])}")
print(f"Shape of validation data: {len(datasets['validate'])}")

batch_size = 4
trainloader = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size,
                                          shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(datasets['validate'], batch_size=batch_size,
                                          shuffle=False, num_workers=0)

print("dataloader successful!")

with open('model_setting.json') as config_file:
        model_configs = json.load(config_file)
model_save_path = "./model_weights/"    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is set to: {device}")
alexnet = AlexNet(model_configs)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

epochs = 10
losses = []
acc = []
steps = 0
predicted = []
for epoch in range(epochs):
    total = 0
    correct = 0
    for idx, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        
        model_output = alexnet(images)
        loss = criterion(model_output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        with torch.no_grad():
            _, pred = torch.max(model_output.data,1)
            predicted.append(pred)
            total += labels.size(0)
            correct += (labels == pred).sum().item()
    
    accuracy = (correct/total)
    print(f"Epoch {epoch+1}/{epochs}: Loss: {round(loss.item(), 4)} Accuracy : {round(accuracy*100, 4)}")
            
    
    losses.append(loss.item())
    acc.append(accuracy)
    steps += 1
    if acc[epoch-1] > acc[epoch]:
        torch.save(alexnet.state_dict(), os.path.join(model_save_path, 'best_model.pt'))
        break
    
    
    
    
    

# losses = [1.8, 0.6, 0.4, 0.05, 0.01]
plt.plot(range(len(losses)), losses, label='loss')
plt.scatter(range(len(losses)), losses, c="red", marker='o')
plt.plot(range(len(acc)), acc, label='accuracy')
plt.scatter(range(len(acc)), acc, c="red", marker='o')
plt.xlabel("epochs")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()