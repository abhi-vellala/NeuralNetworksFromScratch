import json
from unet import Unet
from dataloader import LoadData, BuildData
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

with open("model_configs.json") as config_file:
        model_configs = json.load(config_file)

model_configs["0"]['type']
        
image = "data/sample/images/JPCLN001.bmp"
mask = "data/sample/masks/left_lung/JPCLN001.gif"

dataset = BuildData({image:mask}, "train")
# print(dataset)

dataload = LoadData(datasets=dataset,batch_size=1, shuffle=True, num_workers=0).load()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Unet(model_configs, input_channels=1, num_classes=2)
model = model.to(device)

for img, mask in dataload:
    output = model(img.to(device))

image_expalin = {'input': Image.open(image)}
k = 0
for key, (idx, out) in model.explain_output.items():
    if 'conv' in str(key).lower():
        if model_configs[str(idx)]['type'] == 'cnn_encode':
            image_expalin['Encoder Layer: '+str(k+1)] = out
        if model_configs[str(idx)]['type'] == 'cnn_decode':
            image_expalin['Decoder Layer: '+str(k+1)] = out
        if model_configs[str(idx)]['type'] == 'transpose':
            image_expalin['Upsample Layer: '+str(k+1)] = out
        if model_configs[str(idx)]['type'] == 'output':
            image_expalin['Output Mask'] = out
    k += 1

for idx, (key, value) in enumerate(image_expalin.items()):
        if key == "input":
                value = np.array(value)
                # print(value.shape)
                # im_shape = (value.shape[1], value.shape[2])
                # print(im_shape)
                plt.imshow(value, cmap='gray')
                plt.suptitle('Input Image')
                plt.grid(False)
                plt.axis('off')
                plt.savefig('./unet_explain_figs/00input.png')
                plt.close()
        else:
            
            for i in range(value.size(1)):
                    im_shape = (value.size(2), value.size(3))
                    plt.imshow(value[:,i,:,:].detach().reshape(im_shape), cmap='gray')
                    if 'output' in key:
                        plt.suptitle(key)    
                    plt.suptitle(f'{key} - Image: {str(i)}')
                    plt.grid(False)
                    plt.axis('off')
                    plt.savefig(f'./unet_explain_figs/{idx}_image{i}.png')
                    plt.close()
            
print("SUCCESS!!") 