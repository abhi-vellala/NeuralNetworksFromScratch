import json
from alexnet import AlexNet
import torch
import pydicom
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms

image = pydicom.dcmread('data/sample/class0.dcm')
image = Image.fromarray(image.pixel_array)
image = torchvision.transforms.functional.resize(image, (227, 227))
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
image = trans(image)
image = image.reshape(1,1,227,227)

with open('model_setting.json') as config_file:
        model_configs = json.load(config_file)
    
alexnet = AlexNet(model_configs)
alexnet.load_state_dict(torch.load('./model_weights/best_model.pt'))
alexnet.eval()
output = alexnet(image)
print(f"Out of the image: {output}")
image_expalin = {'input':image}
k = 0
for i, key in enumerate(alexnet.output.keys()):
    if 'conv' in str(key).lower():
        image_expalin['conv'+str(k)] = alexnet.output[key]
        print(alexnet.output[key].shape)
        k += 1

for idx, (k, v) in enumerate(image_expalin.items()):
        if k == "input":
                im_shape = (image_expalin['input'].size(2), image_expalin['input'].size(3))
                print(im_shape)
                plt.imshow(image_expalin['input'].reshape(im_shape), cmap='gray')
                plt.suptitle('Input Image')
                plt.grid(False)
                plt.axis('off')
                plt.savefig('./all_figs/00input.png')
                plt.close()
        else:
                for i in range(image_expalin['conv'+str(idx-1)].size(1)):
                        im_shape = (image_expalin['conv'+str(idx-1)].size(2), 
                                image_expalin['conv'+str(idx-1)].size(3))
                        plt.imshow(image_expalin['conv'+str(idx-1)][:,i,:,:].detach().reshape(im_shape), 
                                cmap='gray')
                        plt.suptitle(f'Convolution Layer {idx-1} - Image: {i}')
                        plt.grid(False)
                        plt.axis('off')
                        plt.savefig(f'./all_figs/conv{idx-1}_image{i}.png')
                        plt.close()
                
print("SUCCESS!!")     
# plt.imshow(image_expalin['conv0'][:,50,:,:].detach().reshape(55,55), cmap='gray')
# plt.suptitle('Check this')
# plt.grid(False)
# plt.axis('off')
# plt.show()

# fig, axs = plt.subplots(2,3)
# j = 0
# for i,key in enumerate(image_expalin.keys()):
#     if i < 3:
#         axs[0,i].imshow(image_expalin[key].reshape(227,227), cmap='gray')
#         axs[0,i].set_title(key)
#     else:
#         axs[1,j].imshow(image_expalin[key].reshape(227,227), cmap='gray')
#         axs[0,j].set_title(key)
#         j += 1
        


# plt.show()
