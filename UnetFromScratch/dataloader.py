import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt


class BuildData(torch.utils.data.Dataset):
    
    def __init__(self, data_dict, mode):
        self.data_dict = data_dict
        self.mode = mode
        self.trans = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.data_dict)
    def __getitem__(self,idx):
        all_images = list(self.data_dict.keys())
        all_masks = list(self.data_dict.values())
        image_path = all_images[idx]
        mask_path = all_masks[idx]
        assert mask_path.split("/")[-1].split(".")[0] == image_path.split("/")[-1].split(".")[0]
        image = Image.open(image_path)
        # print(f"Shape of Image: {np.array(image).shape}")
        if len(np.array(image).shape) == 3:
            # print("converting image to p")
            image = image.convert("L")
            # print(f"Shape of Image after conversion: {np.array(image).shape}")
        
        mask = Image.open(mask_path)
        # print(f"Shape of mask: {np.array(mask).shape}")
        if len(np.array(mask).shape) == 3:
            # print("converting mask to p")
            mask = mask.convert("L")
            # print(f"Shape of mask after conversion: {np.array(mask).shape}")
        image = np.array(image)
        mask = np.array(mask)
        
        if image.shape != (572, 572):
            image = np.array(torchvision.transforms.Compose(
                [Resize((572,572))])(Image.fromarray(image)))
        
        if self.mode == "train":
            mask = np.array(torchvision.transforms.Compose(
                [Resize((388,388))])(Image.fromarray(mask)))
        if self.mode == "test":    
            if image.shape != mask.shape:
                img_shape = image.shape
                mask = np.array(torchvision.transforms.Compose([Resize(img_shape)])(Image.fromarray(mask)))
            # print(f"Shape of mask after resize: {mask.shape}")
            
        image = torch.tensor(image).float()
        mask = (torch.tensor(mask) == 255).float()
        # print(f"Image tensor shape: {image.shape}")
        # print(f"Mask tensor shape: {mask.shape}")
        
        return self.trans(np.array(image)), self.trans(np.array(mask))
    
    
class Resize():
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        
        data_resize = torchvision.transforms.functional.resize(data, self.output_size,
                                                torchvision.transforms.InterpolationMode.NEAREST)
        
        return data_resize
    
class LoadData:
    def __init__(self, datasets, batch_size, shuffle, num_workers) -> None:
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def load(self):
        dataloaded = {}
        try:
            for phase in ['train', 'valid']:
                loader = torch.utils.data.DataLoader(dataset=self.datasets[phase], 
                                                    batch_size=self.batch_size, shuffle=self.shuffle,
                                                    num_workers=self.num_workers)
                dataloaded[phase] = loader
        except:
            loader = torch.utils.data.DataLoader(dataset=self.datasets, 
                                                    batch_size=self.batch_size, shuffle=self.shuffle,
                                                    num_workers=self.num_workers)
            dataloaded = loader
        return dataloaded



if __name__ == "__main__":
    images_path = "./data/sample/images"
    masks_path = "./data/sample/masks/left_lung/"
    image_mask_mapping = {}
    for file in os.listdir(images_path):
        if os.path.exists(os.path.join(masks_path, file.split(".")[0]+".gif")):
            key = os.path.join(images_path, file)
            value = os.path.join(masks_path, file.split(".")[0]+".gif")
            image_mask_mapping[key] = value
    
    for i in range(len(image_mask_mapping)):
        buildData = BuildData(image_mask_mapping)[i]
        plt.imshow(buildData[0], cmap="gray")
        plt.imshow(buildData[1], alpha=0.4)
        plt.show()
        

    
    