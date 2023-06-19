import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pydicom
from PIL import Image
import pandas as pd


class BuildData(torch.utils.data.Dataset):
    
    def __init__(self, directory, df, transform=False, resize=None):
        self.df = df
        self.directory = directory
        self.transform = transform
        self.resize = resize
        self.trans = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self,index):
        image_path = os.path.join(self.directory,self.df.iloc[index,0]+'.dcm')
#         print(image_path)
        image = pydicom.dcmread(image_path)
        image = Image.fromarray(image.pixel_array)
        
        label = np.asarray(self.df.iloc[index,5])
        if self.transform:
            image = torchvision.transforms.functional.resize(image, self.resize)
        return self.trans(image), label
    
class DataBuilder:
    def __init__(self, train_images_path, df, trainval_split, transform=False, resize=None):
        self.train_images_path = train_images_path
        self.df = df
        self.transform = transform
        self.resize = resize
        self.trainval_split = trainval_split
        
        
    def prepare(self):
        seed = torch.initial_seed()
        # print('Used seed : {}'.format(seed))
        df = self.df
        train_images_path = self.train_images_path
        trainlen = np.floor(df.shape[0]*self.trainval_split).astype(int)
        # print(trainlen)
        trainidx = np.random.choice(range(df.shape[0]),trainlen, replace=False)
        # print(len(set(trainidx)))
        traindata = df[df.index.isin(trainidx)]
        valdata = df[~df.index.isin(trainidx)]
        # print(traindata.shape, valdata.shape)

        datasets = {"train":[], "validate":[]}
        for i in range(traindata.shape[0]):
            datasets['train'].append(BuildData(train_images_path, traindata, self.transform,self.resize)[i])
            
        for i in range(valdata.shape[0]):
            datasets['validate'].append(BuildData(train_images_path, valdata, self.transform, self.resize)[i])
            
        return datasets



if __name__ == "__main__":
    train_images_path = "./data/sample/"
    df = pd.DataFrame(columns=["patientId", "x", "y", "w", "h", "Target"])
    pids = []
    target = []
    for f in os.listdir(train_images_path):
        if ".dcm" in f:
            pids += [f[:-4]]*5
            target += [f[-5]]*5
    
    df["patientId"] = pids
    df["Target"] = target
    df = df.sample(frac=1).reset_index(drop=True)
    
    dataBuilder = DataBuilder(train_images_path, df, trainval_split=0.8,transform=True,resize=(227,227))
    datasets = dataBuilder.prepare()
    assert len(datasets['train'])+len(datasets['validate'])==df.shape[0]
    print("You Rocked!")
    


# if __name__ == "__main__":
#     train_images_path = "./data/rsna-pneumonia-detection-challenge_data/stage_2_train_images"
#     df = pd.read_csv("./data/rsna-pneumonia-detection-challenge_data/stage_2_train_labels.csv")
#     print(df.head())
#     t1 = df.loc[df['Target'] == 1].iloc[:1000]
#     t0 = df.loc[df['Target'] == 0].iloc[:1000]
#     df = pd.concat([t0,t1], axis=0)
#     print(df.shape)
#     df = df.sample(frac=1).reset_index(drop=True)
#     np.random.seed(1234)
#     trainval_ratio = 0.8
#     trainlen = np.floor(df.shape[0]*trainval_ratio).astype(int)
#     print(trainlen)
#     trainidx = np.random.choice(range(df.shape[0]),trainlen, replace=False)
#     print(len(set(trainidx)))
#     traindata = df[df.index.isin(trainidx)]
#     valdata = df[~df.index.isin(trainidx)]
#     print(traindata.shape, valdata.shape)
    
#     datasets = {"train":[], "validate":[]}
    # for i in range(traindata.shape[0]):
    #     datasets['train'].append(BuildData(train_images_path, traindata, True,(250,250))[i])
        
    # for i in range(valdata.shape[0]):
    #     datasets['validate'].append(BuildData(train_images_path, valdata, True,(250,250))[i])