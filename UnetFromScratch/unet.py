import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json

class Unet(nn.Module):
    def __init__(self, model_configs) -> None:
        super(Unet, self).__init__()
        self.model_configs = model_configs
        self.net, self.decoder_layers = self.convnet(self.model_configs)
        # (inp - fil + 2*pad)/str + 1 
    def convnet(self, model_configs):
        hidden_layers = []
        decoder_layers = []
        for layer_num, configs in model_configs.items():
            layers = []
            if configs['type'] in ["cnn_encode", 'cnn_decode']:
                in_channels = configs['in_channels']
                out_channels = configs['out_channels']
                kernel_size = configs['kernel_size']
                padding = configs['padding']
                stride = configs['stride']
                layers.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                        kernel_size=kernel_size, padding=padding, stride=stride))
            if configs['type'] == "activation":
                layers.append(nn.ReLU())
            if configs['type'] == "maxpool":
                layers.append(nn.MaxPool2d(kernel_size=configs['kernel_size'], 
                                            stride = configs['stride']))
                
            if configs['type'] == 'transpose':
                layers.append(nn.ConvTranspose2d(in_channels=configs['in_channels'], 
                                                    out_channels=configs['out_channels'], 
                                                    kernel_size=configs['kernel_size'],
                                                    stride=configs['stride']))
                
            if 'decode' in configs.keys():
                if configs['decode']:
                    decoder_layers.append(layer_num)
                
            hidden_layers += layers
            
        return nn.Sequential(*hidden_layers), decoder_layers
    
    def forward(self, x):
        self.decoder_map = {}
        print(f"This is decode layers: {self.decoder_layers}")
        for idx, layer in enumerate(self.net):
            # print(layer)
            print(idx)
            if str(idx) in self.decoder_layers:
                x = layer(x)
                print(x.shape)
                if "crop" in self.model_configs[str(idx)].keys():
                    print(f"crop at {idx}")
                    crop_size = self.model_configs[str(idx)]['crop_size']
                    crop_img = transforms.CenterCrop(crop_size)
                    self.decoder_map[str(idx)] = crop_img(x)
                    print(self.decoder_map[str(idx)].shape)
            
            elif 'get_img' in model_configs[str(idx)]:
                if model_configs[str(idx)]['get_img']:
                    print("In decoder part")
                    x_get = self.decoder_map[model_configs[str(idx)]['decoder_map']]
                    print(f"x_get shape: {x_get.shape}, x shape: {x.shape}")
                    x = torch.cat([x_get,x], dim=1)
                    print(f"x cat shape: {x.shape}")
                    x = layer(x)
                    print(f"layer func: {x.shape}")
            else:
                x = layer(x)
                print(x.shape)             
        return x     
    
    
        


with open("model_configs.json") as config_file:
    model_configs = json.load(config_file)

image = torch.rand((1, 1, 572, 572))
unet_model = Unet(model_configs)
out = unet_model(image)
