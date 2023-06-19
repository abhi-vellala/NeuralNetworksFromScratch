import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

# output shape: (Input - kernel + 2*padding)/stride + 1

class AlexNet(nn.Module):
    def __init__(self, model_configs):
        super(AlexNet, self).__init__()
        self.model_configs = model_configs
        self.convnet = self.alexnet_cnn(self.model_configs)
        self.dense = self.alexnet_dense(self.model_configs)
        
    def forward(self,x):
        self.output = {}
        for layer in self.convnet:
            x = layer(x)
            self.output[layer] = x
            # print(f"{layer}: {x.shape}")
        x = torch.flatten(x,1)
        # x = x.reshape(x.size(0), -1)
        # print(f"flatten: {x.shape}")
        
        for layer in self.dense:
            x = layer(x)
            self.output[layer] = x
            # print(f"{layer}: {x.shape}")
        x = self.softmax(x)
        return x
        
    def alexnet_cnn(self, model_configs):
        conv_layers = model_configs['conv']
        hidden_layers = []
        for num in range(len(conv_layers)):
            layer_configs = conv_layers[str(num)]
            in_channels = layer_configs['in_channels']
            out_channels = layer_configs['out_channels']
            kernel_size = layer_configs['kernel_size']
            stride = layer_configs['stride']
            padding = layer_configs['padding']
            layer = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
            if "batch_norm" in layer_configs.keys():
                if layer_configs['batch_norm']:
                    layer.append(nn.BatchNorm2d(out_channels))
            if "relu" in layer_configs.keys():
                layer.append(nn.ReLU())
            if "max_pool" in layer_configs.keys():
                layer.append(nn.MaxPool2d(kernel_size=layer_configs['max_pool']['kernel_size'], 
                                          stride = layer_configs['max_pool']['stride']))
                
            hidden_layers += layer
            
        return nn.Sequential(*hidden_layers)
    
    def alexnet_dense(self, model_configs):
        fc_layers = model_configs['fully_connected']
        hidden_layers = []
        for num in range(len(fc_layers)):
            layer = []
            layer_configs = fc_layers[str(num)]
            in_features = layer_configs['in_features']
            out_features = layer_configs['out_features']
            if "drop_out" in layer_configs.keys():
                layer.append(nn.Dropout(layer_configs["drop_out"]))
            layer.append(nn.Linear(in_features, out_features))
            if "relu" in layer_configs.keys():
                layer.append(nn.ReLU())
                
            hidden_layers += layer
            
        return nn.Sequential(*hidden_layers)
    
    def softmax(self, x):
        smres = torch.nn.Softmax(dim=1)
        return smres(x) 
    
            
    
    
if __name__ == "__main__":
    with open('model_setting.json') as config_file:
        model_configs = json.load(config_file)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alexnet = AlexNet(model_configs)
    # for n, p in alexnet.named_parameters():
    #     print(n, p.numel())
    x = torch.randn(1, 1, 227, 227).to(device)
    xnet = alexnet(x)
    assert tuple(xnet.shape) == (1,2)
    # probs = alexnet.softmax(xnet)
    print(xnet.data)
    assert int(xnet.sum()) == 1
    print("You Rocked!")