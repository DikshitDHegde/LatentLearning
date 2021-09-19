import torch
from torch._C import PyTorchFileReader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    """Some Information about Encoder"""
    def __init__(self, input_ = 784, encoderLayers = [500,500,2000,10]):
        super(Encoder, self).__init__()
        
        X = []

        for layer in encoderLayers:
            x = [nn.Linear(input_, layer), nn.LeakyReLU()]
            if layer == encoderLayers[-1]:
                x  = [nn.Linear(input_, layer)]
            
            X.extend(x)
            input_ = layer
        
        self.Enc = nn.Sequential(*X)

    def forward(self, x):
        x = self.Enc
        return x


class Decoder(nn.Module):
    """Some Information about Decoder"""
    def __init__(self,input_, decoderLayers=[10,2000,500,500,784]):
        super(Decoder, self).__init__()
        X = []

        for layer in decoderLayers:
            x = [nn.Linear(input_, layer), nn.LeakyReLU()]
            if layer == decoderLayers[-1]:
                x  = [nn.Linear(input_, layer)]
            
            X.extend(x)
            input_ = layer
        
        self.Dec = nn.Sequential(*X)
        
    def forward(self, x):
        x = self.Dec(x)
        return x

class Model(nn.Module):
    """Some Information about Model"""
    def __init__(self, input_=784, encoderLayers=[500,500,2000,10], decoderLayers=[2000,500,500,784*2]):
        super(Model, self).__init__()
        self.mid = decoderLayers[-1]//2
        self.VAEncoder = Encoder(
            input_= input_,
            encoderLayers= encoderLayers
        )
        
        self.ClaEncoder1 = Encoder(
            input_= input_,
            encoderLayers= encoderLayers
        )
        self.ClaEncoder2 = self.ClaEncoder1

        self.Decoder = Decoder(
            input_= encoderLayers[-1],
            decoderLayers = decoderLayers
        )

        self.latent1 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.latent2 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.latent3 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.mu1 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.var1 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.mu2 = nn.Linear(self.mid, self.mid)
        self.var2 = nn.Linear(self.mid, self.mid)


    def forward(self, x):

        return x