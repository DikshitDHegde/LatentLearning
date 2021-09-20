import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    """Some Information about Encoder"""

    def __init__(self, input_=784, encoderLayers=[500, 500, 2000, 10]):
        super(Encoder, self).__init__()

        X = []

        for layer in encoderLayers:
            x = [nn.Linear(input_, layer), nn.LeakyReLU()]
            if layer == encoderLayers[-1]:
                x = [nn.Linear(input_, layer)]

            X.extend(x)
            input_ = layer

        self.Enc = nn.Sequential(*X)

    def forward(self, x):
        x = self.Enc(x)
        return x


# TODO: Copy weights function

# TODO: EMA class.


class Decoder(nn.Module):
    """Some Information about Decoder"""

    def __init__(self, input_, decoderLayers=[10, 2000, 500, 500, 784]):
        super(Decoder, self).__init__()
        X = []

        for layer in decoderLayers:
            x = [nn.Linear(input_, layer), nn.LeakyReLU()]
            if layer == decoderLayers[-1]:
                x = [nn.Linear(input_, layer)]

            X.extend(x)
            input_ = layer

        self.Dec = nn.Sequential(*X)

    def forward(self, x):
        x = self.Dec(x)
        return x


class Model(nn.Module):
    """Some Information about Model"""

    def __init__(self, input_=784, encoderLayers=[500, 500, 2000, 10], decoderLayers=[2000, 500, 500, 784*2]):
        super(Model, self).__init__()

        self.mid = decoderLayers[-1]//2
        self.VAEncoder = Encoder(
            input_=input_,
            encoderLayers=encoderLayers
        )

        self.ClaEncoder1 = Encoder(
            input_=input_,
            encoderLayers=encoderLayers
        )
        self.ClaEncoder2 = self.ClaEncoder1

        self.Decoder = Decoder(
            input_=encoderLayers[-1],
            decoderLayers=decoderLayers
        )

        self.latent1 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.latent2 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.latent3 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.mu1 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.var1 = nn.Linear(encoderLayers[-1], encoderLayers[-1])
        self.mu2 = nn.Linear(self.mid, self.mid)
        self.var2 = nn.Linear(self.mid, self.mid)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        z = self.VAEncoder(x)

        z = self.latent1(z)

        mu1 = self.mu1(z)
        var1 = self.var1(z)
        std1 = torch.exp(0.5*var1)
        eps = torch.rand_like(std1)
        # print(eps.shape)
        samp1 = mu1 + eps*std1  # REPRAMETERIZATION
        recon = self.Decoder(samp1)
        # print(recon.shape)
        mu2 = recon[:, :self.mid]
        var2 = recon[:, self.mid:]
        std2 = torch.exp(0.5*var2)
        eps2 = torch.rand_like(std2)

        # print(eps.shape)

        recon = F.sigmoid(mu2 + eps2 * std2)  # REPRAMETERIZATION

        cprob2 = self.latent2(self.ClaEncoder1(recon))
        # copy_weights(self.ClaEncoder2, self.ClaEncoder1)
        # copy_weights(self.latent2, self.latent3)
        cprob1 = self.latent1(self.ClaEncoder2(x))
        recon = recon.reshape(x.size(0),1,28,28)
        return z, recon, cprob1, cprob2


# def testEncoder():
#     model = Encoder()
#     input1 = torch.rand()

def testModel():
    model = Model()
    input1 = torch.rand((10, 784))

    z, recon, cprob1, cprob2 = model(input1)

    print(model)


if __name__ == "__main__":
    testModel()
