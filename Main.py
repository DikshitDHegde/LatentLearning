import warnings

warnings.simplefilter(action='ignore') #, category=FutureWarning)
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.utils import save_image
from tqdm import tqdm

import utils
from config import config
from Dataloader import dataSetFn, loadData
from Models import Model
from loss import reconLoss

# from pytorch_msssim import SSIM

torch.set_printoptions(linewidth=50)


def save_latent(dir, epoch, latent, y,name):
    data = {"latent": latent, "target": y}
    with open(os.path.join(dir, f"{name}_{epoch}_.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_optim(optimze, model, learning_rate):
    if optimze == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimze == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print("OPTIMIZER NOT IMPLEMENTED: SORRY")
        exit()

    return optimizer


# use_gpu = torch.cuda.is_available()
use_gpu = False
if use_gpu:
    device = "cuda"
    print(" USING CUDA :)")
else:
    device = "cpu"


batchsize = 512
optimizer = "Adam"
# optimizer = "SGD"
lr = 1e-3
# dataset = loadData("/home/cvg-ws2/Desktop/Dikshit/FashionMNIST_Combined.npz")
dataset = loadData("/home/dikshit/DATA/FashionMNIST/FashionMNIST_Combined.npz")



args = config()
utils.set_seed_globally(0, use_gpu)

model = Model(
    input_= 784,
    encoderLayers=[500,500,2000,128],
    decoderLayers=[10,2000,500,500,784*2]
).to(device)

trainSet = dataSetFn(dataset=dataset, transform_original= transforms.Compose([transforms.ToTensor()]))
trainLoader = DataLoader(trainSet, batch_size=batchsize, shuffle=True,
                         num_workers=10, pin_memory=False, prefetch_factor=batchsize//4)
# print("DATALOADER")
optimize = get_optim(optimizer, model, lr)
# criterion_con = nn.CrossEntropyLoss().to(device)
criterion_mse = reconLoss(in_channels=1, use_ssim=True, alpha=0.75).to(device)

tb = SummaryWriter(
    "./log/FMNIST/BatchSize {} LR {} Optimizer {}".format(batchsize, lr, optimizer))


for epoch in range(args.epochs):
    emb_loss,recon_loss = utils.pretrain(args, model, trainLoader,
                              device, optimize, criterion_mse, epoch)
    # emb_loss,recon_loss = utils.pretrainVAE(args, model, trainLoader, device, optimize, criterion_mse, epoch)
    tb.add_scalar("Emb loss", emb_loss, global_step=epoch)
    tb.add_scalar("Recon loss", recon_loss, global_step=epoch)
    if (epoch+1) % 10 == 0:
        acc_1, nmi_1, ari_1 = utils.Cluster(
            args, model, dataset, device, epoch)
        tb.add_scalar("Pretrain ACC L1", acc_1, global_step=epoch)

        tb.add_scalar("Pretrain NMI L1", nmi_1, global_step=epoch)

        tb.add_scalar("Pretrain ARI L1", ari_1, global_step=epoch)

    if (epoch+1) % 100 == 0:
        k = os.path.join(args.out_dir, "FMNIST")
        if not os.path.exists(k):
            os.makedirs(k)

        torch.save(
            {
                "weights": model.state_dict(),
                "optimizer": optimize.state_dict(),
                "epoch": epoch+1,
                "Emb_loss": emb_loss
            }, os.path.join(k, f"BatchSize {batchsize} LR {lr}  Optim {optimizer}.pth.tar")
        )
