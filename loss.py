from pytorch_msssim import SSIM
import torch.nn as nn
import torch.nn.functional as F


class reconLoss(nn.Module):
    """
        alpha : 0 to 1 (float value) 
                1 : for MSE
                0 : for SSIM
    """

    def __init__(self, in_channels=1, use_ssim=True, alpha=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.MSE = nn.MSELoss()
        self.use_ssim = use_ssim

        if use_ssim:
            self.SSIM = SSIM(
                data_range=1.0, channel=self.in_channels, size_average=True)

    def forward(self, recon, input_):
        mse = self.MSE(recon, input_)
        if self.use_ssim:
            ssim = self.SSIM(recon, input_)

            mse = self.alpha * mse + (1-self.alpha) * ssim

        return mse