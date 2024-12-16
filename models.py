import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(UNet, self).__init__()
        
        # Downsample
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.e3 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        self.e4 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        self.e5 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 16))

        # Upsample

        self.d3 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout(0.5))
        self.d4 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 16, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4),
                                nn.Dropout(0.5))
        self.d5 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 8, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        self.d6 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 4, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        self.d7 = nn.Sequential(nn.ReLU(inplace=True),
                                nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1))
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.e1(x) # 16
        e2 = self.e2(e1) # 32
        e3 = self.e3(e2) # 64
        e4 = self.e4(e3) # 128
        e5 = self.e5(e4) # 128

        d3_ = self.d3(e5) 
        d3 = torch.cat([d3_, e4], dim=1) #256
        d4_ = self.d4(d3)
        d4 = torch.cat([d4_, e3], dim=1) #128
        d5_ = self.d5(d4)
        d5 = torch.cat([d5_, e2], dim=1)
        d6_ = self.d6(d5)
        d6 = torch.cat([d6_, e1], dim=1)
        d7 = self.d7(d6)
        
        # Output
        o1 = self.tanh(d7)
        
        return o1