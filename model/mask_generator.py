import torch
import torch.nn as nn




class mask_generator_basic(nn.Module):
    def __init__(self, f_image_channel=32):
        super().__init__()
        self.f_image_channel = f_image_channel
        self.fusion_features = nn.Conv2d(in_channels=2*f_image_channel+2+2, out_channels=4*f_image_channel, kernel_size=3, padding=1, stride=1)
        self.conv0 = nn.Conv2d(in_channels=4*f_image_channel, out_channels=4*f_image_channel, kernel_size=3, stride=1, padding=1)
        self.relu0 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=4*f_image_channel, out_channels=4*f_image_channel, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=4*f_image_channel, out_channels=2*f_image_channel, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        
    def forward(self, fimage0, fimage1, of0, of1):
        
        x = self.fusion_features(torch.cat([fimage0, fimage1, of0, of1], dim=1))
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        warp0 = x[:,0:self.f_image_channel]
        warp1 = x[:,self.f_image_channel:]
        
        return warp0, warp1