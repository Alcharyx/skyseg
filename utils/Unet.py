import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF


# Unet paper https://arxiv.org/abs/1505.04597
class DoubleConv(nn.Module):
    """
    Double Convolutionnal layer used in UNET architecture composed of two sequential Conv2d, Batchnormalization and ReLu

    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):
    """
    Instance of the UNET model based on the original paper : https://arxiv.org/abs/1505.04597
    
    """
    def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128 ,256, 512]):
        super(UNET,self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.pool = nn.AdaptiveMaxPool2d(output_size=(1,1))

        #Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        #Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        #Decoder could use bi-linear conv
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature *2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature*2,feature))

        self.final_conv = nn.Conv2d(features[0],out_channels, kernel_size= 1)
    
    def forward(self,x):
        skip_connections = []

        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x= self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]

            #if image shape not divisible by 16 (2x2x2x2) when concat
            #if x.shape != skip_connection.shape:
            #    x = TF.resize(x,size= skip_connection.shape[2:])
            #Risk of error in ONNX conversion pick shape divisible by 16

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)

        return self.final_conv(x)
