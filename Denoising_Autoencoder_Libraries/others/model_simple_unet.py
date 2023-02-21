import torch
from others_functions import *

class UNet_simple1(nn.Module):

    def __init__(self,in_channels = 3, out_channels = 3, negative_slope = 0.1, nb_channels = 21, p = 0.5):

        super(UNet_simple1, self).__init__()
        ##Premier maxpool
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels,nb_channels,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(nb_channels,nb_channels,3,stride=1,padding=1), 
            nn.LeakyReLU(negative_slope),

            nn.MaxPool2d(2))  ##16x16
                            
        ##on remonte vers le premier noeud du U a droite
        self.special = nn.Sequential( #bottleneck + upsampling
            nn.Conv2d(nb_channels,2*nb_channels,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(2*nb_channels,2*nb_channels,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(2*nb_channels,2*nb_channels,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(2*nb_channels,2*nb_channels, 3, stride=2, padding=1, output_padding=1))

        ##final
        self.decode_out = nn.Sequential (
            nn.Conv2d(2*nb_channels + in_channels,2*nb_channels,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(2*nb_channels,nb_channels,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope))
        ## output layer
        self.output_layer = nn.Sequential(nn.Conv2d(nb_channels,out_channels,3,stride=1,padding=1))
        
    def forward(self,x):
        '''
        forward function
        '''
        pool1 = self.encode1(x) #16x16 c 48
        upsample1 = self.special(pool1) #8x8 c 48

        concat1 = torch.cat((upsample1,x),dim=1)
        upsample0 = self.decode_out(concat1)

        output = self.output_layer(upsample0)
        #here we go
        return output
