from torch import optim
from torch import nn

from pathlib import Path
from .others.others_functions import *

torch.set_default_dtype(torch.float32)

class UNet(nn.Module):
    def __init__(self,in_channels = 3, out_channels = 3, negative_slope = 0):

        '''
        Implementation of Model3 (see Report_1.pdf file)
        '''

        super(UNet, self).__init__()
        ## con0_conv1_pool1
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
           
            nn.Conv2d(48,48,3,stride=1,padding=1), 
            nn.LeakyReLU(negative_slope ),
            nn.MaxPool2d(2))
        ##conv2_pool2
        self.encode2 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool2d(2))
        ##conv3_pool3 on est en bas du U
        self.encode3 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.MaxPool2d(2))
        ##conv6_upsample5 on remonte vers le premier noeud du U a droite
        self.special = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
        #8x8
        ## decon5a_b_upsample4 on doit prendre en compte le cat 8x8
        self.decode1 = nn.Sequential(
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope ),
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope ),
            #16x16
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
         ## decon5a_b_upsample4 2
        self.decode2 = nn.Sequential(
            nn.Conv2d(144,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            #32x32
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
        ## deconv1a_1b final
        self.decode_out = nn.Sequential (
            nn.Conv2d(96 + in_channels,64,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope),
            nn.Conv2d(64,32,3,stride=1,padding=1),
            nn.LeakyReLU(negative_slope))     
        ## output layer
        self.output_layer = nn.Conv2d(32,out_channels,3,stride=1,padding=1)

      
    def forward(self,x):

        '''
        forward function
        '''

        pool1 = self.encode1(x) #16x16 c 48
        pool2 = self.encode2(pool1) #8x8 c 48
        pool3 = self.encode3(pool2) #4x4 c 48
      


        upsample3 = self.special(pool3) #8x8 c 48
        concat3 = torch.cat((upsample3,pool2),dim=1) #8x8 c 96(48+48)
        upsample2 = self.decode1(concat3)

        concat2= torch.cat((upsample2,pool1),dim=1)
        upsample1 = self.decode2(concat2)

        concat1 = torch.cat((upsample1,x),dim=1)
        upsample0 = self.decode_out(concat1)

        
        output = self.output_layer(upsample0)

        return output

class Model (nn.Module):
    def __init__(self):

        '''
        Initializes the Unet 
        '''

        super(Model, self).__init__()
        self.model = UNet()
        for p in self.model.parameters(): 
            p = xavier_normal_(p)
        self.optimizer = optim.Adamax(self.model.parameters(), lr =3*1e-3)
         
    def load_pretrained_model(self):

        '''
        Loads the parameters saved in bestmodel.pth
        '''

        model_path = Path(__file__).parent / "bestmodel.pth"
        my_parameters = torch.load(model_path, map_location = torch.device('cpu'))
        self.model.load_state_dict(my_parameters)

    def train (self,train_input, train_target, num_epochs) :

        '''
        Model training, works on CPU and GPU (if available)
        train_input and train_target given in RGB [0,255] range
        '''

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set parameters : 
        train_input = train_input.float()
        train_target = train_target.float()

        train_input = train_input/255
        train_target = train_target/255

        mini_batch_size = 100

        self.model.to(device)
        criterion = nn.MSELoss()
        criterion.to(device)  

        for e in range(num_epochs):
            tot_loss = 0
            for b in range(0, train_input.size(0), mini_batch_size):
                train_in= train_input.narrow(0, b, mini_batch_size)
                train_in= train_in.to(device)
                output = self.model(train_in)
                train_tar=train_target.narrow(0, b, mini_batch_size)
                train_tar=train_tar.to(device)
                loss = criterion(output, train_tar)
                tot_loss += loss
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, test_input):

        '''
        Model prediction on a test_input tensor, given in RGB range [0, 255]
        '''

        test_input = test_input.float()
        output = self.model(test_input/255)*255
        output = torch.clamp(output,0,255)
        return output