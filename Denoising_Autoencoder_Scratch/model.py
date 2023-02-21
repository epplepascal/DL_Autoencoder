from .others.sequential import *
from .others.convolutions import *
from .others.activations import *
from .others.losses import *
from .others.optimizers import *

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

import pickle
from pathlib import Path


class Model():
    def __init__(self):

        """
        Initializes the UNet presented in Report_2.pdf
        """
        
        self.model = Sequential(
            Conv2d(3, 32, kernel_size = (2,2), bias = True, stride = 2),
            ReLU(),
            Conv2d(32, 64, kernel_size = (2,2), bias = True, stride = 2),
            ReLU(),
            Upsampling(64,32, kernel_size= (3,3), bias = True, padding = 1, stride = 1, scale_factor= 2),
            ReLU(),
            Upsampling(32,3, kernel_size= (3,3), bias = True, padding = 1, stride = 1, scale_factor= 2),
            Sigmoid()
        )

    def load_pretrained_model(self):

        """
        Set the parameters of the model stored in the
        'bestmodel.pth' file. 
        'bestmodel.pth' file contains a list of all the weight and bias
        tensors that make up the neural network.
        """
        
        p = Path(__file__).with_name('bestmodel.pth')
      
        with open(p, 'rb') as handle:
            param = pickle.load(handle)
        handle.close()

        self.model.set_param(param)
        
    def train(self, train_input, train_target, num_epochs):

        '''
        Model training, works on CPU and GPU (if available)
        train_input and train_target given in RGB [0,255] range
        '''

        train_input = train_input.float()
        train_target = train_target.float()
        train_target = train_target/255
        train_input = train_input/255

        criterion = MSE()
        optimizer = SGD(self.model, lr = 3)
        batch_size = 50

        for e in range(num_epochs):
            #print(e)
            tot_loss = 0
            for b in range(0, train_input.size(0), batch_size):
                output = self.model.forward(train_input.narrow(0, b, batch_size))
                loss = criterion.forward(output,train_target.narrow(0, b, batch_size))
                tot_loss += loss
                gradwrtloss = criterion.backward()
                self.model.backward(gradwrtloss)
                optimizer.step()
                self.model.zero_grad()          

    def predict(self,test_input):

        '''
        Model prediction on a test_input tensor, given in RGB range [0, 255]
        '''

        test_input = test_input.float()
        prediction = self.model.forward(test_input/255)*255
        return prediction
