from .module import *

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

class Sequential(Module):

    """
    Similar to nn.Sequential
    Defines a container where one can create its own neural network

    self.conv is needed in order to set the parameters of the convolutions
    convolutions are appended twice because weights and biases of the
    convolutions need to be set
    """

    def __init__(self, *args):

        self.modules = []
        self.conv = []
        for module in args:
            self.modules.append(module)
            if (module.name == "Conv2d" or module.name == "Upsampling"):
                self.conv.extend([module]*2)

    def forward(self, *input):

        """
        Repeatedly calls the forward passes of the modules
        that form the network
        """

        self.input = input[0]
        output = input[0]
        for module in self.modules:
            output = module.forward(output)
        self.output = output
        return self.output
    
    def backward(self, derivative):

        """
        Repeatedly calls the backward passes of the modules
        that form the network
        """

        for module in reversed(self.modules):
            derivative = module.backward(derivative)
        self.grad = derivative
        return self.grad
    
    def param(self):
        parameters = []
        for module in self.modules:
            parameters.extend(module.param())
        return parameters

    def set_param(self,params_pytorch):

        """
        Set the parameters of the convolutions
        Params_pytorch is a list containing (Weights, Biases) 
        of the corresponding convolution
        Note that the order matters!
        """

        convolutions = self.conv
        for ind, weight in enumerate(params_pytorch):

            if ind % 2 == 0: 
                if (convolutions[ind].name == "conv2d"):
                    convolutions[ind].weight = weight[:]
                else:
                    convolutions[ind].set_weight_conv(weight[:])     
            else:
                if (convolutions[ind].name == "conv2d"):
                    convolutions[ind].bias = weight[:]
                else:
                    convolutions[ind].set_bias_conv(weight[:])


    def zero_grad(self):

        """
        Sets the gradients of modules with weights back to zero
        """

        for parameter in self.param():
            weight, grad = parameter
            if (weight is None) or (grad is None):
                continue
            else:
                grad.zero_()
                
    def add_modules(self, *modules):

        """
        Adds a new module at the end of the existing container.
        """

        for module in modules:
            self.modules.append(module)   

     