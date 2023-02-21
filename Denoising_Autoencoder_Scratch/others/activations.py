from .module import Module

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

class Sigmoid(Module):

    """
    Sigmoid activation function
    """

    def __init__(self) -> None:

        self.name = "Sigmoid"
        self.s = None
        self.back = None
        

    def forward(self, *input):

        self.s = 1 / (1 + (-input[0]).exp())
        return self.s

    def backward(self, *gradwrtoutput):

        self.back = (self.s - self.s**2)
        return gradwrtoutput[0] * self.back
    
class ReLU(Module):

    """
    Rectified Linear Unit activation function
    """
    
    def __init__(self) -> None:

        self.s = None
        self.name = "ReLU"
    def forward(self, *input):

        s = input[0]
        self.s = s
        self.s[self.s <= 0] = 0
        return self.s
    def backward(self, *gradwrtoutput):

        self.back = self.s
        self.back[self.back <= 0] = 0
        self.back[self.back > 0] = 1
        return self.back * gradwrtoutput[0]

class leakyReLU(Module):

    """
    leakyReLU activation function, which takes a non-zero parameter alpha.
    """

    def __init__(self,alpha) -> None:
        if alpha < 0.0:
            raise ValueError("Slope must be positive!")
        self.alpha = alpha
        self.s = None
        self.name = "leakyRelu"

    def forward(self, *input):
        s = input[0]
        self.s = s 
        self.s[self.s <= 0] = self.alpha * self.s[self.s <= 0]
        return self.s
    def backward(self, *gradwrtoutput):

        self.back = self.s
        self.back[self.back <= 0] = self.alpha
        self.back[self.back > 0] = 1
        return self.back * gradwrtoutput[0]


