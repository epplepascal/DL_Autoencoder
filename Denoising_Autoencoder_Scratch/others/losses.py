from .module import Module

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold


class MSE(Module): 

    """
    Mean Squared Error loss function
    """
    
    def __init__(self):  
        self.input = None
        self.target = None
        self.n = None
        self.e = None

    def forward(self, input, target):

        self.input = input
        self.target = target

        s_in = self.input.size()

        self.n = s_in[0]*s_in[1]*s_in[2]*s_in[3]

        self.e = (input - target)
        loss = (self.e).pow(2)

        return loss.mean()

    def backward(self):
        return 2 * self.e / self.n