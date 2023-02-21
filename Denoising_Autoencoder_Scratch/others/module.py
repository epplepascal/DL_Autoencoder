from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

class Module (object) :

    """
    Module class, building block of every Module
    """
    
    def forward (self, *input):
        raise NotImplementedError
    def backward (self, *gradwrtoutput):
        raise NotImplementedError 
    def param (self) :
        return []
        
