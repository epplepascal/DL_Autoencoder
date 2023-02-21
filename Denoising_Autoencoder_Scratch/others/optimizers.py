from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

class SGD():

    """
    Optimizer based on Stochastic Gradient Descent 
    with a positive learning rate
    """

    def __init__(self, model, lr):

        if lr < 0.0:
            raise ValueError("Learning rate of optimizer must be positive!")
        self.model = model
        self.lr = lr

    def step(self):

        """
        The only modules to update are Upsampling and Conv2d, 
        as they are the only layers with weights and biases 
        """

        for module in self.model.modules:
            if (module.name == "Conv2d"):
                module.set_weight_conv(module.weight - self.lr * module.gradW)
                module.set_bias_conv(module.bias - self.lr * module.gradb)
            if (module.name == "Upsampling"):
                
                tupWeight, tupBias = module.param()
                weight = tupWeight[0]
                gradW = tupWeight[1]
                bias = tupBias[0]
                gradB = tupBias[1]

                module.set_weight_conv(weight - self.lr * gradW)
                module.set_bias_conv(bias - self.lr * gradB)

        