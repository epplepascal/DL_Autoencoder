from .module import Module

from torch import empty
from torch.nn.functional import fold, unfold
import math

class Conv2d(Module):

    """
    Convolutional layer
    """

    def __init__(self, input_channel, output_channel, kernel_size, bias = None, stride = 1, padding = 0):
        
        self.name = "Conv2d"

        if (isinstance(kernel_size,tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size,kernel_size)
        
        self.input_channel = input_channel
        self.output_channel=output_channel
        self.stride=stride
        self.padding=padding
        
        self.weight = empty((self.output_channel,self.input_channel,self.kernel_size[0],self.kernel_size[1])).normal_()

        self.gradW = None
         
        if bias == None:
          self.bias = None
          self.gradb = None
        else :
          self.bias = empty(self.output_channel).normal_()
        
        self.gradb = None
    
    def set_weight_conv(self,weight):
        self.weight = weight

    def set_bias_conv(self,bias):
        self.bias = bias

    def set_weight_bias_conv(self,weight,bias):
        self.weight = weight
        self.bias = bias
    
    def forward(self, *input_):

        x = input_[0]

        self.batch_size = x.size(0)
        self.input_channel = x.size(1)
        self.s_in = x.size(-1)
        self.input = input_[0]
        
        self.X_unf = unfold(x ,kernel_size = self.kernel_size,stride=self.stride,padding=self.padding)

        if self.bias is not None:
            wxb = self.weight.view(self.output_channel,-1) @ self.X_unf + self.bias.view(1,-1,1)
        else:
            wxb = self.weight.view(self.output_channel,-1) @ self.X_unf



        H_out = math.floor((x.shape[2]+2*self.padding-self.kernel_size[0])/self.stride+1)
        W_out = math.floor((x.shape[3]+2*self.padding-self.kernel_size[1])/self.stride+1)
        
        self.S1 = wxb.size()[1]
        self.S2 = wxb.size()[2]
        self.s_out = H_out

        return wxb.view(x.size()[0],self.output_channel,H_out,W_out)
  
    def backward(self, *gradwrtoutput):

            grad = gradwrtoutput[0]
            dl_dy = grad.view(grad.size()[0], self.S1, self.S2)

            input_unf = unfold(self.input, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding)
            dl_dw_unv = dl_dy @ input_unf.transpose(1,2)

            dl_dw_summed = dl_dw_unv.sum(0)

            dl_dw_v = dl_dw_summed.view(self.output_channel,self.input_channel,self.kernel_size[0],self.kernel_size[1])
            
            self.gradW = dl_dw_v

            s_in_h = self.input.size()[3]
            s_in_w = self.input.size()[2]

            dl_dx_unf = self.weight.view(self.output_channel,-1).transpose(0,1) @ dl_dy
            dl_dx_f = fold(dl_dx_unf,output_size = (s_in_w,s_in_h), kernel_size = self.kernel_size, padding = self.padding, stride = self.stride)

            if self.bias is not None:
                bias_sum = dl_dy.sum(0)
                bias_sum = bias_sum.sum(1)
                self.gradb  = bias_sum
            else:
                self.gradb = None

            return dl_dx_f

    def param(self):
        return ((self.weight,self.gradW),(self.bias,self.gradb))

class NNUpsampling(Module):

    """
    Nearest Neighbour Upsampling layer
    """

    def __init__(self, scale_factor = 2):
        self.name = "NNUpsampling"
        self.scale_factor = scale_factor #A changer encore
        self.grad = None
        
    def forward(self, *input):
        self.input = input[0]
        upsampled = self.input.repeat_interleave(self.scale_factor, dim = 2).transpose(2, 3).repeat_interleave(2, dim=2).transpose(2, 3)
        return upsampled

    def backward(self, *gradwrtoutput):
        input_size = self.input.size()
        batch_size = input_size[0]
        c_in = input_size[1]
        h_in = input_size[2]
        w_in = input_size[3]

        grad_s = gradwrtoutput[0].size()

        grad_unf = unfold(gradwrtoutput[0], kernel_size = (self.scale_factor,self.scale_factor), stride = (self.scale_factor,self.scale_factor))
        grad_unf = grad_unf.view(batch_size,grad_s[1],2*self.scale_factor,h_in*w_in)
        
        grad = grad_unf.sum(-2).reshape(batch_size,c_in,h_in,w_in)
        return grad
     
    def param(self):
        return [(None,None)]
    
class Upsampling(Module):

    """
    Upsampling Layer implemented with 
    Nearest Neighbour Upsampling + 2D Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias, padding, stride, scale_factor):

        self.name = "Upsampling"

        self.ups_layer = NNUpsampling(scale_factor)
        self.conv_layer = Conv2d(in_channels,out_channels,kernel_size,bias,stride,padding)

        self.conv_layer_weight = None
        self.conv_layer_bias = None
        
    def set_weight_conv(self,weight):
        self.conv_layer.weight = weight

    def set_bias_conv(self,bias):
        self.conv_layer.bias = bias
        
    def forward(self, *input):
         ups_forward = self.ups_layer.forward(input[0])
         conv_forward = self.conv_layer.forward(ups_forward)
         return conv_forward

    def backward(self, *gradwrtoutput):
        conv_backward = self.conv_layer.backward(gradwrtoutput[0])
        ups_backward = self.ups_layer.backward(conv_backward)
        return ups_backward

    def param(self):
        return ((self.conv_layer.weight,self.conv_layer.gradW),(self.conv_layer.bias,self.conv_layer.gradb))