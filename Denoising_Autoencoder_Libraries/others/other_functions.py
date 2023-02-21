import torch
import torchvision
from torch import nn
import math 

def xavier_normal_(tensor, gain = 1):
    if tensor.dim()<2:
        return tensor
    fan_in,fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0, std)



def data_augmentation(noisy_imgs_1,noisy_imgs_2,c,data_a1=True,data_a2=True) :
    '''
    Here we add two boolean parameters data_a1 and data_a2 for the two data augmentation we perform.
    Where data_a1 correspond to data augmentation where just switch the role of the input and the taget
    data_a2 corresponds to data augmentation where just rotate some images and we add a number of c images to our dataset to our data set 
    
    '''
    #randomized the order of our images
    k_train=noisy_imgs_1.size()[0]
    idx = torch.randperm(k_train)
    idx = idx[:k_train]

    noisy_imgs_1 = noisy_imgs_1[idx]
    noisy_imgs_2 = noisy_imgs_2[idx]
    
    if data_a2 :
        noisy_imags_add1= torchvision.transforms.functional.rotate(noisy_imgs_1,angle=180)
        noisy_imags_add2= torchvision.transforms.functional.rotate(noisy_imgs_2,angle=180)
        
    if data_a1 :
        train_set = torch.cat((noisy_imgs_1,noisy_imgs_2),dim = 0)
        test_set = torch.cat((noisy_imgs_2,noisy_imgs_1),dim = 0)
        
    if data_a2 :
        train_set = torch.cat((train_set,noisy_imags_add1[:c]),dim = 0)
        test_set = torch.cat((test_set,noisy_imags_add2[:c]),dim = 0)
        #randomized the order of our images
        k_train = 100000+c
        idx = torch.randperm(100000+c)
        idx = idx[:k_train]
        train_set=train_set[idx]
        test_set=test_set[idx]

        
    return (train_set,test_set)