import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    layers = []
    
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
    layers.append(conv_layer)
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)

def t_conv(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    layers = []
    
    t_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
    layers.append(t_conv_layer)
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                                kernel_size=3, stride=1, padding=1, batch_norm=True)
        
        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                               kernel_size=3, stride=1, padding=1, batch_norm=True)
        
    def forward(self, x):
        out_1 = F.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2


class Generator(nn.Module):
    def __init__(self, conv_dim):
        super(Generator, self).__init__()
        
        self.conv1 = conv(3, conv_dim, 4, batch_norm = False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 3, 1, 1)
        self.conv4 = conv(conv_dim*4, conv_dim*4, 3, 1, 1)
        
        self.res1 = ResidualBlock(conv_dim*4)
        self.res2 = ResidualBlock(conv_dim*4)

        self.t_conv1 = t_conv(conv_dim*4, conv_dim*4, 3, 1, 1)
        self.t_conv2 = t_conv(conv_dim*4, conv_dim*2, 3, 1, 1)
        self.t_conv3 = t_conv(conv_dim*2, conv_dim, 4)
        self.t_conv4 = t_conv(conv_dim, 3, 4, batch_norm = False)
        
        
    def forward(self, x):
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        
        x = self.res1(x)
        x = self.res2(x)

        x = F.leaky_relu(self.t_conv1(x))
        x = F.leaky_relu(self.t_conv2(x))
        x = F.leaky_relu(self.t_conv3(x))
        x = torch.tanh(self.t_conv4(x))
         
        
        return x

class Generator_R(nn.Module):                           #4 residual blocks
    def __init__(self, conv_dim):
        super(Generator_R, self).__init__()
        
        self.conv1 = conv(3, conv_dim, 4, batch_norm = False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 3, 1, 1)
        
        self.res1 = ResidualBlock(conv_dim*4)
        self.res2 = ResidualBlock(conv_dim*4)
        self.res3 = ResidualBlock(conv_dim*4)
        self.res4 = ResidualBlock(conv_dim*4)


        self.t_conv1 = t_conv(conv_dim*4, conv_dim*2, 3, 1, 1)
        self.t_conv2 = t_conv(conv_dim*2, conv_dim, 4)
        self.t_conv3 = t_conv(conv_dim, 3, 4, batch_norm = False)
        
        
    def forward(self, x):
        
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = F.leaky_relu(self.t_conv1(x))
        x = F.leaky_relu(self.t_conv2(x))
        x = torch.tanh(self.t_conv3(x))
         
        
        return x
