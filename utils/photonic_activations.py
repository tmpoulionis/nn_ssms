"""Photonic activation functions implemented in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PSigmoid(nn.Module):
    def __init__(self, a1: float = 0.0198, a2: float = 0.07938, x0: float = 1.26092, d: float = 0.48815):
        super().__init__()
        self.a1 = a1
        self.a2 = a2
        self.x0 = x0
        self.d = d

    def forward(self, x):
        z = (x - self.x0)/self.d
        z = torch.clamp(z, -80, 80) # Prevent overflow
        
        return self.a2 + (self.a1 - self.a2)/(1 + torch.exp(z))
    
class PSinusoidal(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.where(((x > 0) & (x < 1)), torch.pow(torch.sin(torch.pi/2*x), 2), torch.where(x > 1, 1, 0))
    
class PTanhLike(nn.Module): 
    def __init__(self, a: float = 0.24057, b: float = 0.34184, c: float = 1.74544, d:float = -1.65912, e: float = 3.2698, x0: float = -0.30873):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.x0 = x0
        
    def forward(self, x):
        return self.a + (self.d + self.b*torch.sinh(x - self.x0)) / (self.e + self.c*torch.cosh(x - self.x0))

class PELULike(nn.Module):
    def __init__(self, a: float = 0.0368, b: float = 0.18175, c: float = -0.01957, x0: float = 0.37042, scale: float = 5.778):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.x0 = x0
        self.scale = scale
        
    def forward(self, x):
        out = torch.where(x>=self.x0, self.b*(x - self.x0) + self.c, self.a*(torch.exp(x - self.x0) - 1) + self.c)
        return out * self.scale
    
    def inverse(self, y):
        y_unscaled = y / self.scale

        # Linear region
        x_lin = (y_unscaled - self.c)/self.b + self.x0
        
        # Exponential region
        x_exp = torch.log((y_unscaled - self.c)/self.a + 1) + self.x0
        
        return torch.where(y_unscaled >= self.c, x_lin, x_exp)

class NNPELULike(nn.Module):
    def __init__(self, a: float = 0.0368, b: float = 0.18175, c: float = -0.01957, x0: float = 0.37042, scale: float = 5.778):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.x0 = x0
        self.scale = scale
        
    def forward(self, x):
        out = torch.where(x>=self.x0, self.b*(x - self.x0) + self.c, self.a*(torch.exp(x - self.x0) - 1) + self.c)
        out = out * self.scale
        return torch.clamp(out, min=0.0)
    
    def inverse(self, y):
        y_unscaled = y / self.scale

        # Linear region
        x_lin = (y_unscaled - self.c)/self.b + self.x0
        
        # Exponential region
        x_exp = torch.log((y_unscaled - self.c)/self.a + 1) + self.x0
        
        return torch.where(y_unscaled >= self.c, x_lin, x_exp)
    
class PInvELU(nn.Module):
    def __init__(self, a: float = 0.02395, b: float = 0.15568, c: float = 0.08616, d: float = 0.04855, x0: float = -0.2):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x0 = x0
        
    def forward(self, x):
        return torch.where(x<=self.x0, self.b*(x - self.x0) + self.c, self.a/(torch.exp(self.x0 - x) + 1) + self.d)

class PDSinSq(nn.Module):
    def __init__(self, a: float = 1.7917, b: float = 0.8571, c: float = 0.2514, d: float = 1.1066, e: float = 0.1416, x0: float = 0.9807):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.x0 = x0
    
    def forward(self, x):
        x = x.clamp(-1.9, 1.9)
        return torch.where(x >= 0, self.a*torch.pow(torch.sin(self.d*(x + self.x0)), 2) + self.c, self.b*torch.pow(torch.sin(self.e*(-x + self.x0)), 2) + self.c)
    
class PReSin(nn.Module):
    def __init__(self, a: float = 0.23299, b: float = 0.00047, c: float = 0.01692, d: float = -0.71482, x0: float = 0.44184):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x0 = x0
        
    def forward(self, x):
        return torch.where(x>=self.x0, self.a*torch.pow(torch.sin(self.d*(x - self.x0)), 2) + self.c, self.a*(x - self.x0) + self.c)
    
class PExpSin(nn.Module):
    def __init__(self, a: float = 1, b: float = 1, c: float = 0, d: float = 1, x0: float = 0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x0 = x0
        
        
    def forward(self, x):
        return torch.where(x>=self.x0, self.a*torch.pow(torch.sin(self.d*(x - self.x0)), 2) + self.c, self.b*torch.exp((x) - 1) + self.c)