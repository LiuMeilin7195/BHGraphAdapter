import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pdb

def max_min_norm(H):
    
    min_val = H.min()  
    max_val = H.max()  

    scaled_H = (H - min_val) / (max_val - min_val)  
 
    positive_H = scaled_H + 0.0001 
    return  positive_H


def generate_G_from_H(H, variable_weight=False):  

    n_edge = H.size(1) 
     
    W = torch.ones(n_edge, dtype=H.dtype, device=H.device)
    
    DV = torch.sum(H * W, dim=1) 

    DE = torch.sum(H, dim=0)  
    DE = torch.where(DE == 0, torch.tensor(1e-10, dtype=DE.dtype, device=DE.device), DE)  
    invDE = torch.diag(1.0 / DE)  


    DV = torch.where(DV == 0, torch.tensor(1e-10, dtype=DV.dtype, device=DV.device), DV)  
    DV2 = torch.diag(torch.pow(DV, -0.5))  

    W = torch.diag(W)  

    HT = H.T  

    if variable_weight:  
        DV2_H = DV2 @ H  
        invDE_HT_DV2 = invDE @ HT @ DV2   
        return DV2_H, W, invDE_HT_DV2  
    else:  
        G = DV2 @ H @ W @ invDE @ HT @ DV2  
        return G

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))  
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor): 
        
        x = x.matmul(self.weight) 
       
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x) 
        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.3):
        super(HGNN, self).__init__()
        self.dropout = dropout
        
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, H, x):
        
        H_norm = max_min_norm(H)
        
        G = generate_G_from_H(H_norm)   
        
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x
