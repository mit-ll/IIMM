import torch
import torch.nn as nn


# https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
class Adapter(nn.Module):
    def __init__(self, c_in=512, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x        


# https://github.com/Chris210634/word_soups/blob/main/source/models.py
class LoraMultiheadAttention(nn.Module):
    def __init__(self, mhn, rank=4):
        super().__init__()
        # pytorch mhn stores the QKV projection matrices 
        # concatenated together along the first dimension
        in_dim = mhn.in_proj_weight.shape[0] // 3
        out_dim = mhn.in_proj_weight.shape[1]
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # only tune lora on Q and V matrices
        self.lora_Q_A = torch.nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_Q_B = torch.nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_V_A = torch.nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_V_B = torch.nn.Parameter(torch.zeros(rank, in_dim))
        
        nn.init.normal_(self.lora_Q_A, std=0.02)
        torch.nn.init.zeros_(self.lora_Q_B)
        nn.init.normal_(self.lora_V_A, std=0.02)
        torch.nn.init.zeros_(self.lora_V_B)
        
        self.mhn = mhn
        self.register_buffer('mhn_in_proj_weight', torch.clone(self.mhn.in_proj_weight.data))
        del self.mhn.in_proj_weight

        self.scaling = 0.2
        
    def forward(self, q, k, v, **kwargs):
        self.mhn.in_proj_weight = self.mhn_in_proj_weight.detach()
        self.mhn.in_proj_weight[:self.in_dim, :] += (self.lora_Q_A @ self.lora_Q_B).T * self.scaling
        self.mhn.in_proj_weight[self.in_dim*2:, :] += (self.lora_V_A @ self.lora_V_B).T * self.scaling
        
        return self.mhn(q,k,v, **kwargs)


class LoraMLP(nn.Module):
    def __init__(self, c_fc, rank=4):
        super().__init__()
       
        # add lora to mlp layers
        in_dim = c_fc.weight.shape[0]
        out_dim = c_fc.weight.shape[1]
        
        self.lora_A = torch.nn.Parameter(torch.zeros(out_dim, rank))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, in_dim))
        
        nn.init.normal_(self.lora_A, std=0.02)
        torch.nn.init.zeros_(self.lora_B)

        self.c_fc = c_fc
        self.register_buffer('c_fc_weight', torch.clone(c_fc.weight.data))
        self.bias = c_fc.bias
        self.register_buffer('weight', torch.clone(c_fc.weight.data))
        del self.c_fc.weight
        
        self.scaling = 0.2
        
    def forward(self, x, **kwargs):
        self.c_fc.weight = self.c_fc_weight.detach()
        self.c_fc.weight += (self.lora_A @ self.lora_B).T * self.scaling
        self.weight = self.c_fc.weight.data
        return self.c_fc(x, **kwargs)