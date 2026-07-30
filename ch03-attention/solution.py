"""第 3 章整理版：注意力演进。"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, torch.nn as nn
from src.gpt import MultiHeadAttention

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_in, d_out)))
        self.W_key = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_in, d_out)))
        self.W_value = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_in, d_out)))
    def forward(self, x):
        q,k,v = x@self.W_query, x@self.W_key, x@self.W_value
        return torch.softmax(q@k.T/k.shape[-1]**0.5, dim=-1) @ v

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query, self.W_key, self.W_value = (nn.Linear(d_in,d_out,bias=qkv_bias) for _ in range(3))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
    def forward(self, x):
        b,n,_ = x.shape
        k,q,v = self.W_key(x), self.W_query(x), self.W_value(x)
        s = q@k.transpose(1,2); s.masked_fill_(self.mask.bool()[:n,:n], -torch.inf)
        return self.dropout(torch.softmax(s/k.shape[-1]**0.5, dim=-1)) @ v

def main():
    inputs = torch.tensor([[0.43,0.15,0.89],[0.55,0.87,0.66],[0.57,0.85,0.64],[0.22,0.58,0.33],[0.77,0.25,0.10],[0.05,0.80,0.55]])
    batch = torch.stack((inputs,inputs),dim=0)
    print("v1:", SelfAttention_v1(3,2)(inputs).shape)
    print("Causal:", CausalAttention(3,2,6,0.0)(batch).shape)
    print("MHA:", MultiHeadAttention(3,4,6,0.0,2)(batch).shape)

if __name__ == "__main__": main()
