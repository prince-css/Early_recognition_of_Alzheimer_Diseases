#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


# In[24]:


class SentMultiHeadAttn(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, hidden_dim=512):
        super(SentMultiHeadAttn, self).__init__()

        # define the layers
        num_classes=2
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.ffnn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.layerNorm1 = LayerNorm(embedding_dim)
        self.layerNorm2 = LayerNorm(embedding_dim)
        
    def forward(self, x):
        # (x)input shape= (batch_size, num_of_sent, embedding_dim) 
        
        # apply multihead attention
        attn_output, _ = self.attention(x, x, x)
        #print(attn_output.size())
        # apply layer normalization and residual connection
        x = self.layerNorm1(attn_output + x)
        #print(x.size())
        # apply FFNN
        x = self.ffnn(x)
        #print(x.size())
        # apply linear layer
        x = self.linear(x)
        #print(x.size())
        # apply layer normalization and residual connection
        x = self.layerNorm2(x + attn_output)
        #print(x.size())
        # (x)shape= (batch_size, num_of_sent, embedding_dim) 
        return x


# In[25]:


if __name__=="__main__":
    input_dim = 256
    num_heads = 4
    mha = SentMultiHeadAttn(input_dim, num_heads)

    # Create an example input batch
    batch_size = 16
    num_of_tokens = 512 #num_of_sent
    embedding_dim = 256 #embedding_dim of each sent
    x = torch.randn(batch_size, num_of_tokens, embedding_dim)

    # Apply the MHA module to the input batch
    output = mha(x)

