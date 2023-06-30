#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from han_sent_mha_attn import SentMultiHeadAttn


# In[2]:


class SentEncoder(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4, hidden_dim=512, max_sent_length=100):
        super(SentEncoder, self).__init__()

        # define the layers
        num_classes=2
        
        self.attentionLayer1 = SentMultiHeadAttn(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.attentionLayer2 = SentMultiHeadAttn(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.attentionLayer3 = SentMultiHeadAttn(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.attentionLayer4 = SentMultiHeadAttn(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.attentionLayer5 = SentMultiHeadAttn(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.attentionLayer6 = SentMultiHeadAttn(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.attentionLayer7 = SentMultiHeadAttn(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.attentionLayer8 = SentMultiHeadAttn(embedding_dim=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.mean_pool = nn.AvgPool1d(kernel_size=max_sent_length)
        self.ffnn_clf = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_classes),
            )
    def forward(self, x, _):
        # (x)input shape= (batch_size, num_of_sent, embedding_dim) 
        
        # apply multihead attention
        x= self.attentionLayer1(x)
        x= self.attentionLayer2(x)
        x= self.attentionLayer3(x)
        x= self.attentionLayer4(x)
        x= self.attentionLayer5(x)
        x= self.attentionLayer6(x)
        # x= self.attentionLayer7(x)
        # x= self.attentionLayer8(x)

        #applying average pooling
        print("after multiple layers and average pooling")
        print(x.size())
        x = self.mean_pool(x.permute(0, 2, 1))
        # (x)shape= (batch_size, embedding_dim, 1) 
        print(x.size())
        # applying linear classifier layer
        x = self.ffnn_clf(x.squeeze())#squeezed x shape= (batch_size, embedding_dim)
        # output shape=(batch_size,num_classes)
        print(x.size())

        return x, _


# In[4]:


if __name__=="__main__":
    input_dim = 256
    num_heads = 4
    hidden_dim=512
    max_sent_length=512
    mha = SentEncoder(input_dim, num_heads, hidden_dim, max_sent_length)

    # Create an example input batch
    batch_size = 16
    num_of_tokens = max_sent_length #num_of_sent
    embedding_dim = 256 #embedding_dim of each sent
    x = torch.randn(batch_size, max_sent_length, embedding_dim)

    # Apply the MHA module to the input batch
    output,_ = mha(x,0)
    print(output)

