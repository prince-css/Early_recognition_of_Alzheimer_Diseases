#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


def matrix_mul(input, weight, bias=False):
    feature_list=[]
    for feature in input:
        feature= torch.mm(feature, weight)      #performing matrix multiplication
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature= feature + bias.expand(feature.size()[0], bias.size()[1])
        feature=torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
        
    return torch.cat(feature_list, 0).squeeze() 

def element_wise_mul(input1, input2):
    feature_list=[]
    for feature1, feature2 in zip(input1, input2):
        feature2= feature2.unsqueeze(1).expand_as(feature1)
        feature=feature1*feature2
        feature_list.append(feature.unsqueeze(0))
    out= torch.cat(feature_list, 0)
    return torch.sum(out, 0).unsqueeze(0)


# In[3]:


class SentAttn(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
        super(SentAttn, self).__init__()
        #--------------------------------------------------------------------------------------------------------------------------
        # D E C L A R I N G       C U S T O M       P A R A M E T E R S     I N      T H E      A T T E N T I O N      M O D E L
        #--------------------------------------------------------------------------------------------------------------------------
        # Recent PyTorch releases just have Tensors, it came out the concept of the Variable has been deprecated.
        # Parameters are just Tensors limited to the module they are defined in (in the module constructor __init__ method).
        # They will appear inside module.parameters(). This comes handy when we build our custom modules that learn.
        # thanks to these parameters gradient descent.
        # Anything that is true for the PyTorch tensors is true for parameters, since they are tensors.
        # Additionally, if a module goes to the GPU, parameters go as well. If a module is saved parameters will also be saved.
        self.sent_weight=nn.Parameter(torch.Tensor(2* sent_hidden_size, 2*sent_hidden_size))
        self.sent_bias= nn.Parameter(torch.Tensor(1, 2*sent_hidden_size))
        self.context_weight= nn.Parameter(torch.Tensor(2*sent_hidden_size, 1))
        
        self.gru=nn.GRU(2* word_hidden_size, sent_hidden_size, bidirectional=True)
        self.linear= nn.Linear(2* sent_hidden_size, num_classes)
        
        
        self._create_weights(mean=0.0, std=0.05)
        
    def _create_weights(self, mean=0.0, std=0.05):
        # self.word_weight.data should be taken from a normal distribution
        self.sent_weight.data.normal_(mean, std)        # Normal distribution to initialize the weights
        self.context_weight.data.normal_(mean, std)     # Normal distribution to initialize the weights
        self.sent_bias.data.normal_(mean, std)
        
    def forward(self, input, hidden_state):
        print("@ sent attn...")
        print("input_shape= ", input.shape)
        f_output, h_output= self.gru(input, hidden_state)
        print("f_output shape= ", f_output.shape)
        print("h_output shape= ", h_output.shape)
        print("sent_weight shape= ", self.sent_weight.shape)
        print("sent_bias shape= ", self.sent_bias.shape)
        x= matrix_mul(f_output, self.sent_weight, self.sent_bias)
        print("x_1 shape= ", x.shape)
        x=matrix_mul(x, self.context_weight).permute(1,0)
        print("context_weight shape= ", self.context_weight.shape)
        print("x_2 shape= ", x.shape)
        x= F.softmax(x,  dim=1)
        print("x_2_soft shape= ", x.shape)
        x= element_wise_mul(f_output, x.permute(1, 0)).squeeze(0)
        print("x_3= ", x.shape)
        output= self.linear(x)
        print("output shape= ", output.shape)
        print(output)
        
        return output, h_output
        
        


# In[4]:


if __name__== "main":
    abc= SentAttn()

