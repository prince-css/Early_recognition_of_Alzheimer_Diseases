#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv


# In[2]:


def matrix_mul(input, weight, bias=False):
    feature_list=[]
    
    # here input is a 3D tensor or, we can say collection of many 2D tensors
    # On the other hand, weight is a 2D tensor. 
    # So, for example: : :
    # (35,128,100) multiply by (100,100) means,multiplying (128,100) with (100,100) 35 times
    # this will yields (35,128,100) shaped tensor
    # again, for adding the bias term, which is again a 2D tensor
    # (35,128,100) to add with (1,100) first we need to expand (1,100) to (128,100)
    # then we have to add the tenosr of (128,100)size with (128,100)size 35 times
    
    for feature in input:
        print("into deep--> feature: nan? = ",torch.isnan(feature).any())
        print("into deep--> weight: nan? = ",torch.isnan(weight).any())
        
        feature= torch.mm(feature, weight)      #performing matrix multiplication
        #print("feature in mat_mul= ",feature)
        if isinstance(bias, torch.nn.parameter.Parameter):
            print("into deep--> bias: nan? = ",torch.isnan(bias).any())
            feature= feature + bias.expand(feature.size()[0], bias.size()[1])   # adding bias after expanding
        feature=torch.tanh(feature).unsqueeze(0)
        #print("feature after tanh in mat_mul= ",feature)
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


class WordAttn(nn.Module):  # The module torch.nn contains different classess that help to
                            # build neural network models. All models in PyTorch inherit from the subclass nn.Module, 
                            # which has useful methods like parameters(), __call__() and others
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttn, self).__init__()
        dict_df=pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:,1:]
        dict_len, embd_size=dict_df.shape
        # creating an extra embedding for unknown words
        unk_word=np.zeros((1,embd_size))
        dict=torch.from_numpy(np.concatenate([dict_df, unk_word], axis=0).astype(np.float))
        dict_len += 1   #for that extra one([UNK]) row
        
        #--------------------------------------------------------------------------------------------------------------------------
        # D E C L A R I N G       C U S T O M       P A R A M E T E R S     I N      T H E      A T T E N T I O N      M O D E L
        #--------------------------------------------------------------------------------------------------------------------------
        # Recent PyTorch releases just have Tensors, it came out the concept of the Variable has been deprecated.
        # Parameters are just Tensors limited to the module they are defined in (in the module constructor __init__ method).
        # They will appear inside module.parameters(). This comes handy when we build our custom modules that learn.
        # thanks to these parameters gradient descent.
        # Anything that is true for the PyTorch tensors is true for parameters, since they are tensors.
        # Additionally, if a module goes to the GPU, parameters go as well. If a module is saved parameters will also be saved.
        self.word_weight= nn.Parameter(torch.Tensor(2 * hidden_size, 2*hidden_size))
        self.word_bias= nn.Parameter(torch.Tensor(1,2*hidden_size))
        self.context_weight= nn.Parameter(torch.Tensor(2*hidden_size, 1))
        
        #----------------------------------------------------------------------------------------------------
        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        # This module is often used to store word embeddings and retrieve them using indices. 
        # The input to the module is a list of indices, and the output is the corresponding word embeddings.
        self.lookup= nn.Embedding(num_embeddings=dict_len, embedding_dim=embd_size).from_pretrained(dict) 
        #----------------------------------------------------------------------------------------------------
        
        self.gru= nn.GRU(embd_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)  # this is a method to initialize the weights of those newly created parameters 
        
    def _create_weights(self, mean=0.0, std=0.05):
        # self.word_weight.data should be taken from a normal distribution
        self.word_weight.data.normal_(mean, std)        # Normal distribution to initialize the weights
        self.context_weight.data.normal_(mean, std)     # Normal distribution to initialize the weights
        self.word_bias.data.normal_(mean, std)
        
    def forward(self, input, hidden_state):
        print("@ word attn...")
        print("input_shape= ",input.shape)
        print("word_weight_shape= ",self.word_weight.shape)
        print("word_bias_shape= ",self.word_bias.shape)
        
        print("word_bias: nan? = ",torch.isnan(self.word_bias).any())
        #print("word_bias=", self.word_bias)
        print("word_weight: nan? = ",torch.isnan(self.word_weight).any())
        
        #print(input)
        
        # N= batch size
        # L= sequence length or, max_word_length 
        # D= 2 as bidirectional
        # H_in= input_size or, embedding_dim
        # H_out= hidden_size 
        
        
        ##### in GRU default option is batch_first= False
        ##### thats why input_shape is like (max_word_length= k, batch_size= i)
        ##### But, if it was batch_first= True
        ##### it would be (batch_size= i, max_word_length= k)
        
        # input_shape       = (L, N) or (max_word_length= k, batch_size= i)
        # word_weight_shape = (hidden_size*2, hidden_size*2)
        # word_bias_shape   = (1, hidden_size*2)
        # context_weight    = (hidden_size*2, 1)-------> transposed from the beginning
        
        x=self.lookup(input)    # Input: IntTensor or LongTensor of arbitrary shape containing the "indices" to extract
                                # that means, sequence/tensor of indices to extract from the embedding dictionary
        # input to GRU unit : : :
        # x shape= (L, N, H_in) or (max_word_length= k, batch_size= i, embedding_dim)
        # h_0 or hidden_state shape= (D*num_layers, N, H_out) or (2*1, batch_size= i, hidden_size)
        
        print("x_shape= ",x.shape)                        
        h_t, h_n= self.gru(x.float(), hidden_state)     # feature output and hidden state output
        
        # Many to One GRU
        # output of GRU unit : : :
        # h_t shape= (L, N, D*H_out) or (max_word_length= k, batch_size= i, hidden_size*2)-------> etai lagbe
        # containing the ###"output features"### (h_t) from the last layer of the GRU, for each t----------> prottekta word er hidden state
        
        # h_n shape= (D*num_layers, N, H_out) or (2*1, batch_size= i, hidden_size)
        # containing the final hidden state for the entire input sequence
        
        print("h_t_shape= ",h_t.shape)
        print("h_n_shape= ",h_n.shape)
        
        
        # PAPER----> for i-th sentence
        # h_it  = h_t
        # W_w   = self.word_weight
        # b_w   = self.word_bias
        # u_it  = u_t
        # u_w   = self.context_weight
        # a_it  = a_t 
        # s_i   = s_i
        
        u_t=matrix_mul(h_t, self.word_weight, self.word_bias)
        
        
        print("u_t_shape= ",u_t.shape)
        print("context_weight_shape= ",self.context_weight.shape)
        a_t_temp=matrix_mul(u_t, self.context_weight).permute(1,0)  # <----------------- issue
        print("a_t_temp_shape= ",a_t_temp.shape)
        a_t=F.softmax(a_t_temp, dim=1)
        print("a_t by soft_shape= ",a_t.shape)
        s_i=element_wise_mul(h_t, a_t.permute(1,0))
        print("out_shape= ",s_i.shape)
        
        # print("----------------all parameters------------------")
        # print("h_t = ", h_t)
        # print("word_weight(W_w)", self.word_weight)
        # print("word_weight(b_w)", self.word_bias)
        # print("u_t = ", u_t)
        # print("context_weight(u_w)", self.context_weight)
        # print("a_t_temp = ",a_t_temp)
        # print("a_t = ",a_t)
        print("output from word_attn or s_i = ", s_i)
        
        return s_i, h_n
        


# In[4]:


if __name__=="main":
    attn=WordAttn()


# In[ ]:




