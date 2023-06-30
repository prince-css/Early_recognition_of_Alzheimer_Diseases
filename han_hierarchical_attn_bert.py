#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from han_word_attn_robert import WordAttn
from han_sent_attn_dep import SentAttn


# In[ ]:


class HierAttn(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size,num_classes, word2vec_path, max_sent_length, max_word_length, embd_size, model_name):
        super(HierAttn, self).__init__()
        self.model_name         =   model_name
        self.batch_size         =   batch_size
        self.embd_size          =   embd_size
        self.word_hidden_size   =   word_hidden_size
        self.sent_hidden_size   =   sent_hidden_size
        self.max_sent_length    =   max_sent_length
        self.max_word_length    =   max_word_length
        self.word_attn_layer    =   WordAttn(word2vec_path, self.model_name, self.embd_size, word_hidden_size)
        self.sent_attn_layer    =   SentAttn(sent_hidden_size, word_hidden_size, num_classes)
        
        self._init_hidden_state()
        
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size= last_batch_size
        else:
            batch_size= self.batch_size
        #print(self.word_hidden_size, type(self.word_hidden_size))
        #print(self.sent_hidden_size, type(self.sent_hidden_size))
        self.word_hidden_state= torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state= torch.zeros(2, batch_size, self.sent_hidden_size)
        
        if torch.cuda.is_available():
            self.word_hidden_state= self.word_hidden_state.cuda()
            self.sent_hidden_state= self.sent_hidden_state.cuda()
            
    def forward(self, input):
        output_list=[]
        print("@ hier attn")
        #print(input)
        print(type(input))
        print(input.shape) # (batch_size=i, max_sent_length=j,num_of_bert_features=n, max_word_length=k)= e.g. (16, 150, 2. 512)
        
        # input_shape= (batch_size=i, max_sent_length=j,num_of_bert_features=n, max_word_length=k) means,(batch of i docs, each doc having j sentences, each sentence having n features, for each token of k tokens)
        input=input.permute(1, 0, 2, 3)    # Returns a view of the original tensor input with its dimensions permuted.
                                        # new 0th dim = old 1st dim
                                        # new 1st dim = old 0th dim
                                        # new 2nd dim = old 2nd dim 
                                        # new 3rd dim = old 3rd dim
        # input_shape= (max_sent_length=j ,batch_size=i, num_of_bert_features=n, max_word_length=k)
        for i in input:
            # print(i.shape)
            # print(i)
            # i shape= (batch_size=i, num_of_bert_features=n, max_word_length=k)
            
            output, self.word_hidden_state= self.word_attn_layer(i, self.word_hidden_state)   # i shape= (batch_size=i, num_of_bert_features=n, max_word_length=k)
            # print("out_from_word_attn",output.shape)
            output_list.append(output)
        
        output= torch.cat( output_list, 0)
        # print("in_to_sent_attn",output.shape)
        output, self.sent_hidden_state= self.sent_attn_layer(output, self.sent_hidden_state)
        # print("out_from_sent_attn",output.shape)
        # print(output)
        return output
        

