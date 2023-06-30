#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


# In[ ]:


# handle chunking
import math
def handleChunking(tokenized_sentence):
    #tokenized_sentence=word_tokenize(long_sentence)
    total_words=len(tokenized_sentence)
    chunks_num=math.ceil(total_words/150)
    i=0
    chunks=[]
    while i<total_words:
        if total_words>=200 and (i+200)<total_words:
            chunks.append(tokenized_sentence[i:i+200])
            i=i+150
        elif total_words>=200 and (i+200)>=total_words:
            chunks.append(tokenized_sentence[i:])
            break
        elif total_words<200:
            chunks.append(tokenized_sentence)
            break
        #print("toto: ",i, total_words)
        
    #print(len(chunks))
    # for j in chunks:
    #     print("length: ",len(j))
    
    return chunks


# In[5]:


class HanInDataset(Dataset):
    def __init__(self, dataset, dict_path, max_sent_length, max_word_length, model_name):
        super(HanInDataset, self).__init__()
        

        self.texts= [doc for doc in dataset["text"]]                               # list of all the documents(text blocks)
        #print(self.texts)
        self.ids= dataset["id"].tolist()
        self.labels= dataset["label"].tolist()
        self.dict= pd.read_csv(filepath_or_buffer= dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0]).values #ndarray of only words from the glove embedding dictionary
        #print("df-dict:",self.dict) 
        self.dict=[word[0] for word in self.dict]       # list of words from the glove embedding dictionary
        #print("index_dict:",self.dict)
        self.max_allowable_full_length=7500
        self.max_sent_length= max_sent_length
        self.max_word_length= max_word_length
        self.num_classes= len(set(self.labels))
        self.tokenizer= AutoTokenizer.from_pretrained(str(model_name))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # for robert
        id=self.ids[idx]
        label=self.labels[idx]
        text=self.texts[idx]
        text=word_tokenize(text)
        text_len=len(text)
        trimmed_text=text[(text_len-self.max_allowable_full_length):]
        #---------------------- One Single text document(with multiple sentences) and it's label ------------------------
        # print("raw doc", text)
        # kind of one hot encoding : setting the index in place of word in the sentence , for unknown words place -1
        document_encode=handleChunking(trimmed_text)
        
        # print("encoded_doc", document_encode)
        
        # making all of the sentences having equal length means equal number of words at least (can have more number of words than the max_word_length)
        
        # for sentence in document_encode:
        #     if len(sentence) < self.max_word_length:
        #         extra_words=[" " for i in range(self.max_word_length-len(sentence))]
        #         sentence.extend(extra_words)
        
        # print("encoded_doc w minimal sent length", document_encode)
        
        # making the entire document having equal number of sentences at least 
        # means, each document(text block) must contain 
        # at least max_sent_length number of sentences(it can have more) if number of sentences is lower than max_sent_length 
        if len(document_encode)< self.max_sent_length:
            extra_sent=[[" " for i in range(self.max_word_length)] for _ in range(self.max_sent_length-len(document_encode))]
            document_encode.extend(extra_sent)
        
        #print("encoded_doc_w_fixed_sent_length", document_encode)
        
        # if sentences have more number of words than max_word_length or if the document contains more sentences than max_sent_length
        # trip all the words and sentences if they cross the maximum margin
        document_encode=[sentence[:self.max_word_length] for sentence in document_encode][-self.max_sent_length:] # taking last sentences means tripping oldest data and taking the recent one
        # print(document_encode)
        
        # at this point, document encode is list[list[str]]
        for i,k in enumerate(document_encode):
            document_encode[i]=" ".join(k) # joining all the word tokens for tokenizer (tokenizer expects sequence of texts)
        # at this point, document encode is list[str] and (len(document_encode) = max_sent_length)
        
        
        # for BERT, max_word_length <= 512
        wc_encoding = self.tokenizer(document_encode, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
        wc_input_ids=wc_encoding["input_ids"].unsqueeze(1)              # shape (max_sent_length, 1 , max_word_length) e.g. (150, 1, 512)
        wc_attention_mask=wc_encoding["attention_mask"].unsqueeze(1)    # shape (max_sent_length, 1 , max_word_length) e.g. (150, 1, 512)
        
        # concatenating those two tensors for making one tensor for one sample
        document_encode=torch.cat((wc_input_ids,wc_attention_mask),1)   # shape (max_sent_length, 2 , max_word_length) e.g. (150, 2, 512)
        #print(document_encode.size())# should be (max_sent_len=150,bert_input_features=2,embd_dim=512)

        
        return id, document_encode, label
    


# In[7]:


if __name__== "__main__":
    pass

