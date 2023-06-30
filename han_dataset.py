#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


# In[33]:


class HanDataset(Dataset):
    def __init__(self, data_path, dict_path, max_sent_length=30, max_word_length=35):
        super(HanDataset, self).__init__()
        
        texts, labels= [], []
        
        # reading text and label from the data file
        with open(data_path) as csv_file:
            reader= csv.reader(csv_file, quotechar='"') # reader is now the entire file 
                                                        # containing many documents(a document is made of multiple sentences) and their label
                                                        # ex: [[label_0, doc_0],[label_1, doc_1],[label_3, doc_3]......]
            #print("reader: ",reader) 
            for idx, line in tqdm(enumerate(reader)):         # line=each pair of  [label_i,doc_i]
                text=""
                #print("line no.",idx,line)
                for tx in line[1:]:                     # tx is only the "doc" element
                    text += tx.lower()
                    text+=" "
                label= int(line[0])-1
                texts.append(text)
                labels.append(label)
                # if(idx == 500):
                #     break
        self.texts= texts                               # list of all the documents(text blocks)
        #print(self.texts)
        self.labels= labels
        self.dict= pd.read_csv(filepath_or_buffer= dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE, usecols=[0]).values #ndarray of only words from the glove embedding dictionary
        #print("df-dict:",self.dict) 
        self.dict=[word[0] for word in self.dict]       # list of words from the glove embedding dictionary
        #print("index_dict:",self.dict)
        self.max_sent_length= max_sent_length
        self.max_word_length= max_word_length
        self.num_classes= len(set(self.labels))
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label= self.labels[idx]
        text=self.texts[idx]
        
        #print("raw doc", text)
        # kind of one hot encoding : setting the index in place of word in the sentence , for unknown words place -1
        document_encode=[
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(sentence)] for sentence in sent_tokenize(text)
        ]
        
        #print("encoded_doc", document_encode)
        
        # making all of the sentences having equal length means equal number of words at least (can have more number of words than the max_word_length)
        for sentence in document_encode:
            if len(sentence) < self.max_word_length:
                extra_words=[-1 for i in range(self.max_word_length-len(sentence))]
                sentence.extend(extra_words)
        
        #print("encoded_doc w minimal sent length", document_encode)
        
        # making the entire document having equal number of sentences at least 
        # means, each document(text block) must contain 
        # at least max_sent_length number of sentences(it can have more) if number of sentences is lower than max_sent_length 
        if len(document_encode)< self.max_sent_length:
            extra_sent=[[-1 for i in range(self.max_word_length)] for _ in range(self.max_sent_length-len(document_encode))]
            document_encode.extend(extra_sent)
        
        #print("encoded_doc_w_fixed_sent_length", document_encode)
        
        # if sentences have more number of words than max_word_length or if the document contains more sentences than max_sent_length
        # trip all the words and sentences if they cross the maximum mergin
        document_encode=[sentence[:self.max_word_length] for sentence in document_encode][:self.max_sent_length]
        # print(document_encode)
        document_encode=np.stack(document_encode, axis=0)
        # print(document_encode)
        document_encode += 1        # adding 1 to each element to make all -1 to 0
        
        return document_encode.astype(np.int64), label
    


# In[35]:


if __name__== "__main__":
    test= HanDataset(data_path="../../dataset/data/test.csv", dict_path="../../dataset/data/glove.6B.50d.txt")
    print(test.__getitem__(idx=1))

