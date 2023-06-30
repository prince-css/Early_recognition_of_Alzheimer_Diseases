#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
import numpy as np
import nltk
import math

import random
from sklearn.model_selection import train_test_split
import pandas as pd


# In[9]:


#  Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
#     installed)
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        
        # safe to call this function even if cuda is not available
        torch.cuda.manual_seed_all(seed)
        
    if is_tf_available():
        import tensorflow as tf
        
        tf.random.set_seed(seed)
set_seed(1)


# In[10]:


# convert our dataframe into a torch Dataset for BERT


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self,df, tokenizer):
        # here both texts and labels must have to be list (indexable)
        self.texts=[each_sent for each_sent in df["text"]]
#         self.texts=[tokenizer(text, padding="max_length", 
#                               max_length=512, 
#                               truncation=True,
#                               add_special_tokens=False, 
#                               return_tensors="pt") for text in df["text"]] #list of tensor containing input_id, attention_mask 
        #print(self.texts)
        self.labels=df["label"].tolist() # list of labels
        self.clusters=df["case"].tolist() # list of case to use in testing at last
        self.studyid=df["STUDYID"].tolist()
        self.oldcase=df["OLDCASE"].tolist()
    
    def __len__(self):
        return len(self.labels)
        
    def classes(self):
        return self.labels
    
    def get_batch_labels(self, idx):
        return np.array(int(self.labels[idx]))
    
    def get_batch_texts(self, idx):
        return self.texts[idx]  
    
    def get_batch_clusters(self,idx):  
        return np.array(int(self.clusters[idx]))
    
    def get_batch_studyid(self, idx):
        return self.studyid[idx]  
    
    def get_batch_oldcase(self,idx):  
        return np.array(int(self.oldcase[idx]))
    
    def __getitem__(self, idx):
        batch_splitted_texts=self.get_batch_texts(idx)
        batch_y= self.get_batch_labels(idx)
        batch_clusters=self.get_batch_clusters(idx)
        batch_studyid=self.get_batch_studyid(idx)
        batch_oldcase=self.get_batch_oldcase(idx)
        return batch_splitted_texts, batch_y, batch_clusters, batch_studyid, batch_oldcase  # tuple
        
    
    


# In[11]:


# defining model
"""
    BERT model outputs two variables:

        1) The first variable, which we named _ in the code above, contains the embedding vectors of all of the tokens in a sequence.
        2) The second variable, which we named pooled_output, contains the embedding vector of [CLS] token. For a text classification task, 
        it is enough to use this embedding as an input for our classifier.

    We then pass the pooled_output variable into a linear layer with ReLU activation function. 
    At the end of the linear layer, we have a vector of size 2, each corresponds to a category of our labels (case or control).
    
"""

from torch import nn
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    
    def __init__(self, dropout=0.4, model_name="bert-base-cased",input_size=768, h_unit=128, num_of_labels=2):
        super(BertClassifier, self).__init__()
        self.model_name=model_name
        self.bert= BertModel.from_pretrained(model_name)
        self.h_unit=h_unit
        self.input_size=input_size
        self.lstm=nn.LSTM(input_size=input_size, hidden_size=h_unit,num_layers=1,bidirectional=False, batch_first=True)
        self.linear=nn.Linear(h_unit,num_of_labels)
        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()
        
    def forward(self, input_id_stack,attention_mask_stack,  batch_size, chunks_len):
        # the input dimension will be (batch_size, number of chunk, 512)
        all_pooled_output=[]
        print(chunks_len)
        #print(len(chunks_len))
        max_len=max(chunks_len)
        for i in range(0,batch_size):
            sequence_output, pooled_output=self.bert(input_ids=(input_id_stack[i][:chunks_len[i],:]).to(torch.long),
                                                     attention_mask=(attention_mask_stack[i][:chunks_len[i],:]).to(torch.long), 
                                                     return_dict=False)

            all_pooled_output.append(pooled_output)
            del pooled_output
        output=[]
        for i in all_pooled_output:
            lstm_out,_=self.lstm(i.unsqueeze(0).to(torch.float))
            #lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            lstm_last=lstm_out[:,-1,:]
            linear_out=self.linear(lstm_last)
            output.append(self.relu(linear_out))
            
        output=torch.cat(output)
        
        return output


# In[1]:


import torch
a=torch.randint(0,10,(3,3,10))
print(a.shape)
print(a[0].shape)
b=a.reshape(a.shape[0],a.shape[1]*a.shape[2])
print(b.shape)
c=a.reshape(a.shape[0],-1)
print(c.shape)


# In[15]:


def save_checkpoint(state, epoch,model_num,dataset_len):
    PATH="./checkpoints/model_{}_w_{}_dataset_at_epoch_{}.pth.tar".format(model_num,dataset_len,epoch)
    torch.save(state, PATH)



# In[16]:


#training
def train(bert_model, train_data, valid_data, lr, epochs,tokenizer,optimizer,criterion,model_num,dataset_len,window_size):
    
    print("Training RoBERT model...")
    

    train, valid=TorchDataset(train_data, tokenizer), TorchDataset(valid_data, tokenizer)
    #print(valid)
    train_dataloader=torch.utils.data.DataLoader(train, batch_size=3)
    valid_dataloader=torch.utils.data.DataLoader(valid, batch_size=3)
    #print(train_dataloader)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    

    
    if use_cuda:

        bert_model = bert_model.cuda()
        criterion = criterion.cuda()

    #---------------------------------------------------------------------------------------------
    # T R A I N I N G      T H E       C L A S S I F I E R
    #---------------------------------------------------------------------------------------------
    train_loss=[]
    valid_loss=[]
    train_acc=[]
    valid_acc=[]
    for epoch_num in range(epochs):
        
        ############## for training step ################
        total_acc_train=0
        total_loss_train=0
        
        
        
        out_bound_t=0
        for x_data, x_label, x_cluster, x_studyid, x_oldcase in tqdm(train_dataloader):
        
            # working with whole batch in each iteration
            x_label=torch.tensor(x_label)
            x_studyid=torch.tensor(x_studyid)
            x_oldcase=torch.tensor(x_oldcase)
            
            #print(x_label.shape)
        
            #generating the embedding
            #here, x_data is composed of 3 sentences
            all_input_embed_stack, x_data_chunked_len, out_bound_c, x_label, x_studyid, x_oldcase= handleEmbeddingGenerationOfChunks(x_data, 
                                                                                                                                     x_label, 
                                                                                                                                     x_studyid, 
                                                                                                                                     x_oldcase, 
                                                                                                                                     bert_model,
                                                                                                                                     tokenizer
                                                                                                                                     )
            out_bound_t = out_bound_t + out_bound_c
            
            output = classifier_model(all_input_embed_stack.to(torch.float), x_data_chunked_len)
            #print(output.shape)
            x_label=x_label.unsqueeze(1)
            #print(x_label.shape)
            
            #-------------------------------------------------------------------------------
            batch_loss= criterion(output, x_label.to(torch.float32))
            #print(batch_loss)
            total_loss_train +=batch_loss.item()
            
            rounded_output=(output>0.5).float()
            #acc=torch.sum(rounded_output == train_label_final)
            acc=(rounded_output== x_label).sum().item()
            total_acc_train +=acc
            #print("batch_acc=",acc)
            classifier_model.zero_grad()
            a = list(classifier_model.parameters())[0].clone()
            batch_loss.backward()
            optimizer.step()
            b = list(classifier_model.parameters())[0].clone()
            #print("updating weights: ",torch.equal(a.data, b.data))
            #print("grad:",list(classifier_model.parameters())[0].grad)

        print(out_bound_t)
        

        ############## for validation step #############    
        total_acc_valid=0
        total_loss_valid=0
        Y_pred=[]
        Y_valid=[]
        out_bound_v=0
        with torch.no_grad():
            for v_data, v_label, v_cluster, v_studyid, v_oldcase in tqdm(valid_dataloader):
                classifier_model.mode="valid"
                v_label=torch.tensor(v_label)
                v_studyid=torch.tensor(v_studyid)
                v_oldcase=torch.tensor(v_oldcase)
                
                all_input_embed_stack, v_data_chunked_len, out_bound_c, v_label, v_studyid, v_oldcase= handleEmbeddingGenerationOfChunks(v_data, 
                                                                                                                   v_label, 
                                                                                                                   v_studyid, 
                                                                                                                   v_oldcase, 
                                                                                                                   bert_model,
                                                                                                                   tokenizer
                                                                                                                   )
                out_bound_v = out_bound_v + out_bound_c
                
                output = classifier_model(all_input_embed_stack.to(torch.float), v_data_chunked_len)
                #print(output.shape)
                v_label=v_label.unsqueeze(1)
                #print(v_label.shape)
                
                #-------------------------------------------------------------------------------
                batch_loss= criterion(output, v_label.to(torch.float32))
                #print(batch_loss)
                total_loss_valid +=batch_loss.item()
                rounded_output=(output>0.5).float()
                #acc=torch.sum(rounded_output == valid_label)
                acc=(rounded_output== v_label).sum().item()
                #print(acc)
                total_acc_valid +=acc
                #print(rounded_output, v_label)
                Y_pred = Y_pred + [i.item() for i in rounded_output]
                Y_valid = Y_valid + [i.item() for i in v_label]
                
        #print all stats
        
        print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_data)-out_bound_t): .3f} \
                | Train Accuracy: {total_acc_train / (len(train_data)-out_bound_t): .3f} \
                | Val Loss: {total_loss_valid / (len(valid_data)-out_bound_v): .3f} \
                | Val Accuracy: {total_acc_valid / (len(valid_data)-out_bound_v): .3f}')
        
        
        train_acc.append(total_acc_train / (len(train_data)-out_bound_t))
        train_loss.append(total_loss_train / (len(train_data)-out_bound_t))
        valid_acc.append(total_acc_valid / (len(valid_data)-out_bound_v))
        valid_loss.append(total_loss_valid / (len(valid_data)-out_bound_v))
        df_stat=pd.DataFrame({"train_acc":train_acc, "train_loss":train_loss, "valid_acc":valid_acc, "valid_loss":valid_loss})
        print(Y_pred)
        print(Y_valid)
        
        # if(epoch_num % 100 == 0):
            #Y_pred=np.array([item.numpy()[0] for batch in Y_pred for item in batch ])
            #Y_valid=np.array([item.numpy()[0] for batch in Y_valid for item in batch ])
        cr=classification_report(Y_valid, Y_pred)
        print(cr)
        cm=confusion_matrix(Y_valid, Y_pred)   
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()
        # plt.show()
        print(cm)
        
#         for param in classifier_model.parameters():
#             print("all parameters:")
#             print(param)
            
        save_checkpoint(
                        state={
                            'epoch': epoch_num + 1,
                            'state_dict': classifier_model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        },
                        epoch=epoch_num+1,
                        model_num=model_num, 
                        dataset_len=dataset_len
                        )
        df_stat.to_csv("./training_stats.csv")
    return bert_model, classifier_model


# In[17]:


# handle chunking
def handleChunking(tokenized_sentence):

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
   


# In[18]:


#embedding generation
#-------------------------------------------------------------------------------------------
# P R E P A R I N G     B A T C H     W I T H     T H E     E M B E D D I N G S
#-------------------------------------------------------------------------------------------
def handleEncodingGenerationOfChunks(x_data, x_label,x_studyid, x_oldcase, bert_model, tokenizer):
        
    x_data_chunked=[]
    x_new_label=[]
    x_new_studyid=[]
    x_new_oldcase=[]
    chunks_length=[]
    wc_encoding=[]
    extra_part=[]
    batch_input_ids=[]
    batch_attention_mask=[]
    out_bound=0
    for i,x_i_data in enumerate(x_data):
        #---------------------------------------
        ##########    C H U N K I N G ##########
        #---------------------------------------
        
        x_i_chunks=handleChunking(x_i_data.split())
        if len(x_i_chunks)> 150:
            out_bound=out_bound+1
            # x_label=torch.cat((x_label[:i],x_label[i+1:]))
            # x_studyid=torch.cat((x_studyid[:i],x_studyid[i+1:]))
            # x_oldcase=torch.cat((x_oldcase[:i],x_oldcase[i+1:]))
            continue
        elif len(x_i_chunks)<= 150:
            x_new_label.append(x_label[i])
            x_new_oldcase.append(x_oldcase[i])
            x_new_studyid.append(x_studyid[i])
        # input_embed = torch.tensor([[0 for j in range(768)] for i in range(1)], dtype=torch.float)
        
        for i,chunk in enumerate(x_i_chunks):
            
            x_i_chunks[i]=" ".join(chunk)
            #wc_encoding = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512) 
            # wc_input_ids = wc_encoding['input_ids']
            # wc_input_mask = wc_encoding['attention_mask']            
            # wc_outputs = bert_model(wc_input_ids, wc_input_mask)
            # hidden_layer = wc_outputs[1] # taking the pooled representation
            
            #input_embed=torch.cat((input_embed,hidden_layer),0)
        
        wc_encoding = tokenizer(x_i_chunks, return_tensors='pt', padding="max_length", truncation=True, max_length=512) 
        #print(wc_encoding["input_ids"].shape)
        wc_input_ids=wc_encoding["input_ids"]
        wc_attention_mask=wc_encoding["attention_mask"]
        batch_input_ids.append(wc_input_ids)
        batch_attention_mask.append(wc_attention_mask)
        
        #print("input_embed_size_before: ", input_embed.shape)
        #input_embed=input_embed[1:,:]
        # input_embed=input_embed.unsqueeze(0)
        #print(input_embed.shape)
        #x_data_chunked.append(input_embed)
        chunks_length.append(len(x_i_chunks))
    #--------------------------------------------------------------------------------------------------------------------
    ##################   P A D D I N G     E A C H     E L E M E N T     O F     T H E     B A T C H   ##################
    #--------------------------------------------------------------------------------------------------------------------
    batch_input_ids_stack=torch.tensor([])
    batch_attention_mask_stack=torch.tensor([])
    if len(chunks_length)>0:
        max_len=max(chunks_length)
        for i,c in enumerate(batch_input_ids):
            batch_input_ids_np=np.array(batch_input_ids[i])
            batch_attention_mask_np=np.array(batch_attention_mask[i])
            #all_input_embed_np_arr=x_data_chunked[i].cpu().detach().numpy()
            extra_part= np.zeros(((max_len-chunks_length[i]),512))
            total_input_ids=np.concatenate((batch_input_ids_np,extra_part))
            total_attention_mask=np.concatenate((batch_attention_mask_np,extra_part))
            batch_input_ids[i]=torch.from_numpy(total_input_ids)
            batch_attention_mask[i]=torch.from_numpy(total_attention_mask)
            #print(x_data_chunked[i].shape)
    

        
        batch_input_ids_stack=torch.stack(tuple(batch_input_ids),0)
        batch_attention_mask_stack=torch.stack(tuple(batch_attention_mask),0)
    # else:
    #     batch_input_ids_stack=False
    #     batch_attention_mask_stack=False
    #print(batch_input_ids_stack.shape)
    #print(batch_attention_mask_stack.shape)
    #print(all_input_embed_stack.shape)
    #print(" Embeddings generated successfully .... ")
    del wc_encoding
    del extra_part
    del batch_input_ids
    del batch_attention_mask
    return batch_input_ids_stack, batch_attention_mask_stack, chunks_length, out_bound, torch.tensor(x_new_label),torch.tensor(x_new_studyid), torch.tensor(x_new_oldcase)


# In[19]:


# defining training process using mini batch

from torch.optim import Adam
from tqdm import tqdm

def finetuneBert(bert_model,train_data, valid_data, learning_rate, epochs,tokenizer,optimizer,criterion,model_num,dataset_len):
    
    
    

    train, valid=TorchDataset(train_data, tokenizer), TorchDataset(valid_data, tokenizer)
    #print(valid)
    train_dataloader=torch.utils.data.DataLoader(train, batch_size=3, shuffle= True)
    valid_dataloader=torch.utils.data.DataLoader(valid, batch_size=3, shuffle= True)
    #print(train_dataloader)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    

    
    if use_cuda:

        bert_model = bert_model.cuda()
        #classifier_model=classifier_model.cuda()
        criterion = criterion.cuda()

    #---------------------------------------------------------------------------------------------
    # T R A I N I N G      T H E       C L A S S I F I E R
    #---------------------------------------------------------------------------------------------
    train_loss=[]
    valid_loss=[]
    train_acc=[]
    valid_acc=[]
    for epoch_num in range(epochs):
        #classifier_model.mode="train"
        ############## for training step ################
        total_acc_train=0
        total_loss_train=0
        
        
        
        out_bound_t=0
        for x_data, x_label, x_cluster, x_studyid, x_oldcase in tqdm(train_dataloader):
        
            # working with whole batch in each iteration
            x_label=torch.tensor(x_label)
            x_studyid=torch.tensor(x_studyid)
            x_oldcase=torch.tensor(x_oldcase)
            
            #print(x_label.shape)
        
            #generating the embedding
            #here, x_data is composed of 3 sentences
            batch_input_ids_stack,batch_attention_mask_stack, chunks_length, out_bound_c, x_label, x_studyid, x_oldcase= handleEncodingGenerationOfChunks(x_data, 
                                                                                                                                     x_label, 
                                                                                                                                     x_studyid, 
                                                                                                                                     x_oldcase, 
                                                                                                                                     bert_model,
                                                                                                                                     tokenizer
                                                                                                                                     )
            out_bound_t = out_bound_t + out_bound_c
            batch_size=batch_input_ids_stack.shape[0]
            if(batch_size>0):
                
                output = bert_model(batch_input_ids_stack,batch_attention_mask_stack, batch_size, chunks_length)
                #print(output.shape)
                #x_label=x_label.squeeze(1)
                #print(x_label.shape)
                
                #-------------------------------------------------------------------------------
                batch_loss= criterion(output, x_label.long())
                #print(batch_loss)
                total_loss_train +=batch_loss.item()
                
                acc=(output.argmax(dim=1)==x_label).sum().item()
                #acc=(rounded_output== x_label).sum().item()
                total_acc_train +=acc
                #print("batch_acc=",acc)
                bert_model.zero_grad()
                a = list(bert_model.parameters())[0].clone()
                
                #calculating the gradients
                batch_loss.backward()
                
                #adjusting the weights
                optimizer.step()
                b = list(bert_model.parameters())[0].clone()
                #print("updating weights: ",torch.equal(a.data, b.data))
                #print("grad:",list(classifier_model.parameters())[0].grad)

        print(out_bound_t)
        

        ############## for validation step #############    
        total_acc_valid=0
        total_loss_valid=0
        Y_pred=[]
        Y_valid=[]
        out_bound_v=0
        with torch.no_grad():
            for v_data, v_label, v_cluster, v_studyid, v_oldcase in tqdm(valid_dataloader):
                #classifier_model.mode="valid"
                v_label=torch.tensor(v_label)
                v_studyid=torch.tensor(v_studyid)
                v_oldcase=torch.tensor(v_oldcase)
                
                batch_input_ids_stack,batch_attention_mask_stack, chunks_length, out_bound_c, v_label, v_studyid, v_oldcase= handleEncodingGenerationOfChunks(v_data, 
                                                                                                                                                              v_label, 
                                                                                                                                                              v_studyid, 
                                                                                                                                                              v_oldcase, 
                                                                                                                                                              bert_model,
                                                                                                                                                              tokenizer
                                                                                                                                                            )
                out_bound_v = out_bound_v + out_bound_c
                batch_size=batch_input_ids_stack.shape[0]
                if(batch_size>0):
                
                    output = bert_model(batch_input_ids_stack,batch_attention_mask_stack, batch_size, chunks_length)
                    #print(output.shape)
                    #v_label=v_label.squeeze(1)
                    #print(v_label.shape)
                    
                    #-------------------------------------------------------------------------------
                    batch_loss= criterion(output, v_label.long())
                    #print(batch_loss)
                    total_loss_valid +=batch_loss.item()
                    #rounded_output=(output>0.5).float()
                    #acc=torch.sum(rounded_output == valid_label)
                    acc=(output.argmax(dim=1)==v_label).sum().item()
                    #print(acc)
                    total_acc_valid +=acc
                    #print(rounded_output, v_label)
                    
                    Y_pred = Y_pred + [i.item() for i in output.argmax(dim=1)]
                    Y_valid = Y_valid + [i.item() for i in v_label]
                
        #print all stats
        
        print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_data)-out_bound_t): .3f} \
                | Train Accuracy: {total_acc_train / (len(train_data)-out_bound_t): .3f} \
                | Val Loss: {total_loss_valid / (len(valid_data)-out_bound_v): .3f} \
                | Val Accuracy: {total_acc_valid / (len(valid_data)-out_bound_v): .3f}')
        
        
        train_acc.append(total_acc_train / (len(train_data)-out_bound_t))
        train_loss.append(total_loss_train / (len(train_data)-out_bound_t))
        valid_acc.append(total_acc_valid / (len(valid_data)-out_bound_v))
        valid_loss.append(total_loss_valid / (len(valid_data)-out_bound_v))
        df_stat=pd.DataFrame({"train_acc":train_acc, "train_loss":train_loss, "valid_acc":valid_acc, "valid_loss":valid_loss})
        print(Y_pred)
        print(Y_valid)
        
        # if(epoch_num % 100 == 0):
            #Y_pred=np.array([item.numpy()[0] for batch in Y_pred for item in batch ])
            #Y_valid=np.array([item.numpy()[0] for batch in Y_valid for item in batch ])
        cr=classification_report(Y_valid, Y_pred)
        print(cr)
        cm=confusion_matrix(Y_valid, Y_pred)   
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()
        # plt.show()
        print(cm)
        
#         for param in classifier_model.parameters():
#             print("all parameters:")
#             print(param)
            
        save_checkpoint(
                        state={
                            'epoch': epoch_num + 1,
                            'state_dict': bert_model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        },
                        epoch=epoch_num+1,
                        model_num=model_num, 
                        dataset_len=dataset_len
                        )
        df_stat.to_csv("./training_stats.csv")
    return bert_model
        
      


# In[3]:


import torch
output = torch.randn(10, 120).float()
target = torch.FloatTensor(10).uniform_(0, 120).long()
print(output.shape)
print(target.shape)


# In[20]:


def evaluate(bert_model,classifier_model, test_data, tokenizer, window_size):
    #print(test_data)
    test=TorchDataset(test_data, tokenizer)
    print("chukuchuku")
    #test.__len__()
    test_dataloader=torch.utils.data.DataLoader(test, batch_size=3)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()
        
    total_acc_test=0
    Y_pred=[]
    Y_test=[]
    Y_cluster=[]
    all_studyid=[]
    all_oldcase=[]
    

 
    
    total_acc_test=0
    out_bound_t=0
    with torch.no_grad():
        for t_data, t_label, t_cluster, t_studyid, t_oldcase in tqdm(test_dataloader):
            classifier_model.mode="test"
            t_label=torch.tensor(t_label)
            t_studyid=torch.tensor(t_studyid)
            t_oldcase=torch.tensor(t_oldcase)
            
            all_input_embed_stack, t_data_chunked_len, out_bound_c, t_label, t_studyid, t_oldcase= handleEmbeddingGenerationOfChunks(t_data, 
                                                                                                                                     t_label, 
                                                                                                                                     t_studyid, 
                                                                                                                                     t_oldcase,
                                                                                                                                     bert_model,
                                                                                                                                     tokenizer
                                                                                                                                     )
            out_bound_t = out_bound_t + out_bound_c
            
            output = classifier_model(all_input_embed_stack.to(torch.float), t_data_chunked_len)
            #print(output.shape)
            t_label=t_label.unsqueeze(1)
            #print(t_label.shape)
            
            #-------------------------------------------------------------------------------
            
            rounded_output=(output>0.5).float()
            #acc=torch.sum(rounded_output == test_label)
            acc=(rounded_output== t_label).sum().item()
            #print(acc)
            total_acc_test +=acc
            #print(rounded_output, t_label)
            Y_pred = Y_pred + [i.item() for i in rounded_output]
            Y_test = Y_test + [i.item() for i in t_label]
            all_studyid= all_studyid + t_studyid.tolist()
            all_oldcase= all_oldcase + t_oldcase.tolist()
            #print("Y_test=",Y_test)
    
            
    print(f'Test Accuracy: {total_acc_test / (len(test_data)-out_bound_t): .3f}')
    print(Y_pred)
    print(Y_test)

    #print(len(Y_pred))
    #print(len(Y_test))
    #print(len(all_studyid))
    #print(len(all_oldcase))
    cr=classification_report(Y_test, Y_pred)
    print(cr)
    cm=confusion_matrix(Y_test, Y_pred)   
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()
    print(cm)
    debug_df=pd.DataFrame({"STUDYID":all_studyid,
                           "OLDCASE":all_oldcase,
                           "Y_test": Y_test, 
                           "Y_pred": Y_pred})
    
    return debug_df


# In[21]:


def main():
    EPOCHS= 1
    #MODEL_NAME="bert-base-cased"
    MODEL_NAME="emilyalsentzer/Bio_ClinicalBERT"
    full_dataset=pd.read_csv("./../generated_files/dataset_75_tw_-1_to_-365_w_age.csv")
    full_dataset=full_dataset.iloc[:30,1:]
    full_dataset.head(5)


    ################# Dropping Null Values ###################
    a=full_dataset[(full_dataset["TEXT_REPROT"].isna()) & (full_dataset["case"]==1)]
    b=full_dataset[(full_dataset["TEXT_REPROT"].isna()) & (full_dataset["case"]==0)]
    print(a.shape, b.shape)

    pure_dataset=full_dataset.dropna().reset_index(drop=True)
    pure_dataset.head(5)
    pure_dataset.rename(columns={"TEXT_REPROT":"text", "case":"label"}, inplace=True)
    np.random.seed(112)
    df_train, df_valid, df_test = np.split(pure_dataset.sample(frac=1, random_state=42), [int(0.8*len(pure_dataset)), int(0.9*len(pure_dataset))])

    print(len(df_train), len(df_valid), len(df_test))
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model=BertClassifier(model_name=MODEL_NAME)
    LR=1e-6
    train(model, df_train, df_valid, LR, EPOCHS,tokenizer)
    Y_pred, Y_test=evaluate(model,df_test,tokenizer)


# In[ ]:




