#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from src.utils import get_max_lengths, get_evaluation 
from han_in_dataset import HanInDataset
from han_hierarchical_attn_dep import HierAttn
from tensorboardX import SummaryWriter
from nltk import word_tokenize, sent_tokenize
from sklearn import metrics
from sklearn.metrics import classification_report 
import argparse
from tqdm.notebook import tqdm
import csv
import shutil
import numpy as np
import pandas as pd


# In[2]:


from text_with_topic_words_w_3_context_w_demog import handleTask, handleMergingCaseControlDataset


# In[3]:


def get_args():
    # for amazon_review_polarity dataset
    parser= argparse.ArgumentParser()
    parser.add_argument("--batch_size", type= int, default=128)
    parser.add_argument("--num_epochs", type= int, default= 100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="./../../dataset/data/amazon_review_polarity_csv/train.csv")
    parser.add_argument("--test_set", type=str, default="./../../dataset/data/amazon_review_polarity_csv/test.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="./../../dataset/data/glove.6B.50d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="./../../checkpoints/saved_models")
    
    args, unknown = parser.parse_known_args()
    
    return args
    


# In[4]:


def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        for idx, line in tqdm(enumerate(reader)):
            text = ""
            for tx in line[1:]:
                text += tx.lower()
                text += " "
            sent_list = sent_tokenize(text)
            sent_length_list.append(len(sent_list))

            for sent in sent_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]


# In[5]:


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
        
    try:
        print(classification_report(y_true, y_pred))
    except:
        print('An exception occurred while generating classification report')
        
    return output


# In[6]:


import nltk
def handleTopicSent(dataset: pd.DataFrame, topic_set: pd.DataFrame):
    all_topic_words= topic_set[["0","1"]].values.tolist()
    all_topic_words=[word for subset in all_topic_words for word in subset ]
    ori_len=[]
    short_len=[]
    for i,doc in tqdm(enumerate(dataset["text"])):
        temp_doc=[]
        #print("--------->",doc)
        doc=nltk.sent_tokenize(doc)
        ori_len.append(len(doc))
        for each_sent in doc:
            found=[word for word in nltk.word_tokenize(each_sent) if word in set(all_topic_words)]
            if len(found) !=0:
                #print(found)
                #print(each_sent)
                temp_doc.append(each_sent)
        short_len.append(len(temp_doc))
        #print("==========>", " ".join(temp_doc))
        dataset.loc[i, "text"]=(" ".join(temp_doc))
    ori_avg_len=sum(ori_len)/len(ori_len)
    short_avg_len=sum(short_len)/len(short_len)
    print(ori_avg_len)
    print(short_avg_len)
    return dataset
                


# In[7]:


def handleDataSupport(all_dataset:pd.DataFrame,topic_set:pd.DataFrame,with_context:bool,with_demog:bool, all:bool, window_len:int, cluster_num:int):
    #with context
    full_dataset=handleTopicSent(all_dataset, topic_set) # imported from text_with_topic_words.ipynb file
    #pure_dataset=full_dataset.dropna().reset_index(drop=True) # dropping null value
    #full_dataset=all_dataset
    # full_dataset.rename(columns={"TEXT_REPROT":"text"}, inplace=True)
    # transformed_label_list=[1 if i== cluster_num else 0 for i in full_dataset["case"].tolist()]
    # full_dataset["label"]=transformed_label_list
    #full_dataset.head(5)
    df_train, df_valid= np.split(full_dataset.sample(frac=1, random_state=42), [int(0.8*len(full_dataset)),])
    
    
    df_train.to_csv("./temp_42_tr3-.csv")
    df_train=pd.read_csv("./temp_42_tr3-.csv")
    df_train=df_train.iloc[:,1:]
    df_train=df_train.dropna().reset_index(drop=True)
    print(df_train.shape)
    
    df_valid.to_csv("./temp_42_val3-.csv")
    df_valid=pd.read_csv("./temp_42_val3-.csv")
    df_valid=df_valid.iloc[:,1:]
    df_valid=df_valid.dropna().reset_index(drop=True)
    print(df_valid.shape)
    
    #df_test=df_test.reset_index(drop=True)
    
    #print(df_test.head(5))
    return df_train, df_valid


# In[8]:


def train(opt, train_set, valid_set):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    output_file= open(opt.saved_path+ os.sep + "logs.txt", "w")
    output_file.write("Model Parameters: {}".format(vars(opt)))
    training_params={"batch_size":opt.batch_size, "shuffle": True, "drop_last": True}
    test_params={"batch_size":opt.batch_size, "shuffle": False, "drop_last": False} 
    #max_word_length, max_sent_length= get_max_lengths(opt.train_set)
    max_word_length=35
    max_sent_length=250
    training_set= HanInDataset(train_set, opt.word2vec_path, max_sent_length, max_word_length) # tuple(ndarray(encoded_doc), label)
    training_generator= DataLoader(training_set, **training_params)
    test_set = HanInDataset(valid_set, opt.word2vec_path, max_sent_length, max_word_length)
    test_generator = DataLoader(test_set, **test_params)

    
    
    # defining the model
    model = HierAttn(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                       opt.word2vec_path, max_sent_length, max_word_length)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e5
    best_epoch = 0
    model.train()   # model.train() tells your model that we are training the model. 
                    # This helps inform layers such as Dropout and BatchNorm, which are designed to behave differently during training and evaluation.
                    # For instance, in training mode, BatchNorm updates a moving average on each new batch; 
                    # whereas, for evaluation mode, these updates are frozen.
    num_iter_per_epoch = len(training_generator)
    for epoch in tqdm(range(opt.num_epochs)):
        for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            # here, feature is numpy array
            predictions = model(feature)    # predicting from the model
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epochs,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                #te_predictions = F.softmax(te_predictions)
                te_pred_ls.append(te_predictions.clone().cpu())
            
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epochs,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epochs,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            writer.add_scalar('Test/Loss', te_loss, epoch)
            writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break


# In[9]:


if __name__ == "__main__":
    with_context=True
    with_demog=False
    all=True
    window_len=2
    opt = get_args()
    
    case_dataset= pd.read_csv("./../../nx_cleaned_cases_-1_to_-365_partitioned_ori_final_w_dx_top10_notes_ch1.csv")
    print(case_dataset.shape)
    control_dataset= pd.read_csv("./../../nx_selected_notes_controls_-1_to_-365_partitioned_ori_top10_in_merging_output.csv")
    print(control_dataset.shape)
    topic_set= pd.read_csv("./../../topic_words_-1_to_-365_150_top10_partitioned_ch1.csv")
    case_dataset= case_dataset.iloc[:,1:]
    control_dataset= control_dataset.iloc[:case_dataset.shape[0],1:]
    topic_set= topic_set.iloc[:50,1:]

    #merging cases and controls in one dataframe
    case_dataset= case_dataset[["STUDYID","TEXT_REPROT","case","ethnicity","race","sex","age_index"]]
    #     print(case_dataset.head(5))
    case_dataset.insert(1, "OLDCASE", "0")
    #     print(case_dataset.columns)
    control_dataset= control_dataset[["STUDYID","OLDCASE","TEXT_REPROT","case","ethnicity","race","sex","age_index"]]
    print("cases_in_set: ",case_dataset.shape)
    print("controls_in_set: ", control_dataset.shape)
    all_dataset=pd.concat([case_dataset,control_dataset],axis=0,  ignore_index=True)



# In[10]:


all_dataset.rename(columns={"TEXT_REPROT": "text", "case": "label"}, inplace=True)


# In[11]:


all_datasett=all_dataset.iloc[:, lambda df: [2, 3]]
all_datasett= all_datasett.sample(frac=1, random_state=42).iloc[:,:].reset_index(drop=True)
df_train, df_valid=handleDataSupport(all_datasett,topic_set, with_context,with_demog, all, window_len, 0)


# In[12]:


df_valid.columns


# In[13]:


train(opt, df_train,df_valid)

