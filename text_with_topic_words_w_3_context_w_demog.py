#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm_notebook as tqdm


# In[2]:


TOTAL_WORDS=[]


# In[3]:


def handleEthnicity(encoded_ethnicity):
        eth_dict={0: "not hispanic", 1: "hispanic",2: "unknown"}
        return eth_dict.get(encoded_ethnicity)


# In[4]:


def handleRace(encoded_race):
    race_dict={0:"black", 1:"white", 2:"other",3:"pacific islander",4:"hispanic",5:"multirace",6:"asian",7:"american indian",8:"unknown"}
    return race_dict.get(encoded_race)


# In[5]:


def handleSex(encoded_sex):
    sex_dict={0:"male", 1:"female"}
    return sex_dict.get(encoded_sex)


# In[6]:


def handleAgeDivision(age):
    if age>=60 and age<70:
        return "sixties"
    elif age>=70 and age<80:
        return "seventies"
    elif age>=80 and age<90:
        return "eighties"
    elif age>=90 and age<100:
        return "nineties"
    else:
        return "hundrades"


# In[7]:


# Initially merged case-control row-wise by their setid to know how many exact sets are present
# then separated again to concatenate column-wise for further processing
def handleMergingCaseControlDataset(case_dataset:pd.DataFrame, control_dataset:pd.DataFrame)->pd.DataFrame:
    in_set_df=case_dataset.merge(control_dataset,how="inner", on="setid")
#     print(in_set_df.columns)
#     print(in_set_df.shape)
    case_dataset=in_set_df[["STUDYID_x","TEXT_REPROT_x","case_x","ethnicity_x","race_x","sex_x","age_index_x"]]
#     print(case_dataset.head(5))
    case_dataset.insert(1, "OLDCASE", "0")
#     print(case_dataset.columns)
    control_dataset=in_set_df[["STUDYID_y","OLDCASE","TEXT_REPROT_y","case_y","ethnicity_y","race_y","sex_y","age_index_y"]]
    case_dataset.rename(columns={"STUDYID_x":"STUDYID","TEXT_REPROT_x": "TEXT_REPROT","case_x":"case","ethnicity_x":"ethnicity", "race_x":"race", "sex_x":"sex", "age_index_x":"age_index"}, inplace=True)
#     print(case_dataset.columns)
    control_dataset.rename(columns={"STUDYID_y":"STUDYID","TEXT_REPROT_y": "TEXT_REPROT","case_y":"case","ethnicity_y":"ethnicity", "race_y":"race", "sex_y":"sex", "age_index_y":"age_index"}, inplace=True)
#     print(control_dataset.columns)
    # case_dataset.to_csv("./../generated_files/cases_set_id_0_to_365.csv")
    # control_dataset.to_csv("./../generated_files/controls_set_id_0_to_365.csv")
    case_dataset.drop_duplicates(inplace=True)
    control_dataset.drop_duplicates(inplace=True)
    print("cases_in_set: ",case_dataset.shape)
    print("controls_in_set: ", control_dataset.shape)
    all_dataset=pd.concat([case_dataset,control_dataset],axis=0,  ignore_index=True)
    print(all_dataset)
    return all_dataset


# In[8]:


# removed all the terms that are not in topic words dataset
def handleRemovingTerms(text:str, topic_set:pd.DataFrame, index:int, all:bool):
    #print(text)
    all_terms=str(text).split()
    
    #-----------------------------------------------------------------------------------
    # Extracting topic words from all clusters
    #-----------------------------------------------------------------------------------
    if all==True:
        all_topic_words= topic_set[["0","1","2"]].values.tolist()
        #print(all_topic_words)
        all_topic_words=[word for subset in all_topic_words for word in subset ]
        #print(all_topic_words)
    else:
        all_topic_words=list(topic_set[str(index)])
        
    all_allowed_tokens=[word for word in list(set(all_terms)) if word in list(set(all_topic_words))]
    #print(all_allowed_tokens)
    #print(type(all_allowed_tokens))
    text=(" ").join(all_allowed_tokens)
    
    return text


# In[9]:


def handleKeepingTermsWithContext(text:str, topic_set:pd.DataFrame, index:int, all:bool, window_size=1):
    
    
    #-----------------------------------------------------------------------------------
    # Extracting topic words from all clusters
    #-----------------------------------------------------------------------------------
    if all==True:
        all_topic_words= topic_set[["0","1","2"]].values.tolist()
        #print(all_topic_words)
        all_topic_words=[word for subset in all_topic_words for word in subset ]
        #print(all_topic_words)
    else:
        all_topic_words=list(topic_set[str(index)])
    
    tokenized_text=nltk.sent_tokenize(text)
    for k, sent in enumerate(tokenized_text):
        all_terms=list(str(sent).split())
        all_allowed_tokens=[]
        for i, word in enumerate(all_terms):
            # when topic_words are detected
            if word in list(set(all_topic_words)):
                #print(word)
                starting_point=i-window_size
                #print("starting_point: ",starting_point)   
                end_point=i+window_size 
                #print("end_point: ",end_point)
                
                #-------------------------------------------------------------------------------------------
                # H a n d l i n g    S t a r t-E n d    c o n s t r a i n s   of    t h e    d o c u m e n t
                #-------------------------------------------------------------------------------------------
                #taking all the terms within the window size to the left and to the right of the topic word 
                for j in range(starting_point, end_point+1):  # +1 was added because the ending point is exclusive in for loop
                    #print("topic_word = ", word)
                    if j<0:
                        #all_allowed_tokens.append("xxx")
                        continue

                    if j>=len(all_terms):
                        # all_allowed_tokens.append("xxx")
                        continue
                    all_allowed_tokens.append(all_terms[j])
                #print("extracted_words = ",all_allowed_tokens)
                
                #print(all_allowed_tokens)
        #print(type(all_allowed_tokens))
        tokenized_text[k]=(" ").join(all_allowed_tokens)
    TOTAL_WORDS.append(len(all_allowed_tokens))
    text=("*").join(tokenized_text)
    #print(text)
    return text


# In[10]:


def handleTask(dataset: pd.DataFrame, topic_set: pd.DataFrame,keep_context:bool, keep_demog:bool, all:bool, window_size:int, *args):
    deep_copied_dataset=dataset.copy(deep=True)
    print(deep_copied_dataset.columns)
    print("keep_context: {} \n keep_demogs: {} \n keep_all_clusters: {}".format(keep_context, keep_demog, all))
    #-------------------------------------------
    for i, text in (enumerate(dataset["text"])):
        if keep_context==False:
            deep_copied_dataset.loc[i, "text"]=handleRemovingTerms(text=text,
                                                                          topic_set=topic_set,
                                                                          index= args[0],
                                                                          all=all
                                                                         )
        elif keep_context==True:
            deep_copied_dataset.loc[i, "text"]=handleKeepingTermsWithContext(text=text,
                                                                                    topic_set=topic_set,
                                                                                    window_size= window_size, 
                                                                                    all=all,
                                                                                    index= args[0]
                                                                                   )
        if keep_demog==True:
            demog_string="a {ethnicity} {race} {sex} in his {age_division}".format(ethnicity=handleEthnicity(deep_copied_dataset.loc[i, "ethnicity"]),
                                                                                   race=handleRace(deep_copied_dataset.loc[i, "race"]),
                                                                                   sex=handleSex(deep_copied_dataset.loc[i, "sex"]),
                                                                                   age_division=handleAgeDivision(deep_copied_dataset.loc[i, "age_index"])
                                                                                  )
            #print(demog_string)
            deep_copied_dataset.loc[i, "text"]=demog_string+". "+deep_copied_dataset.loc[i, "text"]
        
    #print("avg words: ",sum(TOTAL_WORDS)/len(TOTAL_WORDS))
    return deep_copied_dataset


# In[ ]:





# In[11]:


def main(*args):
    #import datasets
    case_dataset=pd.read_csv("./../../nx_cleaned_cases_-1_to_-365_final_w_age.csv")
    control_dataset=pd.read_csv("./../../nx_cleaned_controls_-1_to_-365_final_w_age.csv")
    topic_set=pd.read_csv("./../../topic_words_-1_to_-365_w_age.csv")
    case_dataset=case_dataset.iloc[:,1:]
    control_dataset=control_dataset.iloc[:,1:]
    topic_set=topic_set.iloc[:50,1:]
    
    #----------------------------------------------------------------------------------------------------
    # Main Block
    #----------------------------------------------------------------------------------------------------
    all_dataset=handleMergingCaseControlDataset(case_dataset,control_dataset)
    cleaned_dataset=handleTask( all_dataset, topic_set, True, False, True, 5, args[0] )
    cleaned_dataset.head(5)
    #shuffling rows
    text_w_topic_dataset=cleaned_dataset.sample(frac=1).reset_index(drop=True)
    #text_w_topic_dataset.to_csv("./../generated_files/dataset_75_tw_w_context_-1_to_-365_w_age.csv")
    return text_w_topic_dataset


# In[12]:


#main(1)


# In[ ]:




