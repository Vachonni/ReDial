#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:56:41 2020


Pre-processing of ReDial to get "cold items". Also, split both valid and test sets 
of type "Next" in two, depending on the target movie being "seen" in train set or not
(identified as "new").


Cold Items are items never seen in train set.

"Val_seen" and "Test_seen" are sets where target movies have been mentioned in train set.

"Val_new" and "Test_new" are sets where target movies have not been mentioned in train set.



@author: nicholas
"""


import json
from collections import defaultdict
import pandas as pd
from ast import literal_eval




#####
# 1 # - Get all movies mentioned in train set (input and target movies)
#####


# Init. All movies in train (input or target) will have a key with value 1.
seen_in_train = defaultdict(int)


# Load train
with open('/Users/nicholas/ReDial/Data/ED/All/Train.json', 'r') as fp:
    train_set = json.load(fp)
    
# Go through all data points
for _, l_inputs, _, l_targets in train_set:
    # Go through all inputs
    for i, _ in l_inputs:
        # Set value to 1 movies found
        seen_in_train[i] = 1
    # Go through all targets
    for i, _ in l_targets:
        # Set value to 1 movies found
        seen_in_train[i] = 1
        
        

        
        

#####
# 2 # - Create Valid and Test set splits (seen and new)
#####



kind_of_set = {'Val', 'Test'}

for kos in kind_of_set:
    
    
    # ED CASE
    
    # Init new sets
    set_seen = []
    set_new = []
    
    # Load set 
    with open('/Users/nicholas/ReDial/Data/ED/Next/' + kos +'.json') as fp:
        dataset = json.load(fp)
        
    # Split data according to presence in train
    for datapoint in dataset:
        target, rating = datapoint[3][0]   # target is idx 3 and has one tuple in the list
        if target in seen_in_train:
            set_seen.append(datapoint)
        else:
            set_new.append(datapoint)
    
    # Save sets splited
    with open('/Users/nicholas/ReDial/Data/ED/Next/' + kos +'_seen.json', 'w') as fp:
        json.dump(set_seen, fp)
    with open('/Users/nicholas/ReDial/Data/ED/Next/' + kos +'_new.json', 'w') as fp:
        json.dump(set_new, fp)        
  

    
    # BERT CASE
    
    # Init new sets
    raw_seen = []
    raw_new = []
    nl_seen = []
    nl_new = []
    nlg_seen = []
    nlg_new = []
    
    # Load 3 types of sets
    raw = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextRaw/' + kos +'.csv').values    
    nl = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNL/' + kos +'.csv').values 
    nlg = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNLGenres/' + kos +'.csv').values 
    
    for i, datapoint in enumerate(raw):
        target, rating = literal_eval(datapoint[2])[1]   # target is a str at idx 2 and is 2nd tuple in the list
        if target in seen_in_train:
            raw_seen.append(raw[i])
            nl_seen.append(nl[i])
            nlg_seen.append(nlg[i])
        else:
            raw_new.append(raw[i])
            nl_new.append(nl[i])
            nlg_new.append(nlg[i])

    # Save sets splited
    col = ['ConvID', 'text', 'ratings']
    
    df_raw_seen = pd.DataFrame(raw_seen, columns=col)
    df_raw_seen.to_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextRaw/' + kos +'_seen.csv', index=False)
    
    df_raw_new = pd.DataFrame(raw_new, columns=col)
    df_raw_new.to_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextRaw/' + kos +'_new.csv', index=False)    
    
    df_nl_seen = pd.DataFrame(nl_seen, columns=col)
    df_nl_seen.to_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNL/' + kos +'_seen.csv', index=False)
    
    df_nl_new = pd.DataFrame(nl_new, columns=col)
    df_nl_new.to_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNL/' + kos +'_new.csv', index=False)
    
    df_nlg_seen = pd.DataFrame(nlg_seen, columns=col)
    df_nlg_seen.to_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNLGenres/' + kos +'_seen.csv', index=False)
    
    df_nlg_new = pd.DataFrame(nlg_new, columns=col)
    df_nlg_new.to_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNLGenres/' + kos +'_new.csv', index=False)






#%%

import pandas as pd

ts = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNL/Test_seen.csv').values    
tn = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNL/Test_new.csv').values
vs = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNL/Val_seen.csv').values
vn = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/Next/TextNL/Val_new.csv').values

#%%

with open('/Users/nicholas/ReDial/Data/ED/Next/Test_new.json', 'r') as fp:
    edtn = json.load(fp)        
with open('/Users/nicholas/ReDial/Data/ED/Next/Test_seen.json', 'r') as fp:
    edts = json.load(fp)   
with open('/Users/nicholas/ReDial/Data/ED/Next/Val_new.json', 'r') as fp:
    edvn = json.load(fp)   
with open('/Users/nicholas/ReDial/Data/ED/Next/Val_seen.json', 'r') as fp:
    edvs = json.load(fp)         















