#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:46:04 2020


Creating the input_id and KB_input_id. Will be used for CF2.

    (i.e. unique ids for datapoints. 
     Not users since users are by conversation.
     No messages since we take all messages until a recommendation)

 

***     Should have been done while during DataByRecommendation :((     ***



@author: nicholas
"""

import json
import pandas as pd
import numpy as np


# For all 4 datasets, load train, val, test. Concatenate train, val, test.

# ED
with open('/Users/nicholas/ReDial/Data/ED/All/Train.json', 'r') as fp:
    ED_train = json.load(fp)
with open('/Users/nicholas/ReDial/Data/ED/All/Val.json', 'r') as fp:
    ED_val = json.load(fp)
with open('/Users/nicholas/ReDial/Data/ED/All/Test.json', 'r') as fp:
    ED_test = json.load(fp)

ED = ED_train + ED_val + ED_test

# BERT
# Raw
BERT_raw_train = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextRaw/Train.csv').values
BERT_raw_val = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextRaw/Val.csv').values
BERT_raw_test = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextRaw/Test.csv').values
BERT_raw = np.concatenate((BERT_raw_train, BERT_raw_val, BERT_raw_test), axis=0)
# Raw
BERT_nl_train = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextNL/Train.csv').values
BERT_nl_val = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextNL/Val.csv').values
BERT_nl_test = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextNL/Test.csv').values
BERT_nl = np.concatenate((BERT_nl_train, BERT_nl_val, BERT_nl_test), axis=0)
# Raw
BERT_nlg_train = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextNLGenres/Train.csv').values
BERT_nlg_val = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextNLGenres/Val.csv').values
BERT_nlg_test = pd.read_csv('/Users/nicholas/ReDial/Data/BERT/All/TextNLGenres/Test.csv').values
BERT_nlg = np.concatenate((BERT_nlg_train, BERT_nlg_val, BERT_nlg_test), axis=0)


#%%

# Go through all data 

KB_input_id = {}

for i in range(len(ED)):
    
    # Insure same data point
    assert int(ED[i][0]) == BERT_raw[i][0] == BERT_nl[i][0] == BERT_nlg[i][0], 'FUCK'
    
    # Update KB_input_id
    KB_input_id[i] = {'conv_id': int(ED[i][0]),
                      'ED': ED[i],
                      'TextRaw': BERT_raw[i][1],
                      'TextNL': BERT_nl[i][1],
                      'TextNLGenres': BERT_nlg[i][1]}     


#%%

with open('/Users/nicholas/ReDial/Data/PreProcessed/KB_input_id.json', 'w') as fp:
    json.dump(KB_input_id, fp)
    
    
