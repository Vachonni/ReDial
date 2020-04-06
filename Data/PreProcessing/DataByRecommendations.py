#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:58:10 2020


Creating data for ED and BERT_Reco models



@author: nicholas
"""


import json
from collections import defaultdict
import pandas as pd
from pathlib import Path


from Objects.Conversation import Conversation





def ReDialDataToEDData(ReDial_data):
    """
    Takes a Redial dataset and 
    returns **FOUR** sets of data ready for ML:
            ED_next: For ED, target is only one movie, the ones mentioned in next message
            ED_all: For ED, targets are all the movies to be mentioned in the rest of conversation
            BERT_next: For BERT, target is only one movie, the ones mentioned in next message
            BERT_all: For BERT, targets are all the movies to be mentioned in the rest of conversation
  
    Parameters
    ----------
    ReDial_data : TYPE dict
        DESCRIPTION: Raw data from ReDial.

    Returns
    -------
    ED_next : TYPE list of list
        FORMAT:[
        [
        Conv_id -> str, 
        inputs -> [(ReD_or_id, rating)]
        genres -> [str]
        targets -> [(ReD_or_id, rating)]
        ]
        ]
        DESCRIPTION: For ED, target is only one movie, the ones mentioned in next message
    
    ED_all : TYPE list of list
        FORMAT:[
        [
        Conv_id -> str, 
        inputs -> [(ReD_or_id, rating)]
        genres -> [str]
        targets -> [(ReD_or_id, rating)]
        ]
        ]
        DESCRIPTION: For ED, targets are all the movies to be mentioned in the rest of conversation
    
    BERT_next : TYPE **ONE** defaultdict(list) where each keys has values for all dataset
        FORMAT:{
                ConvID: int,
                text: str, 
                ratings: "[(ReD_or_id, ratings)]" -> starts with (-2,qt_mentioned) fills (-1,0)
                }
        DESCRIPTION: For BERT, target is only one movie, the ones mentioned in next message 
    
    BERT_all : TYPE: **ONE** defaultdict(list) where each keys has values for all dataset
        FORMAT:{
                ConvID: [int],
                text: [str], 
                ratings: "[[(ReD_or_id, ratings)]]" -> starts with (-2,qt_mentioned) fills (-1,0)
                }
        DESCRIPTION: For BERT, targets are all the movies to be mentioned in the rest of conversation  
    """
    
    ED_next = []
    ED_all = []
    BERT_next = defaultdict(list)
    BERT_all = defaultdict(list)
    
    
    # For raw conversations in a set
    for conv_raw in ReDial_data:
        
        # Make it a convesations obj
        conv = Conversation(conv_raw)
        
        # Get the data in ED format
        ED_n, ED_a, BERT_n, BERT_a = conv.ConversationToDataByRecommendations()
        
        # Add data for this conv to the other conversations
        ED_next += ED_n
        ED_all += ED_a
        for k in BERT_n.keys():
            BERT_next[k] += BERT_n[k]
        for k in BERT_a.keys():
            BERT_all[k] += BERT_a[k]    

    return ED_next, ED_all, BERT_next, BERT_all





        
if __name__ == '__main__':
    
    # tTrain, valid and test data paths
    paths_to_data = {
                    'Train': '/Users/nicholas/ReDial/Data/Split/train_data.jsonl',
                    'Val': '/Users/nicholas/ReDial/Data/Split/val_data.jsonl',
                    'Test': '/Users/nicholas/ReDial/Data/Split/test_data.jsonl'
                    }


    # FIND MAX NUMBER OF RATED MOVIES IN ALL CONVERSATION
    
    ReDial_data = []
    
    # Load all ReDial's data (train, valid and test) in one variable
    for kind, path in paths_to_data.items():
            
        with open(path, "r") as fp:
            for line in fp:
                ReDial_data.append(json.loads(line))
            
    # Go through all conversations to find the max quanty of rated movies in a conv
    for conv_raw in ReDial_data:
        
        # Make it a convesations obj
        conv = Conversation(conv_raw)
        
        # If there was a form completed (hence ratings) 
        conv.movies_and_ratings = conv.GetMoviesAndRatings()
        if conv.movies_and_ratings != None:
            # If this conv has more ratings than the max number of ratings for all conv 
            qt_rated_this_conv = len(conv.movies_and_ratings)
            if qt_rated_this_conv > Conversation.max_qt_ratings: 
                Conversation.max_qt_ratings = qt_rated_this_conv

   

    # TREAT ALL CONVERSATIONS
    
    for kind, path in paths_to_data.items():
        
        # Load raw data
        ReDial_data = []
        with open(path, "r") as fp:
            for line in fp:
                ReDial_data.append(json.loads(line))    
        
       
        # Get processed data
        ED_next, ED_all, BERT_next, BERT_all = ReDialDataToEDData(ReDial_data)
            
        
        # Save ED data
        
        # ED_next 
        save_path = Path('/Users/nicholas/ReDial/Data/ED/Next/')
        save_path.mkdir(parents=True, exist_ok=True)
        with open(Path(save_path, kind+'.json'), 'w') as fp:
            json.dump(ED_next, fp)   
        # ED_all 
        save_path = Path('/Users/nicholas/ReDial/Data/ED/All/')
        save_path.mkdir(parents=True, exist_ok=True)
        with open(Path(save_path, kind+'.json'), 'w') as fp:
            json.dump(ED_all, fp)   

        
        # Save BERT data
                
        # Link between key and file name
        key_file = {'text_raw': 'TextRaw/', 'text_nl': 'TextNL/', \
                    'text_nl_genres': 'TextNLGenres/'}        
        
        # Split each BERT results in 3 dicts, for 3 types of data, and save
        for k, file in key_file.items():
            
            # BERT_next split
            sub_BERT_next = {'ConvID': BERT_next['ConvID'], \
                             'text': BERT_next[k], \
                             'ratings': BERT_next['ratings']   }
            # Save BERT_next 
            df = pd.DataFrame(sub_BERT_next)
            path = Path('/Users/nicholas/ReDial/Data/BERT/Next/', file)
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(path, kind+'.csv'), index=False)


            # BERT_all split
            sub_BERT_all = {'ConvID': BERT_all['ConvID'], \
                             'text': BERT_all[k], \
                             'ratings': BERT_all['ratings']   }
            # Save BERT_all 
            df = pd.DataFrame(sub_BERT_all)
            path = Path('/Users/nicholas/ReDial/Data/BERT/All/', file)
            path.mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(path, kind+'.csv'), index=False)














