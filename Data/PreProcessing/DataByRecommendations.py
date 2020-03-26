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


from Objects.Conversation import Conversation
from ED.Arguments import args





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
                ConID: int,
                text: str, 
                ratings: "[(ReD_or_id, ratings)]" -> starts with (-2,qt_mentioned) fills (-1,0)
                }
        DESCRIPTION: For BERT, target is only one movie, the ones mentioned in next message 
    
    BERT_all : TYPE: **ONE** defaultdict(list) where each keys has values for all dataset
        FORMAT:{
                ConID: [int],
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
                    'Train': args.path_to_ReDial+'/Data/Split/train_data.jsonl',
                    'Val': args.path_to_ReDial+'/Data/Split/val_data.jsonl',
                    'Test': args.path_to_ReDial+'/Data/Split/test_data.jsonl'
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
            
        # Save ED_next pre-processed data
        save_path = args.path_to_ReDial+'/Data/ED/Next/'+kind+'.json'
        with open(save_path, 'w') as fp:
            json.dump(ED_next, fp)   

        # Save ED_all pre-processed data
        save_path = args.path_to_ReDial+'/Data/ED/All/'+kind+'.json'
        with open(save_path, 'w') as fp:
            json.dump(ED_all, fp)   

        # Save BERT_next pre-processed data
        df = pd.DataFrame(BERT_next)
        df.to_csv(args.path_to_ReDial+'/Data/BERT/Next/'+kind+'.csv', index=False)

        # Save BERT_all pre-processed data
        df = pd.DataFrame(BERT_all)
        df.to_csv(args.path_to_ReDial+'/Data/BERT/All/'+kind+'.csv', index=False)














