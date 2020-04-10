#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:42:22 2020


Creating DataByItems 
    
    (i.e., for each item (movie), which user ranked them, with which rating. In BERT format.)


@author: nicholas
"""


import json
import pandas as pd
from collections import defaultdict
import random



# Load KB of movies  
with open('/Users/nicholas/ReDial/Data/PreProcessed/KB_IMDB_movies.json', 'r') as fp:
    KB_IMDB_movies = json.load(fp)

# Load converter of ids
with open('/Users/nicholas/ReDial/Data/PreProcessed/ReD_or_id_2_ReD_id.json', 'r') as fp:
    ReD_or_id_2_ReD_id = json.load(fp)
        
  
    



def ItemRatings(ED_data):
    """
    From user's ratings on items (as they are in ED) to item's ratings from users.
    Meaning, we'll have a dict of items with their corresponding users and ratings

    Parameters
    ----------
    ED_data :  TYPE list of list
        FORMAT:[
        [
        user-> str, 
        inputs -> [(ReD_or_id, rating)]
        genres -> [str]
        targets -> [(ReD_or_id, rating)]
        ]
        ]
        DESCRIPTION: For ED, target is only one movie, the ones mentioned in next messageTYPE

    Returns
    -------
    item_ratings : TYPE dict of list of tuples
        FORMAT: {ReD_or_Id -> str:     
        [ 
        (user_id -> int, 
          ratings -> int)
        ]
        }
        DESCRIPTION: For each items in ED_data, list of users and the rating they 
                      give to the item (movie)
    """
    
    # Init items ratings    
    item_ratings = defaultdict(list)
    
    # For each user in ED_data
    for user_id, l_inputs, _, l_targets in ED_data:
        
        # Turn user into int (not key anymore)
        user_id = int(user_id)
        
        # Combine inputs and targets (we want both)
        l_inputs_targets = l_inputs + l_targets
        
        # For items and ratings in inputs and targets that were mentioned by user
        for ReD_or_id, rating in l_inputs_targets:
            
            # Turn item id into str (now key)
            ReD_or_id = str(ReD_or_id)
            
            # If the item not in dict yet, add the user and it's rating
            if ReD_or_id not in item_ratings:
                item_ratings[ReD_or_id].append((user_id, rating))
            # If item already in dict, check if user's rating pair already captured
            else:
                if (user_id, rating) not in item_ratings[ReD_or_id]:
                    item_ratings[ReD_or_id].append((user_id, rating))
                    

    return item_ratings





def MaxUser(sets_items_ratings):
    """

    Parameters
    ----------
    sets_items_ratings : TYPE dict
        DESCRIPTION: key = dataset identifier, value = item_ratings

    Returns
    -------
    max_user : TYPE int
        DESCRIPTION: Max quantity of user's who rated a single movie
    """

    
    # Init the max quantity of users who rated a single movie
    max_user = 0
    
    # For all datasets
    for item_ratings in sets_items_ratings.values():
        
        # For all items
        for l_ratings in item_ratings.values():
            qt_ratings = len(l_ratings)
            if qt_ratings > max_user: 
                max_user = qt_ratings
    
    return max_user
     



def ToBERTandSave(dataset, item_ratings, max_user):

    # 3 types of text input from the knowledge base we'll use
    l_KB_title = []
    l_KB_ti_g_ac = []
    l_KB_full = []
    
    # For all items 
    for item_id, ratings in item_ratings.items():
    
        # Fill the ratings appropriately
        # Initialize 
        filled_targets = [(-2, 0)] + ratings
        # Filling
        fill_size = max_user - len(ratings)
        filling = [(-1,0)] * fill_size
        filled_targets += filling
        
        # Get the NL Title
        KB_title = KB_IMDB_movies[str(ReD_or_id_2_ReD_id[item_id])]['title']
        # If empty, put a space (needed for BERT input)
        if KB_title == '': KB_title = ' '
        # Add to the list with item's id and the targets
        l_KB_title.append([int(item_id), KB_title, filled_targets])
        
        # ...same for the NL Title, Genres and Actors
        KB_ti_g_ac = KB_IMDB_movies[str(ReD_or_id_2_ReD_id[item_id])]['title'] + '. Genres: ' + \
                     str(KB_IMDB_movies[str(ReD_or_id_2_ReD_id[item_id])]['genres']) + '. Actors: ' + \
                     str(KB_IMDB_movies[str(ReD_or_id_2_ReD_id[item_id])]['actors'])
        if KB_ti_g_ac == '': KB_ti_g_ac = ' ' 
        l_KB_ti_g_ac.append([int(item_id), KB_ti_g_ac, filled_targets])         
         
        # ...same for all the KB of a movie in NL    
        KB_full =  KB_IMDB_movies[str(ReD_or_id_2_ReD_id[item_id])]
        if KB_full == '': KB_full = ' '
        l_KB_full.append([int(item_id), KB_full, filled_targets])  
        

    # Save those 3 files
    
    col = ['ConvID', 'text', 'ratings']
    
    df_KB_title = pd.DataFrame(l_KB_title, columns=col)
    df_KB_title.to_csv('/Users/nicholas/ReDial/Data/BERT/Items/Title/'+dataset+'.csv', index=False)
        
    df_KB_ti_g_ac = pd.DataFrame(l_KB_ti_g_ac, columns=col)
    df_KB_ti_g_ac.to_csv('/Users/nicholas/ReDial/Data/BERT/Items/TGA/'+dataset+'.csv', index=False) 
        
    df_KB_full = pd.DataFrame(l_KB_full, columns=col)
    df_KB_full.to_csv('/Users/nicholas/ReDial/Data/BERT/Items/FullKB/'+dataset+'.csv', index=False)
        
    
    
    
#%%


if __name__ == '__main__':
    
    
    # Paths to train, val and test datasets. 
    # Using ED because has unique ordered user_id
    # Using All to get a more general item (movie) representation
    paths = {'Train': '/Users/nicholas/ReDial/Data/ED/All/Train.json', \
              'Val': '/Users/nicholas/ReDial/Data/ED/All/Val.json', \
              'Test': '/Users/nicholas/ReDial/Data/ED/All/Test.json'}
      
    all_sets_item_ratings = {}    
    
    
    ###################    
    # REVERSE RATINGS #  
    ################### 
    
    
    # Concatenate 3 datasets in all_data
    all_data = []
    for dataset, datapath in paths.items():
        
        # Load data
        with open(datapath, 'r') as fp:
            data = json.load(fp)

        all_data += data
        
    # items (ReD_or_id) to list of user's (user_id) ratings
    all_item_ratings = ItemRatings(all_data)
        
        
    ##############################    
    # SPLIT - train, valid, test #  
    ##############################     
        
    indices = list(all_item_ratings.keys()) 
    random.shuffle(indices)
    
    indices_by_set = {'Train': indices[0:int(len(indices)*0.8)], \
                      'Val': indices[int(len(indices)*0.8):int(len(indices)*0.9)], \
                      'Test': indices[int(len(indices)*0.9):]}
    
    for this_set, these_indices in indices_by_set.items():  
        
        items_ratings_this_set = {}
        for indice in these_indices:
            items_ratings_this_set[str(indice)] = all_item_ratings[str(indice)]
            
        all_sets_item_ratings[this_set] = items_ratings_this_set
        
        
        
        
    ########################################    
    # TURN ITEM'S RATING TO BERT.csv FILES #
    ########################################    
        
    # Get the max quantity of user's who rated a single movie
    max_user = MaxUser(all_sets_item_ratings)
    
    # Turn all items_ratings to BERT.csv files and save them
    for dataset, item_ratings in all_sets_item_ratings.items():
        
        ToBERTandSave(dataset, item_ratings, max_user)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
