#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:02:56 2019


Creating data for BERT MF (Matrix Factorization)   ---> Training

    Starting with CF2 data, augmenting it with items at 0 and putting it in
    in the format required by fast_bert
    
    
    
    
        *** To change quantity of random items at 0, 
            do it in GetRandomItemsAt0 func.
            Should also change name in saving portion  ;)      ***




@author: nicholas
"""


########  IMPORTS  ########

import time
import json
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing






########################
#                      # 
#     Augment data     #
#                      # 
########################        



def GetRandomItemsAt0(user_row, qt_random_ratings=20):
    """

    Parameters
    ----------
    user_row : Numpy Array (vector)
        All inforations about a user.
            (eg: data_idx,ConvID,qt_movies_mentioned,user_chrono_id,movie_ReDOrId,rating)
        2nd to last position contains str-list of items that have real ids 
        Last position contains str-list of real ratings


    Returns
    -------
    Numpy Array (len(user_row[-1]) + qt  x  len(user_row))
        Each line has same colums values, except last 2 colums, where:
            2nd to last is real ids or new item ids
            Last position contains real ratings or 0

    """
    
    all_movies_rated, _, _, _, ReD_or_id, rating = user_row
    
    # Get real items_ids and associated ratings as numpy arrays (vectors)
    real_items = np.array([ReD_or_id])
    real_values = np.array([rating])
    
    # Get range of movies to choose from 
    range_size = 6924
        
    # Get random items ids, different from all_movies_rated
    random_ids = np.random.choice(np.setdiff1d(range(range_size), \
                                               np.array(all_movies_rated)), \
                                  qt_random_ratings)
    
    # Concat random ids with real_ones
    items = np.concatenate((real_items, random_ids))
    values = np.concatenate((real_values, np.zeros(qt_random_ratings)))
    
    # Expand the user_row to a matrix that will be returned
    user_mat = np.tile(user_row, (len(real_items)+qt_random_ratings, 1))
    
    # Replace 2 last columns 
    user_mat[:,-2] = items
    user_mat[:,-1] = values
    
    return user_mat    






########################
#                      # 
#         Main         #
#                      # 
########################  
    

def main():
    


    # Making paths
    
    dataPATH = '/Users/nicholas/ReDial/Data/CF2/'
    savePATH = '/Users/nicholas/ReDial/Data/BERTMF/'
    


    # Loading data

    print('\n******* Loading CF2 TRAIN samples from *******', dataPATH + 'Train.json')
    with open(dataPATH + 'Train.json', 'r') as fp:
        train_data = json.load(fp)
    train_data = np.array(train_data)
    
    print('******* Loading CF2 VALID samples from *******', dataPATH + 'Val.json')
    with open(dataPATH + 'Val.json', 'r') as fp:
        valid_data = json.load(fp)
    valid_data = np.array(valid_data)
    
    
      
    # Loading KB

    print('\n******* Loading KB_users')
    with open('/Users/nicholas/ReDial/Data/PreProcessed/KB_users.json', 'r') as fp:
        KB_users = json.load(fp)
    
    print('******* Loading KB_items')
    with open('/Users/nicholas/ReDial/Data/PreProcessed/KB_IMDB_movies.json', 'r') as fp:
        KB_items = json.load(fp)

    print('******* Loading Converter ReD_or_id_2_ReD_id  \n')
    with open('/Users/nicholas/ReDial/Data/PreProcessed/ReD_or_id_2_ReD_id.json', 'r') as fp:
        ReD_or_id_2_ReD_id = json.load(fp)



    # Augment train data    
    
    start_time = time.time()

    # Augment train_data with random ratings at 0
    print('before concurent futures')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        train_data_generator = executor.map(GetRandomItemsAt0, \
                                            train_data, chunksize=100)
    print('out of concurent furures')
    train_data_augmented = np.vstack(list(train_data_generator))
       
    augment_time = time.time()    
        
    print(f'Augmenting train data \
          took {augment_time - start_time} seconds with {multiprocessing.cpu_count()} CPUs.')
        
       
        
    datasets = {'Train20': train_data_augmented}   #,
              #  'Val': valid_data}        
    
    # Treat both dataset     
    for data_type, dataset in datasets.items():
         
        data_BERT_format = []
        
        
        # Treat each datapoint 
        for _, _, qt_movies_mentioned, user_id, ReD_or_id, rating in dataset:
            
            # Get the text
            ReD_id_str = str(ReD_or_id_2_ReD_id[str(ReD_or_id)])
            item_text = KB_items[ReD_id_str]['title'] + '. Genres: ' + \
                        str(KB_items[ReD_id_str]['genres']) + '. Actors: ' + \
                        str(KB_items[ReD_id_str]['actors'][:3])
            
            user_text = KB_users[str(user_id)]['text_nlg']
            
            text = item_text + ' [SEP] ' + user_text
    
            # Get the rating lsit
            l_rating = [(-2, qt_movies_mentioned), (ReD_or_id, rating)]
    
            # Append this data to the others
            data_BERT_format.append([(user_id, ReD_or_id),
                                     text, l_rating])

        
        # Save in .csv for fast_bert
        col = ['ConvID', 'text', 'ratings']
        df = pd.DataFrame(data_BERT_format, columns=col)
        df.to_csv(savePATH + data_type + '.csv', index=False)
        
        
    
        
        
        
        

if __name__ == '__main__':
    main()














































    
    







































