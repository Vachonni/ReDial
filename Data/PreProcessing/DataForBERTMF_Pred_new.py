#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:02:56 2019


====--->>> Same as DataForBERTMF_Pred.py, but for new items (unseen in Train) <<<---===

    Creating data for BERT MF (Matrix Factorization)   ---> Prediction
    
        We do Test data since very long process. Best model identified with valid loss.
        
        For every user datapoint, we create a .csv file in fast_bert format, where 
        text is a combination of user's text and text of every item.
    



@author: nicholas
"""


########  IMPORTS  ########


import json
import numpy as np
import pandas as pd
from pathlib import Path






########################
#                      # 
#         Main         #
#                      # 
########################  
    

def main():
    


    # Making path
    
    dataPATH = '/Users/nicholas/ReDial/Data/CF2/'
    



    # Loading data

    print('\n******* Loading CF2 TEST samples from *******', dataPATH + 'Test_new.json')
    with open(dataPATH + 'Test_new.json', 'r') as fp:
        test_data = json.load(fp)
    test_data = np.array(test_data)
    

      
      
    # Loading KB

    print('\n******* Loading KB_users')
    with open('/Users/nicholas/ReDial/Data/PreProcessed/KB_users.json', 'r') as fp:
        KB_users = json.load(fp)
    
    print('******* Loading KB_items')
    with open('/Users/nicholas/ReDial/Data/PreProcessed/KB_IMDB_movies.json', 'r') as fp:
        KB_items = json.load(fp)
    qt_items = len(KB_items)
    print('******* Loading Converter ReD_or_id_2_ReD_id  \n')
    with open('/Users/nicholas/ReDial/Data/PreProcessed/ReD_or_id_2_ReD_id.json', 'r') as fp:
        ReD_or_id_2_ReD_id = json.load(fp)



    # Getting the text for every items (once, use for every user)
    l_item_texts = []
    for i in range(qt_items):
        
        # Get ReD_id string related to i
        ReD_id_str = str(ReD_or_id_2_ReD_id[str(i)])
        item_text = KB_items[ReD_id_str]['title'] + '. Genres: ' + \
                    str(KB_items[ReD_id_str]['genres']) + '. Actors: ' + \
                    str(KB_items[ReD_id_str]['actors'][:3])     
        l_item_texts.append(item_text)
    

    count_users = 0
    # Treat each datapoint (~user)
    for _, _, qt_movies_mentioned, user_id, ReD_or_id, rating in test_data:
        
        # Not evaluating rank of a rating at 0
        if rating == 0: continue
        
        # Get user's text
        user_text = KB_users[str(user_id)]['text_nlg']
        l_user_text = [user_text] * qt_items
        
        # Create list of SEP tokens
        l_SEP = [' [SEP]'] * qt_items
        
        # Create all texts
        l_text = [t[0]+t[1]+t[2] for t in zip(l_item_texts, l_SEP, l_user_text)]

        # Repeat the rating list
        l_rating = [[(-2, qt_movies_mentioned), (ReD_or_id, rating)]] * qt_items

        # Repeat user_id
        l_user_id = [user_id] * qt_items
        
        # Zip it all together. 
        data_BERTMF_this_user = [[t[0],t[1],t[2]] for t in zip(l_user_id, l_text, l_rating)]
    
    
        # Save in .csv for fast_bert
        col = ['ConvID', 'text', 'ratings']
        df = pd.DataFrame(data_BERTMF_this_user, columns=col)
        savePATH = Path('/Users/nicholas/ReDial/Data/BERTMF/Test_new/User' + \
                                                                str(count_users))
        savePATH.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(savePATH) + '/Test.csv', index=False)
        
        count_users += 1

        
        

if __name__ == '__main__':
    main()
    














































    
    







































