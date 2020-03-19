#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:19:25 2020


File where we create the popularity vector (i.e the number of times a movie is mentioned in ReDial)

Created from the original file movies_with_mentions.csv
    
Output is a torch tensor where idx are ReD_or_id and values are number of mentions


@author: nicholas
"""


import pandas as pd
import torch



# Import movies_with_mentions.csv to Numpy Array
mv_arr = pd.read_csv('/Users/nicholas/ReDial/Data/Raw/movies_with_mentions.csv').to_numpy()


# Get the appropriate column and turn into float 
popularity = mv_arr[:,2].astype(float)


# As torch float tensor for futher purposes
popularity = torch.from_numpy(popularity)
    
# Saving dicts
torch.save(popularity, 'popularity.pth')