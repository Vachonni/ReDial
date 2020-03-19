#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:52:25 2020


File where we create the ReD_or_id (i.e ReDial Ordered id i.e. 0 to lenght)

Created from the original file movies_with_mentions.csv
    
Output is {ReD_id: ReD_or_id}


@author: nicholas
"""


import pandas as pd
import json



# Import movies_with_mentions.csv to Numpy Array
mv_arr = pd.read_csv('/Users/nicholas/ReDial/Data/Raw/movies_with_mentions.csv').to_numpy()


# Initialize 2 dicts
ReD_id_2_ReD_or_id = {}
ReD_or_id_2_ReD_id = {}


# For all movies, attribute the values
for i, (ReD_id, _, _) in enumerate(mv_arr):
    
    ReD_id_2_ReD_or_id[ReD_id] = i
    ReD_or_id_2_ReD_id[i] = ReD_id
    
    
# Saving dicts
with open('ReD_id_2_ReD_or_id.json', 'w') as fp:
    json.dump(ReD_id_2_ReD_or_id, fp)   
    
with open('ReD_or_id_2_ReD_id.json', 'w') as fp:
    json.dump(ReD_or_id_2_ReD_id, fp)   
