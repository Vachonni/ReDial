#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:24:21 2020


Managing genres in ReDial:
    
    1- From movies KB, get the dict of unique genres to the ReD_or_id related. 
    Save list of keys in Settings.
    
    2- Make a dict of combined genres by intersection to ReD_or_id of format:
        {['genres']: [genres_idx, [ReD_or_id]]}
        The combined genres are identifed according to the ones in ReDial data.


@author: nicholas
"""



import sys
from pathlib import Path
import torch
import json
from collections import defaultdict

# Adding ReDial's folder to the sys.path for imports
path = Path(sys.executable)
# If using cpu, assume at root of user's machine
if not torch.cuda.is_available():
    path_to_ReDial = str(path.home()) + '/ReDial'
# If not, assume Compute Canada, hence in scratch
else:
    path_to_ReDial = str(path.home()) + '/scratch/ReDial'
if path_to_ReDial not in sys.path:
    sys.path.insert(0, path_to_ReDial)


from Settings import ReD_id_2_ReD_or_id




# Load KB

with open('/Users/nicholas/ReDial/Data/PreProcessed/KB_IMDB_movies.json', 'r') as f:
    KB = json.load(f)


#%%



#######################
# Make unique genres to ReD_or_id
#######################
    
    
genres_unique_2_ReD_or_id = defaultdict(list)

for k, v in KB.items():

    l_genres = v['genres'] 
    
    # For all unique genres of that movie k, add movie as ReD_or_id
    for g in l_genres:
        genres_unique_2_ReD_or_id[g.lower()].append(ReD_id_2_ReD_or_id[k])
  
    
# Save it  
with open('../PreProcessed/genres_unique_2_ReD_or_id.json', 'w') as f:
    json.dump(genres_unique_2_ReD_or_id, f)



# Add unique genres existance list to Settings (hand copied)
l_unique_genres = [k for k in genres_unique_2_ReD_or_id.keys()]





#%%



#######################
# Make a dict of combined genres by intersection to ReD_or_id
#######################


# Load ReDial Data in ED format
kind_data = ['train', 'valid', 'test']

# Init and treat the no genres mentioned case now
genres_inter_2_ReD_or_id = {'[]': []}


for kind in kind_data:
    
    with open('/Users/nicholas/ReDial/Data/EncoderDecoder/ED_next_'+kind+'.json', 'r') as f:
        ED_data = json.load(f)

    # Get the list of genres combinations for every data point
    for _, _, genres, _ in ED_data:
        
        # If this genres combination doesn't exist yet
        if str(genres) not in genres_inter_2_ReD_or_id:
            
            # Get movies for every genres
            movies_by_genres = []
            for g in genres:
                # Add movies of that genres as a set (for intersection after)
                movies_by_genres.append(set(genres_unique_2_ReD_or_id[g]))
            
            # Get intersection 
            movies_inter = list(movies_by_genres[0].intersection(*movies_by_genres))
    
            # If there are such movies, add this genres combination and movies related 
            if movies_inter != []:
                genres_inter_2_ReD_or_id[str(genres)] = movies_inter


               
# Save it 
with open('../PreProcessed/genres_inter_2_ReD_or_id.json', 'w') as f:
    json.dump(genres_inter_2_ReD_or_id, f)





#%%

# Add idx for case g_type = genres and Dataset format of ReDial_E19
genres_inter_IDX_2_ReD_or_id = {k: [i, v] for i,(k,v) in \
                                enumerate(genres_inter_2_ReD_or_id.items())}

#%%
    
# Insure index 1 is no genres, i.e []. No, mystery has 1
genres_inter_IDX_2_ReD_or_id["[]"][0] = 1
genres_inter_IDX_2_ReD_or_id["['mystery']"][0] = 0

#%%

# Save
with open('../PreProcessed/genres_inter_IDX_2_ReD_or_id.json', 'w') as f:
    json.dump(genres_inter_IDX_2_ReD_or_id, f)









