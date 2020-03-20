#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:38:14 2018


All constants for ReDial


@author: nicholas
"""

import json
import torch

from Arguments import args 



# Number of movies in ReDial only
nb_movies_ReDial = 6924  # 48272 



# List of genres in IMDB for ReDial movies
genres = ['thriller', 'comedy', 'family', 'fantasy', 'sport', 'drama', 'music', 
          'mystery', 'action', 'sci-fi', 'animation', 'adventure', 'musical', 
          'romance', 'horror', 'documentary', 'crime', 'history', 'western', 
          'news', 'war', 'biography', 'film-noir', 'short', 'reality-tv']



# Load dictionary of conversions between ReD_id and ReD_or_id
with open(args.path_to_ReDial + '/Data/PreProcessed/ReD_id_2_ReD_or_id.json', 'r') as fp:
    ReD_id_2_ReD_or_id = json.load(fp)



# Get the popularity vector (Numpy Array where idx are ReD_or_id and values are number of mentions)
popularity = torch.load(args.path_to_ReDial + '/Data/PreProcessed/popularity.pth').float()