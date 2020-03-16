#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:19:35 2020


Class Message
    - Access to elements of a message dict in ReDial
        
    - Methods:
            - 


@author: nicholas
"""


import sys
from pathlib import Path 
import torch
import re


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


import Settings





re_filmId = re.compile('@[0-9]{5,6}')


class Message:
    
    def __init__(self, message_dict, seeker_id):
        """
        Initializing a Message from a ReDial message dict 
        (which are in a list of messages inb a Conversation)

        Parameters
        ----------
        message_dict : TYPE: dict.
                       FORMAT: {"messageId": int, "text": str, 
                                "timeOffset": int, "senderWorkerId": int}
                       DESCRIPTION: A message dict from ReDial

        Returns
        -------
        None.
        """
        
        self.speaker_id = message_dict['senderWorkerId']
        self.role = 'S::' if seeker_id == self.speaker_id else 'R::'
        self.text = message_dict['text']
        
        
        
    def GetGenres(self):
        """
        Takes a Message object and returns genres mentioned in the text of that Message

        Returns
        -------
        l_genres : TYPE: list of genres
                   FORMAT: [str]
                   DESCRIPTION: List of genres in the text of that Message
        """
        
        l_genres = []
        
        # Spit str by words after lowering case
        l_words = self.text.lower().split(' ')
        
        for w in l_words:
            if w in Settings.genres:
                l_genres.append(w)
        
        return l_genres
    
    
    
    
    def GetMovies(self):
        """
        Takes a Message object and returns movies ReD_id mentioned in the text of that Message

        Returns
        -------
        l_movies : TYPE: list of movies by ReD_id
                   FORMAT: [int]
                   DESCRIPTION: List of movies ReD_id in the text of that Message
        """
        
        l_movies = []
        # Use 'regular expressions' (re) to extract movie mentions
        l_movies = re_filmId.findall(self.text)
        
        # Remmove '@'at begining and turn str into int
        l_movies = [m[1:] for m in l_movies]
    
        return l_movies
    
        
        
        
        
        
        