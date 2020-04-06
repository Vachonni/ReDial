#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 10:48:29 2020


Creating the KB for movies in ReDial


@author: nicholas
"""

import pandas as pd
import re
# from SPARQLWrapper import SPARQLWrapper, JSON
import imdb 
from tqdm import tqdm
import json



# Initialize KB with titles and dates we have in movies_with_mentions.csv
mv_arr = pd.read_csv('../Raw/movies_with_mentions.csv').to_numpy()



########################
#                      # 
#         INIT         #
#                      # 
########################

# The KB will be a dict {ReD_id: {}}
KB = {}


# Titles include dates. Need to seperate them. Using re (regualr expressions) to identify date
re_date = re.compile(' *\([0-9]{4}\)| *$')      # Date (....), spaces before + after
                                                # for spaces, could have used .strip() below


for ReD_id, title_and_date, _ in mv_arr:
    
    # Get data
    date_obj = re_date.search(title_and_date)              
    date = date_obj.group().strip()                        # get date fom obj and remove space
    if date == '':
        date = None
    else:
        date = int(date[1:5])
    
    # Get title
    title = re_date.sub("", title_and_date).strip()       # Remove date from title

    # Add to KB
    KB[ReD_id] = {'title': title, 'date': date}




#%%


def get_IMDB_movie_object(title, date):
    """
    Performs Google search to find self movieName IMDB id
    (NOTE: IMDB search doesn't work with date, eg: A star id born (1954))
    Returns IMDB movie object, 0 if not.

    Attributes: 'cast', 'genres', 'year',...
    Examples here: https://www.programcreek.com/python/example/65711/imdb.IMDb
    
    TODO: Check if right date (like what I tried in method below, but good)
    TODO: Implement CHECK IF Google doesn't get IMDB first
    """
    
    # OLD IMDB search
    ia = imdb.IMDb()
    s_result = ia.search_movie(title)
    
    if s_result == []:
        IMDB_movie_obj = None                     # If no results.
    
    else:
        for result in s_result:
            if 'movie' in result.get('kind'):
                
                # If IMDB has no date, no more validation possible
                if result.get('year') == None:
                    Movie_ID = result.movieID
                    IMDB_movie_obj = ia.get_movie(Movie_ID)
                    break 
                # If we have no date, no more validation possible
                elif date == None:
                    Movie_ID = result.movieID
                    IMDB_movie_obj = ia.get_movie(Movie_ID)
                    break
                # If both dates, check if IMDB date within a year of our date
                elif date - 1 <= result.get('year') <= date + 1:
                    Movie_ID = result.movieID
                    IMDB_movie_obj = ia.get_movie(Movie_ID)
                    break
                # It's a movie with a date, but not the right one.
                else:
                    IMDB_movie_obj = None 
            
            # If not a movie kind.
            else:
                IMDB_movie_obj = None   
                
    return IMDB_movie_obj




#%%
    
count = 0 

for ReD_id, values in tqdm(KB.items()):

    IMDB_obj = get_IMDB_movie_object(values['title'], values['date'])   
    
    # If we found an IMDB_ojb (else, just continue since KB already has info it can)
    if IMDB_obj != None:
        
        KB[ReD_id]['IMDB_ID'] = int(IMDB_obj.getID())
        KB[ReD_id]['genres'] = IMDB_obj.get('genres', [])
        KB[ReD_id]['actors'] = [IMDB_obj['cast'][i]['name'] for i in range(min(10, len(IMDB_obj.get('cast',[]))))]
        KB[ReD_id]['directors'] = [IMDB_obj['directors'][i]['name'] for i in range(min(2, len(IMDB_obj.get('directors', []))))]
        KB[ReD_id]['plot'] = IMDB_obj.get('plot outline', '')
        
        count += 1
#        if count >=5: break
        
        # Save every 100 searches
        if count % 100 == 0:
            with open('../PreProcessed/KB_IMDB_movies.json', 'w') as fp:
                json.dump(KB, fp, indent=4,)   



#%%
    
# Save
    
with open('../PreProcessed/KB_IMDB_movies.json', 'w') as fp:
    json.dump(KB, fp, indent=4)   


#%%



""" 

        Updating KB with [] when no IMBD_obj were found for standardization
        
        (should have been done during the proceess, but forgot, so here to save 5 hours)


"""

# Load KB

with open('/Users/nicholas/ReDial/Data/PreProcessed/KB_IMDB_movies.json', 'r') as f:
    KB = json.load(f)

# For all items
for values in KB.values():
    # If no 'genres' values, add 'empty value' for each 
    if 'genres' not in values:
        values['IMDB_ID'] = -1
        values['genres'] = []
        values['actors'] = []
        values['directors'] = []
        values['plot'] = ''
        
        
#%%

with open('../PreProcessed/KB_IMDB_movies.json', 'w') as fp:
    json.dump(KB, fp, indent=4)   






#%%































""" 
Problem with DBPedia:
    there is no genres for a film, only subject that users do no use"
"""



























# #%%

# def get_from_DBPedia(query):
    
    
#     """ 
#     Function get_from_DBPedia:

#         Queries DBPedia according to a SQARQL query
#         The query as a literal string
#         Returns a JSON

#     """    
    
    
#     sparql = SPARQLWrapper("http://dbpedia.org/sparql")
#     sparql.setReturnFormat(JSON)

#     sparql.setQuery(query)  # the query as a literal string

#     return sparql.query().convert()



# #%%


    
# def get_subjects_for_uri(end_uri):
    
#     """ 
#     Function get_subjects_for_uri:

#         Takes the end_uri of a movie
#         Returns a list of sujects (strings) related to the movie.

#     """
    
    
#     # Create SPARQL query
#     query = "SELECT * WHERE {<http://dbpedia.org/resource/%s> <http://purl.org/dc/terms/subject> ?Subjects  .\
#              ?Subjects <http://www.w3.org/2000/01/rdf-schema#label> ?Cat .}" % end_uri
    
#     json_result = get_from_DBPedia(query)
    
#     list_subjects = [dic['Cat']['value'] for dic in json_result['results']['bindings']  ]
    
#     return list_subjects



# #%%

# def get_abstract_for_uri(end_uri):
    
#     """ 
#     Function get_subjects_for_uri:

#         Takes the end_uri of a movie
#         Returns a list of sujects (strings) related to the movie.

#     """
    
    
#     # Create SPARQL query
#     query = "PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> \
#              SELECT * WHERE {<http://dbpedia.org/resource/%s> dbpedia-owl:abstract ?abstract. \
#              filter(langMatches(lang(?abstract),'en'))}" % end_uri
#     #         ?Abstract ?Cat .}" % end_uri
    
#  #   print(query)
    
#     json_result = get_from_DBPedia(query)
    
#  #   print('OUT', json_result)
    
#  #   list_subjects = [dic['abstract']['value'] for dic in json_result['results']['bindings']  ]
    
#     #return list_subjects
#     if json_result['results']['bindings'] != []:
#         return json_result['results']['bindings'][0]['abstract']['value']
#     else:
#         return '' 

    
# #%%

# # Test
# print(get_abstract_for_uri("The_Matrix"))


# #%%

# print(get_abstract_for_uri("A_Beautiful_Mind_(film)"))

# #%%

# def prepare_end_uri_list(title, date):
    
#     """ Takes both string title and date. 
#     Returns list of string to replace s in movie URI:
#     <http://dbpedia.org/resource/%s>
    
#     EXAMPLE: 
#         prepare_end_uri_list('nicholas', '(1974)')
#         returns ['nicholas', 'nicholas_(1974)', 'nicholas_(1974_film)', 'nicholas_(film)']
#     """

#     end_uri_list = []
#     title = title.replace(' ','_')
#     end_uri_list.append(title)
#     end_uri_list.append(title + '_' + date)
#     end_uri_list.append(title + '_' + date[:-1] + '_film)')
#     end_uri_list.append(title + '_(film)')
    
#     return end_uri_list



# #%%
    
# print(prepare_end_uri_list('nicholas', '(1974)'))



# #%%



# def get_DBPedia_subjects_abstract_for_movie(title, date):
#     """
#     Method that attributes list of subjects (list of string) and 
#     uri (string) to a movie object (uri is 0 if uri not found)
#     """

#     uri = 0
#     end_uri_list = prepare_end_uri_list(title, date)

#     for end_uri in end_uri_list:
#         subjects = get_subjects_for_uri(end_uri)
#         # If we found something
#         if subjects != []:
#             abstract = get_abstract_for_uri(end_uri)
#             uri = end_uri
#             return subjects, abstract
#             break

#     if uri == 0:
#         print('\n\nURI not found for title *****: ', title, '\n\n')
#         return None, None
            
            
# #%%       
            
            
            
   
# s, a = get_DBPedia_subjects_abstract_for_movie(KB[94298]['title'], '(' + str(KB[94298]['date']) + ')')   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            