#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:01:28 2020


Class Conversation
    - Access to elements of a Conversation in ReDial
        
    - Methods:
            - GetMessages to get Messages objects out of the messages dict in Conversations
            - Messages by chunks of speakers


@author: nicholas
"""

from collections import defaultdict
import copy

from Objects.Message import Message
from Settings import ReD_id_2_ReD_or_id





class Conversation():
    
    
    # Class variable for targets of BERT Data 
    max_qt_ratings = 0
    
    
    
    def __init__(self, conv):
        """
        Initializing a Conversation from ReDial dataset

        Parameters
        ----------
        conv : TYPE: dict.
               FORMAT: {"initiatorWorkerId": int, "respondentWorkerId": int, 
                        "conversationId": str,
                        "messages": [message dict],
                        "movieMentions": {ReD_id(str): tile (str)},
                        "initiatorQuestions": {ReD_id(str): {"suggested": int, "seen": int, "liked": int}}
                        "respondentQuestions": {ReD_id(str): {"suggested": int, "seen": int, "liked": int}}
               DESCRIPTION: A full Conversation from ReDial.

        Returns
        -------
        None.

        """

        self.conv_id = int(conv['conversationId'])
        self.seeker_id = conv['initiatorWorkerId']
        self.recommender_id = conv['respondentWorkerId']
        self.messages = conv['messages']
        self.movie_form_seeker = conv['initiatorQuestions']
        self.movie_form_recommender = conv['respondentQuestions']




    def GetMessages(self):
        """
        Takes a Conversation and returns all the messages as a list of Message objects

        Returns
        -------
        utterances : TYPE: list of Message objects
                           format: [Message_obj]
                     DESCRIPTION: Messages part of conversation 

        """
        
        l_messages = []
        
        for message in self.messages:
            l_messages.append(Message(message, self.seeker_id))
        
        return l_messages
        



    def GetMessagesByChunks(self):
        """
        Takes a Conversation and returns Message objects the by chunks of speaker.
        (If same speaker speaks multiple time in a row, it's now a single Message')
        
        Returns
        -------
        by_chunks : TYPE: list of Message objects
                    FORMAT: [Message_obj]
                    DESCRIPTION: Messages part of conversation by chuncks of same speakers
        """

        by_chucks = []
        
        # Get messages in simple format
        messages = self.GetMessages()
        
        # Initialize with first speaker
        actual_speaker = messages[0].speaker_id
        chunked_message = messages[0]

        # Treat all messages
        for message in messages[1:]:
            # If same speaker, just add the text
            if actual_speaker == message.speaker_id:
                chunked_message.text_raw += ' ' + message.text_raw   
            # If it changes speaker, add actual to by_chucks and initialise new Message
            else:   
                by_chucks.append(chunked_message)
                actual_speaker = message.speaker_id
                chunked_message = message    
                
        # Add last message
        by_chucks.append(chunked_message)        
                
        return by_chucks




    def MessagesAndMentionsByChunks(self):
        """
        Takes a Conversation and returns a list dict. 
        Each dict corresponds to a message chunk, each dict has:
            message
            unique movies mentioned prior to this message
            unique genres mentioned prior to this message
            new unique movies mentioned in this message

        Returns
        -------
        messages_and_mentions : TYPE: list of dict
                                FORMAT: [ ({'message': Message_obj, 
                                            'movies_mentioned': [ReD_id], 
                                            'genres_mentioned': [str],
                                            'new_movies': [ReD_id]} ]
                                DESCRIPTION: (see above)

        """
        
        # List of all dict to return
        messages_and_mentions = []
        
        # Running dict 
        m_n_m = defaultdict(list)
        # Init 'text_mentioned' to ' ' since it's not a list
        m_n_m['text_mentioned'] = ' '
        m_n_m['text_mentioned_nl'] = ' '
        m_n_m['text_mentioned_nl_genres'] = ' '
        
        messages = self.GetMessagesByChunks()
        
        for message in messages:
            
            # Get the new movies
            new_movies = list(set(message.GetMovies()) - set(m_n_m['movies_mentioned']))
            new_genres = list(set(message.GetGenres()) - set(m_n_m['genres_mentioned']))
            
            # Update info for this message
            m_n_m['message'] = message
            m_n_m['new_movies'] = new_movies
            
            # Deep copy dict (so append of list does influence what is already in messages_and_mentions)
            deep_copy_m_n_m = copy.deepcopy(m_n_m)
            
            # Append new message
            messages_and_mentions.append(deep_copy_m_n_m)
            
            # Update info for mext message
            # Add all movies mentioned (because we'll have a rating even if by recommender)
            m_n_m['movies_mentioned'] += new_movies
            # Add only genres mentioned by seeker 
            # (we suppose seeker's mentions are positve, not as certain for recommender's mentions)
            if message.role == 'S::':
                m_n_m['genres_mentioned'] += new_genres  
            # Add text mentioned in this message to the previous. Include role.
            m_n_m['text_mentioned'] += message.role + ' ' + message.text_raw +'  '
            m_n_m['text_mentioned_nl'] += message.role + ' ' + message.TextNL() +'  '
            m_n_m['text_mentioned_nl_genres'] += message.role + ' ' + \
                                                          message.TextNLGenres() +'  '
            
        
        return messages_and_mentions






    def GetMoviesAndRatings(self):
        """
        Takes a Conversation and returns the movies and their ratings (only 
        rated ones).
        First considers Seeker's movie form, then recommender's and returns 
        None if both are empty

        Returns
        -------
        l_ratings : TYPE: dict. (None if no movie form completed)
                          FORMAT: {ReD_id: rating}
                    DESCRIPTION: dict of movies mentioned and their
                                 associated ratings.

        """
        
        # Get an non-empty movie_form 
        # (seeker first, recommender 2nd, drop if none)
        if self.movie_form_seeker != []:
            movie_form = self.movie_form_seeker
        elif self.movie_form_recommender != []:
            movie_form = self.movie_form_recommender
        else:
            return None

        # Get a list of all the ratings 
        l_movies_and_ratings = {}
        # Retreive all movies in movie form with rating provided
        for ReD_id, values in movie_form.items():
            # If we have the rating (==2 would be did not say)
            if values['liked'] == 0 or values['liked'] == 1:
                l_movies_and_ratings[ReD_id] = values['liked']

        return l_movies_and_ratings




    
    def ReDOrIddAndRatings(self, movies):
        """
        Takes a list of movies and returns a list of movies (ReD_or_id) and associated rating 
        in this conversation.  
        
        If no rating exists, return [].
    
        Parameters
        ----------
        movies : TYPE: List of ReD_id
                 FORMAT: [ReD_id(str))]
        conv : TYPE: Conversation obj
    
        Returns
        -------
        data_ED : TYPE: List of tuple
                  FORMAT: [ (ReD_or_id(int), rating) ] 
    
        """
        
        data_ED = []
        
        # Add the attribute 'movies_and_ratings' to this Conversation instance if not there
        if not hasattr(self, 'movies_and_ratings'):
            self.movies_and_ratings = conv.GetMoviesAndRatings()
        
        if self.movies_and_ratings != None:
    
            for m in movies:
                # If there is a rating for this movie
                if self.movies_and_ratings.get(m, None) != None:
                    # ...it becomes a data point
                    data_ED.append([ReD_id_2_ReD_or_id[m], self.movies_and_ratings[m]])
    
        return data_ED




    def BERTTarget(self, target, movies_mentioned):
        """
        Convert targets to format needed for BERT_Reco

        Parameters
        ----------
        target : TYPE: list 
                 FORMAT: [ [ReD_or_id, rating] ]
                 DESCRIPTION: list of movies and ratings
        movies_mentioned : TYPE: list
                FORMAT: [ [ReD_or_id] ]
                DESCRIPTION: Movies that were already mentioned in the conversation

        Returns
        -------
        filled_targets : TYPE: list 
                FORMAT: [(ReD_or_id, ratings)]   
                DESCRIPTION: Starts with (-2,qt_mentioned), then the actual targets and \
                              finishes by filling (-1,0) Until max_qr_ratings in all conv

        """
        
        qt_mentioned = len(movies_mentioned)
        
        # Transform: BERT_Reco needs targets as list of tuples, not list of list 
        target = [(m,r) for m,r in target]       
        
        # Initialize 
        filled_targets = [(-2, qt_mentioned)] + target
        
        # Filling
        fill_size = Conversation.max_qt_ratings - len(target)
        filling = [(-1,0)] * fill_size
        filled_targets += filling
        
        return filled_targets




    def ConversationToDataByRecommendations(self):
        """
        Takes a ReDial Conversation obj and 
        returns **FOUR** sets of data ready for ML (FOR THIS ONE CONVERSATION)
            ED_next: For ED, target is only one movie, the ones mentioned in next message
            ED_all: For ED, targets are all the movies to be mentioned in the rest of conversation
            BERT_next: For BERT, target is only one movie, the ones mentioned in next message
            BERT_all: For BERT, targets are all the movies to be mentioned in the rest of conversation
    
        Parameters
        ----------
        
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
        
        # Initialize for ED and BERT, each for next and all
        ED_next = []
        ED_all = [] 
        BERT_next = defaultdict(list)
        BERT_all = defaultdict(list)
        
        # Add the attribute 'movies_and_ratings' to this Conversation instance if not there
        # (will be used in other methods)
        if not hasattr(self, 'movies_and_ratings'):
            self.movies_and_ratings = self.GetMoviesAndRatings() 
        
        # Keep track of movies to be mentioned, not mentioned yet and that have ratings
        movies_to_be_mentioned = copy.deepcopy(self.movies_and_ratings)
        
        
        # Consider all messages in this conversation
        for m_n_m in self.MessagesAndMentionsByChunks():    
                
            # If no more movies to be mentioned or their were no movies form completed,
            # stop process for this conversation
            if movies_to_be_mentioned == {} or movies_to_be_mentioned == None: break
            
        
            # If message is from recommender and it mentions new_movies (i.e Reco recommends!)
            if m_n_m['message'].role == 'R::' and m_n_m['new_movies'] != []:
                
                # ED_DATA
                                   
                # Add data for every movies in this message in data_next if has rating
                for m in m_n_m['new_movies']:
                    target = self.ReDOrIddAndRatings([m])
                    if target != []:
                        ED_next.append([str(self.conv_id),
                                        self.ReDOrIddAndRatings(m_n_m['movies_mentioned']),
                                        m_n_m['genres_mentioned'],
                                        target])
            
                # Add data for all movies to come (with ratings) in data_all 
                target = self.ReDOrIddAndRatings(list(movies_to_be_mentioned.keys()))
                ED_all.append([str(self.conv_id),
                               self.ReDOrIddAndRatings(m_n_m['movies_mentioned']),
                               m_n_m['genres_mentioned'],
                               target])      
                          
                   
                # BERT_DATA
                
                # Add data for every movies in this message in data_next if has rating
                for m in m_n_m['new_movies']:
                    target = self.ReDOrIddAndRatings([m])
                    if target != []:
                        BERT_next['ConvID'].append(self.conv_id)
                        BERT_next['text_raw'].append(m_n_m['text_mentioned'])
                        BERT_next['text_nl'].append(m_n_m['text_mentioned_nl'])
                        BERT_next['text_nl_genres'].append(m_n_m['text_mentioned_nl_genres'])
                        # Convert to BERT target type
                        B_target = self.BERTTarget(target, m_n_m['movies_mentioned'])
                        BERT_next['ratings'].append(B_target)

                # Add data for all movies to come in data_all if has target
                target = self.ReDOrIddAndRatings(list(movies_to_be_mentioned.keys()))      
                BERT_all['ConvID'].append(self.conv_id)
                BERT_all['text_raw'].append(m_n_m['text_mentioned'])
                BERT_all['text_nl'].append(m_n_m['text_mentioned_nl'])
                BERT_all['text_nl_genres'].append(m_n_m['text_mentioned_nl_genres'])
                # Convert to BERT target type
                B_target = self.BERTTarget(target, m_n_m['movies_mentioned'])
                BERT_all['ratings'].append(B_target) 

            
            # Remove all movies in this message from movies_to_be_mentioned 
            for m in m_n_m['new_movies']:
                movies_to_be_mentioned.pop(m, None)
                
           
            
        return ED_next, ED_all, BERT_next, BERT_all












if __name__ == '__main__':
    
  #  ex_conv = {"initiatorWorkerId":7,"respondentWorkerId":6,"conversationId":"411","messages":[{"messageId":1108,"text":"Hello how are you?","timeOffset":0,"senderWorkerId":6},{"messageId":1109,"text":"Which movies do you suggest to watch?","timeOffset":11,"senderWorkerId":6},{"messageId":1110,"text":"Hey There! Im good. I love scary movies. Have you ever seen @166736","timeOffset":66,"senderWorkerId":7},{"messageId":1111,"text":"I love scary movies too! No I haven't seen that movie yet.","timeOffset":92,"senderWorkerId":6},{"messageId":1112,"text":"Have","timeOffset":94,"senderWorkerId":6},{"messageId":1113,"text":"I love it because the sharks make it so much more scary. I would never want to be stuck like that with them swimming near me! How about you.","timeOffset":146,"senderWorkerId":7},{"messageId":1114,"text":"Have you seen the movie @184418","timeOffset":148,"senderWorkerId":6},{"messageId":1115,"text":"I don't like sharks, And that would be very scary.","timeOffset":173,"senderWorkerId":6},{"messageId":1116,"text":"What other movies do you like?","timeOffset":207,"senderWorkerId":6},{"messageId":1117,"text":"Yes! love that movie as well! Have you seen @204334","timeOffset":213,"senderWorkerId":7},{"messageId":1118,"text":"me neither!","timeOffset":217,"senderWorkerId":7},{"messageId":1119,"text":"I love funny movies as well! Always up for a good laugh!","timeOffset":238,"senderWorkerId":7},{"messageId":1120,"text":"No I haven't seen it yet.","timeOffset":265,"senderWorkerId":6},{"messageId":1121,"text":"I like the movie @204331","timeOffset":287,"senderWorkerId":6},{"messageId":1122,"text":"I love a good laugh as well!","timeOffset":297,"senderWorkerId":6},{"messageId":1123,"text":"Have you seen the mov''ie @ Logan","timeOffset":324,"senderWorkerId":6},{"messageId":1124,"text":"*movie","timeOffset":335,"senderWorkerId":6},{"messageId":1125,"text":"I love that movie also! and nope. Is it good?","timeOffset":344,"senderWorkerId":7},{"messageId":1126,"text":"Yes it is a good action movie. It's where Logan cares for a professor.","timeOffset":404,"senderWorkerId":6},{"messageId":1127,"text":"thats so cute!","timeOffset":422,"senderWorkerId":7},{"messageId":1128,"text":"And he has to defend a young girl from the dark forces that is out to get her.","timeOffset":448,"senderWorkerId":6}],"movieMentions":{"166736":"47 Meters Down (2017)","184418":"Get Out (2017)","204334":"Happy Death Day  (2017)","204331":"Girls Trip (2017)"},"initiatorQuestions":[],"respondentQuestions":[]}
  #  ex_conv = {"initiatorWorkerId": 959, "respondentWorkerId": 958, "conversationId": "20041", "messages": [{"messageId": 204187, "text": "Hello!", "timeOffset": 0, "senderWorkerId": 958}, {"messageId": 204188, "text": "Hello!", "timeOffset": 8, "senderWorkerId": 959}, {"messageId": 204189, "text": "What kind of movies do you like?", "timeOffset": 30, "senderWorkerId": 958}, {"messageId": 204190, "text": "I am looking for a movie recommendation.   When I was younger I really enjoyed the @77161", "timeOffset": 60, "senderWorkerId": 959}, {"messageId": 204191, "text": "Oh, you like scary movies?", "timeOffset": 82, "senderWorkerId": 958}, {"messageId": 204192, "text": "I recently watched @204334", "timeOffset": 99, "senderWorkerId": 958}, {"messageId": 204193, "text": "I also enjoyed watching @132562", "timeOffset": 101, "senderWorkerId": 959}, {"messageId": 204194, "text": "It was really good for a new &quot;scary movie&quot;", "timeOffset": 115, "senderWorkerId": 958}, {"messageId": 204195, "text": "I do enjoy some of the newer horror movies that I have seen as well.", "timeOffset": 141, "senderWorkerId": 959}, {"messageId": 204196, "text": "I heard that @205430 is good. It is still in theaters though.", "timeOffset": 163, "senderWorkerId": 958}, {"messageId": 204197, "text": "I really liked the movie @125431", "timeOffset": 168, "senderWorkerId": 959}, {"messageId": 204198, "text": "Me, too!", "timeOffset": 173, "senderWorkerId": 958}, {"messageId": 204199, "text": "It was really creepy, but I thought it was good!", "timeOffset": 183, "senderWorkerId": 958}, {"messageId": 204200, "text": "Or @118338 I saw while in theaters, this was a very good movie.  It had me on the edge of my seat for the whole show.", "timeOffset": 245, "senderWorkerId": 959}, {"messageId": 204201, "text": "I'm not sure if I saw that one, I'll have to check into it. Sounds familiar, but not sure. Thank you for your suggestions!", "timeOffset": 274, "senderWorkerId": 958}, {"messageId": 204202, "text": "Are there any comedies that you would suggest?", "timeOffset": 310, "senderWorkerId": 959}, {"messageId": 204203, "text": "Sure! I like comedies a lot. I like movies like @175203 and @111776 , but I also like @187061 and @157765 .", "timeOffset": 376, "senderWorkerId": 958}, {"messageId": 204204, "text": "Wonderful! Thank you so much I think I am ready for movie night now.", "timeOffset": 410, "senderWorkerId": 959}, {"messageId": 204205, "text": "No problem! Thank you, too! :)", "timeOffset": 422, "senderWorkerId": 958}], "movieMentions": {"77161": "A Nightmare on Elm Street (1984)", "204334": "Happy Death Day  (2017)", "132562": "The Last House on the Left  (1972)", "205430": "A Quiet Place (2018)", "125431": "Annabelle  (2014)", "118338": "The Forest  (2016)", "175203": "Hot Tub Time Machine", "111776": "Super Troopers (2001)", "187061": "Identity Thief (2013)", "157765": "The Heat  (2013)"}, "initiatorQuestions": {"77161": {"suggested": 0, "seen": 1, "liked": 1}, "204334": {"suggested": 1, "seen": 2, "liked": 2}, "132562": {"suggested": 0, "seen": 1, "liked": 1}, "205430": {"suggested": 1, "seen": 2, "liked": 2}, "125431": {"suggested": 0, "seen": 1, "liked": 1}, "118338": {"suggested": 0, "seen": 1, "liked": 1}, "111776": {"suggested": 1, "seen": 2, "liked": 2}, "157765": {"suggested": 1, "seen": 2, "liked": 2}, "175203": {"suggested": 1, "seen": 2, "liked": 2}, "187061": {"suggested": 1, "seen": 2, "liked": 2}}, "respondentQuestions": {"77161": {"suggested": 0, "seen": 1, "liked": 1}, "204334": {"suggested": 1, "seen": 2, "liked": 2}, "132562": {"suggested": 0, "seen": 1, "liked": 1}, "205430": {"suggested": 1, "seen": 2, "liked": 2}, "125431": {"suggested": 0, "seen": 1, "liked": 1}, "118338": {"suggested": 0, "seen": 1, "liked": 1}, "111776": {"suggested": 1, "seen": 2, "liked": 2}, "157765": {"suggested": 1, "seen": 2, "liked": 2}, "175203": {"suggested": 1, "seen": 2, "liked": 2}, "187061": {"suggested": 1, "seen": 2, "liked": 2}}}
  #  ex_conv = {"movieMentions": {"76012": "The Exorcist  (1973)", "205163": "Avengers: Infinity War (2018)", "165531": "Spider-Man 3 (2007)", "99583": "Iron Man  (2008)", "78874": "Click  (2006)"}, "respondentQuestions": {"76012": {"suggested": 1, "seen": 1, "liked": 1}, "205163": {"suggested": 1, "seen": 1, "liked": 1}, "165531": {"suggested": 1, "seen": 1, "liked": 1}, "99583": {"suggested": 1, "seen": 1, "liked": 1}, "78874": {"suggested": 1, "seen": 1, "liked": 1}}, "messages": [{"timeOffset": 0, "text": "Hello", "senderWorkerId": 960, "messageId": 204309}, {"timeOffset": 7, "text": "me what kind of movies do you like?", "senderWorkerId": 960, "messageId": 204310}, {"timeOffset": 8, "text": "I like all movies, what are your suggestions?", "senderWorkerId": 961, "messageId": 204311}, {"timeOffset": 15, "text": "Did you watch @205163 ?", "senderWorkerId": 960, "messageId": 204312}, {"timeOffset": 23, "text": "No heard it is very good", "senderWorkerId": 961, "messageId": 204313}, {"timeOffset": 27, "text": "I can tell you to watch @78874", "senderWorkerId": 960, "messageId": 204314}, {"timeOffset": 33, "text": "Yes it is really good", "senderWorkerId": 960, "messageId": 204315}, {"timeOffset": 44, "text": "I love Adam Sandler movies", "senderWorkerId": 961, "messageId": 204316}, {"timeOffset": 49, "text": "You can watch too if you want to @76012", "senderWorkerId": 960, "messageId": 204317}, {"timeOffset": 56, "text": "But that is a scary one", "senderWorkerId": 960, "messageId": 204318}, {"timeOffset": 67, "text": "i would recommend you @99583", "senderWorkerId": 960, "messageId": 204319}, {"timeOffset": 71, "text": "Sounds scary but will have to check it out!", "senderWorkerId": 961, "messageId": 204320}, {"timeOffset": 78, "text": "These are great suggestions", "senderWorkerId": 961, "messageId": 204321}, {"timeOffset": 79, "text": "And also @165531", "senderWorkerId": 960, "messageId": 204322}, {"timeOffset": 82, "text": "Thank you!", "senderWorkerId": 961, "messageId": 204323}, {"timeOffset": 82, "text": "Yes", "senderWorkerId": 960, "messageId": 204324}, {"timeOffset": 87, "text": "I hope i have helped", "senderWorkerId": 960, "messageId": 204325}, {"timeOffset": 89, "text": "Good bye", "senderWorkerId": 960, "messageId": 204326}, {"timeOffset": 92, "text": "Enjoy them", "senderWorkerId": 960, "messageId": 204327}, {"timeOffset": 94, "text": "Goodbye", "senderWorkerId": 961, "messageId": 204328}], "conversationId": "20055", "respondentWorkerId": 960, "initiatorWorkerId": 961, "initiatorQuestions": {"76012": {"suggested": 1, "seen": 0, "liked": 1}, "205163": {"suggested": 1, "seen": 0, "liked": 1}, "165531": {"suggested": 1, "seen": 1, "liked": 1}, "99583": {"suggested": 1, "seen": 1, "liked": 1}, "78874": {"suggested": 1, "seen": 1, "liked": 1}}}
    ex_conv = {"movieMentions": {"125431": "Annabelle  (2014)", "118338": "The Forest  (2016)", "119295": "A Nightmare on Elm Street  (2010)", "130591": "Friday the 13th  (1980)", "161244": "Nocturnal Animals  (2016)", "77161": "A Nightmare on Elm Street (1984)", "144779": "Annabelle 2 (2017)", "157190": "Arrival  (2016)"}, "respondentQuestions": {"125431": {"suggested": 1, "seen": 0, "liked": 2}, "118338": {"suggested": 1, "seen": 2, "liked": 2}, "119295": {"suggested": 0, "seen": 1, "liked": 1}, "130591": {"suggested": 1, "seen": 2, "liked": 2}, "161244": {"suggested": 0, "seen": 2, "liked": 2}, "77161": {"suggested": 1, "seen": 1, "liked": 1}, "144779": {"suggested": 0, "seen": 1, "liked": 1}, "157190": {"suggested": 0, "seen": 2, "liked": 2}}, "messages": [{"timeOffset": 0, "text": "Hello, I hear you are looking for movie recommendations.  Do you have any sort of genre in mind?", "senderWorkerId": 959, "messageId": 204857}, {"timeOffset": 37, "text": "Yes, I'd recommend psychological thrillers.  Have you seen @161244?", "senderWorkerId": 965, "messageId": 204858}, {"timeOffset": 95, "text": "No I have never seen that one.  I will have to check it out.  Growing up I always like movies like @77161  and @130591 .", "senderWorkerId": 959, "messageId": 204859}, {"timeOffset": 156, "text": "As for newer movies I really enjoyed @118338 . It was riveting.", "senderWorkerId": 959, "messageId": 204860}, {"timeOffset": 157, "text": "Those are classics.  I'd like to see @77161 or the remake @119295.", "senderWorkerId": 965, "messageId": 204861}, {"timeOffset": 170, "text": "I have not seen the remake.", "senderWorkerId": 959, "messageId": 204862}, {"timeOffset": 192, "text": "Have you seen @157190?", "senderWorkerId": 965, "messageId": 204863}, {"timeOffset": 234, "text": "No I have not seen that one either.  I guess I have a few to look into.  I really liked @125431 though.", "senderWorkerId": 959, "messageId": 204864}, {"timeOffset": 272, "text": "I enjoyed @144779 but never did see the original.  I felt like the sequel stood well on its own.", "senderWorkerId": 965, "messageId": 204865}, {"timeOffset": 287, "text": "I will have to check out the original!", "senderWorkerId": 965, "messageId": 204866}, {"timeOffset": 312, "text": "Well you have a wonderful evening I hope I helped.", "senderWorkerId": 959, "messageId": 204867}, {"timeOffset": 326, "text": "You certainly have!  Thanks!", "senderWorkerId": 965, "messageId": 204868}, {"timeOffset": 335, "text": "Thank you, good bye.", "senderWorkerId": 959, "messageId": 204869}, {"timeOffset": 385, "text": "Goodbye.", "senderWorkerId": 965, "messageId": 204870}], "conversationId": "20277", "respondentWorkerId": 959, "initiatorWorkerId": 965, "initiatorQuestions": {"125431": {"suggested": 1, "seen": 0, "liked": 1}, "118338": {"suggested": 1, "seen": 2, "liked": 2}, "119295": {"suggested": 0, "seen": 0, "liked": 1}, "130591": {"suggested": 1, "seen": 2, "liked": 2}, "161244": {"suggested": 0, "seen": 1, "liked": 1}, "77161": {"suggested": 1, "seen": 0, "liked": 1}, "144779": {"suggested": 0, "seen": 1, "liked": 1}, "157190": {"suggested": 0, "seen": 2, "liked": 2}}}
    conv = Conversation(ex_conv)
  #  print([m.GetGenres() for m in conv.GetMessagesByChunks()])
    mnm = conv.MessagesAndMentionsByChunks()
    print(conv.GetMessages()[1].GetGenres())
    ED_next, ED_all, BERT_next, BERT_all = conv.ConversationToDataByRecommendations()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    