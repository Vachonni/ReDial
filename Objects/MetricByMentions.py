#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:43:31 2020


Class MetricByMentions

    # - Access to elements of a Conversation in ReDial
        
    # - Methods:
    #         - GetMessages to get Messages objects out of the messages dict in Conversations
    #         - Messages by chunks of speakers



@author: nicholas
"""

from collections import defaultdict
from statistics import mean
import matplotlib.pyplot as plt 


class MetricByMentions:
    
    
    def __init__(self, name, infos=None):
        
        self.name = name
        
        if infos == None:
            self.infos = defaultdict(list)
        else:
            assert isinstance(infos, defaultdict), 'MetricByMentions needs to be a defaultdict'
            self.infos = infos
                 
            
            
            
    def Add(self, value, mentions):
        
        self.infos[mentions].append(value)
        
        
        
        
    def Avrg(self):   
        
        return mean([value for values_by_mentions in self.infos.values() \
                     for value in values_by_mentions])
            
            
            
            
    def AvrgByMentions(self):

        return [mean(self.infos[i]) for i in range(9)]         
    
    
        
    def Plot(self):
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Qt of movies mentioned before prediction')
        ax1.set_ylabel(self.name, color='darkblue')
        ax1.plot(self.AvrgByMentions(), color='darkblue')
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        qt_data_by_mentions = [len(self.infos[i]) for i in range(9)]
        ax2.set_ylabel('Qt of data points', color='gray')
        ax2.bar(range(9), qt_data_by_mentions, alpha=0.3, color='gray')
        
        fig.tight_layout()
        plt.show()
        
        return fig
    
        
    
    
    