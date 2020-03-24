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
from statistics import mean, stdev
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats as ss




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

        print (self.name, '\n', self.infos)
        
        return [mean(self.infos[i]) for i in range(9)]        
    
    
    
    
    def StdByMentions(self):
        
        return [stdev(self.infos[i]) for i in range(9)]   
    
    
    
        
    def Plot(self):
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Qt of movies mentioned before prediction')
        ax1.set_ylabel(self.name, color='darkblue')
        ax1.errorbar(range(9), self.AvrgByMentions(), self.StdByMentions(), \
                     elinewidth=0.5, color='darkblue')
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        qt_data_by_mentions = [len(self.infos[i]) for i in range(9)]
        ax2.set_ylabel('Qt of data points', color='gray')
        ax2.bar(range(9), qt_data_by_mentions, alpha=0.3, color='gray')
        
        fig.tight_layout()
        plt.show()
        
        return fig
    
        
    




"""

METRIC'S EVALUATION

"""

    

# DCG (Discounted Cumulative Gain)   

# Needed to compare rankings when the number of items compared are not the same
# and/or when relevance is not binary and/or each place in ranks is revelant 
#(not only the presence or not as in Recall)

def DCG(v, top):
    """
    V is vector of ranks, lowest is better
    top is the max rank considered 
    Relevance is 1 if items in rank vector, 0 else
    """
    
    discounted_gain = 0
    
    for i in np.round(v):
        if i <= top:
            discounted_gain += 1/np.log2(i+1)

    return round(discounted_gain, 4)




def nDCG(v, top, nb_values=0):
    """
    DCG normalized with what would be the best evaluation.
    
    nb_values is the max number of good values there is. If not specified or bigger 
    than top, assumed to be same as top.
    """
    if nb_values == 0 or nb_values > top: nb_values = top
    dcg = DCG(v, top)
    idcg = DCG(np.arange(nb_values)+1, top)
    
    return round(dcg/idcg, 4)




def GetMetrics(all_values, idx_to_rank, topx = 0):
    """
    Takes torch tensor of predictied ratings for all movies and 
    list of idx_to_rank (i.e. indexes of movies rated 1)
    and returns average rank, nDCG and recall@{1,10,50}.
    """    
    
    qt_values_to_rank = len(idx_to_rank)
    
    # If topx not mentionned (no top), it's for all the values
    if topx == 0: topx = len(all_values)
    
    # -1 because because ranking according to increasing values, we want decreasing
    ranks = ss.rankdata((-1*all_values).cpu(), method='average')[idx_to_rank]
        
    ndcg = nDCG(ranks, topx, qt_values_to_rank)
    
    recall_1 = (ranks <= 1).sum() /  qt_values_to_rank
    recall_10 = (ranks <= 10).sum() / qt_values_to_rank
    recall_50 = (ranks <= 50).sum() / qt_values_to_rank    
    
    if ranks.sum() == 0: print('warning, should always be at least one rank')
    
    return ranks.mean(), ndcg, recall_1, recall_10, recall_50

    






"""

TENSORBOARD

"""



def ToTensorboard(tb, metrics, epoch, model, metrics_to_track=['ndcg']):
    """
    Adding different metrics to Tensorboard for a specific epoch
    
    Parameters
    ----------
    tb: SummaryWriter obj of Tensorboard
        Instance to write to our Tensorboard
    metrics: dict of MetricByMentions obj
        All metrics evaluated for an epoch.
    epoch: int
        The epoch concern with those results.
    model: torch.nn.Module
        The model we are training
    metrics_to_track : list of str, optional
        List of metrics that will be tracked. Correspond to keys from the metrics 
        The default is ['ndcg'].

    Returns
    -------
    None.
    """
    
    # Keep track of losses (in same graph) when training
    if 'train_loss' in metrics:
        tb.add_scalars('Losses', {'train': metrics['train_loss'], \
                                  'valid': metrics['eval_loss']}, epoch)
    
    # Track other metrics desired    
    for m in metrics_to_track:
        # Get the plot
        fig = metrics[m].Plot()   
        # Add to Tensorboard
        tb.add_scalar('avrg_'+metrics[m].name, metrics[m].Avrg(), epoch)
        tb.add_figure('avrg_'+metrics[m].name+'_by_mentions', fig, epoch)

    # Track model's parameters
    for name, weights in model.named_parameters():
        tb.add_histogram(name, weights, epoch)


    tb.close()
    