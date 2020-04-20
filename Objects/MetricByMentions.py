#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:43:31 2020


Class MetricByMentions  +  Metrics evaluation  +  Tensorboard updates

(See information below)



@author: nicholas
"""


from collections import defaultdict
from statistics import mean, stdev
import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats as ss




class MetricByMentions:
    """
    Defines object composed of:
        
        self.name = str, name of the metric (ex:ndcg, recall,...)
        
        self.infos = defaultdict(list), where:
                        key = int, qt of movies mentioned
                        values = list, metric's values for a data point 
                                       associated with this qt of movies mentioned 
    """
    
    # A class variable to establish max quantity of mentions evaluated 
    # (used in: AvrgByMentions, StdByMentions and Plot)
    # We'll have for 0 to max_mentions-1 
    max_mentions = 6      
    
    
    def __init__(self, name, infos=None):
        
        self.name = name
        
        if infos == None:
            self.infos = defaultdict(list)
        else:
            assert isinstance(infos, defaultdict), 'MetricByMentions needs to be a defaultdict'
            self.infos = infos
                 
            
            
            
    def Add(self, value, mentions):
        """
        Adds a new values to the object, according to the qt of movies mentioned (key)
        """
        self.infos[mentions].append(value)
        
        
        
        
    def Combine(self, other_metric_by_mentions):
        """
        Combine a metric_by_metion object to another one
        """
        # For each qt of mentions in the other, add it to the main one
        for qt_of_mentions, values in other_metric_by_mentions.infos.items():
            self.infos[qt_of_mentions] += values
        
        
        
    def Avrg(self):   
        """
        Average of all metric's values (independant of mentions)
        Returns a int
        """
        return mean([value for values_by_mentions in self.infos.values() \
                     for value in values_by_mentions])
            
            

            
    def AvrgByMentions(self):
        """
        Average of all metric's values by qt of movies menitioned
        Returns a ordered list (i.e indexes correspond from
                                0 to MetricByMentions.max_mentions-1)
        """
        return [mean(self.infos[i]) for i in range(MetricByMentions.max_mentions)]        
    
    
    
    
    def StdByMentions(self):
        """
        Standard Deviaition of all metric's values by qt of movies menitioned
        Returns a ordered list (i.e indexes correspond from
                                0 to MetricByMentions.max_mentions-1)
        """        
        return [stdev(self.infos[i]) for i in range(MetricByMentions.max_mentions)]   
    
    
    
        
    def Plot(self):
        """
        Plots a double graph.
            x - qt of movies mentioned from 0 to MetricByMentions.max_mentions-1
            y left - values of the metric by qt of movies mentioned
            y right - proportion of data points per qt of movies mentioned
        Returns mathplotlib figure
        """        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Qt of movies mentioned before prediction')
        ax1.set_ylabel(self.name, color='darkblue')
        # Fix range of y_left axis by metric
        if self.name == 'ndcg' or self.name == 'recall@10':
            ax1.set_ylim(0, 0.25)
        elif self.name == 'recall@1':
            ax1.set_ylim(0, 0.06)
        elif self.name == 'recall@50':
            ax1.set_ylim(0.15, 0.5)
            # Include KBDR results (manually extracted from graph on paper) if...   
            if MetricByMentions.max_mentions == 6:
                ax1.plot(range(MetricByMentions.max_mentions), \
                         [0.25, 0.335, 0.383, 0.358, 0.371, 0.379], \
                         '--', color='tan', alpha=0.7)
                
        ax1.plot(range(MetricByMentions.max_mentions), self.AvrgByMentions(), \
                 color='darkblue')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        # How many data points in total
        qt_data = sum(len(l_v_by_m) for l_v_by_m in self.infos.values())
        qt_data_by_mentions = [len(self.infos[i]) / qt_data for i in \
                               range(MetricByMentions.max_mentions)]
        ax2.set_ylabel('Proportion in set (%)', color='gray')
        ax2.set_ylim(0,0.3)
        ax2.bar(range(MetricByMentions.max_mentions), qt_data_by_mentions, \
                alpha=0.3, color='gray')
        
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
    ranks = ss.rankdata((-1*all_values).cpu(), method='min')[idx_to_rank]
        
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

    # Track model's parameters (if not None)
    if model != None:
        for name, weights in model.named_parameters():
            tb.add_histogram(name, weights, epoch)


    tb.close()
    