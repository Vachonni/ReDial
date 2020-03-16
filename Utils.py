#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:53:48 2018


Classes and functions for ReDial project.


@author: nicholas
"""

import numpy as np
from torch.utils import data
import torch
import time
import json

from Objects.MetricByMentions import MetricByMentions




"""
DATASET - Sub classes of Pytorch DATASET to prepare for Dataloader

"""



class RnGChronoDataset(data.Dataset):
    """    
    
    ****** Now inputs and targets are seperated in data ******
    
    
    INPUT: 
        RnGlist format is:
            ["ConvID", [(UiD, Rating) mentionned], ["genres"], [(UiD, Rating) to be mentionned]]
        top_cut is the number of movies in genres vector
        If data from a non-chrono dataset (eg: ML), all data in mentionned (to be mentionned empty).
    
    RETUNRS:
        masks, (inputs, genres) and targets. 
        Genres is vector with value for top_cut movies of intersection of genres mentionned 
        by user, normalized (or deduced for ML).

    """
    
    def __init__(self, RnGlist, dict_genresInter_idx_UiD, nb_movies, popularity, DEVICE, \
                 exclude_genres=False, top_cut=100):
        self.RnGlist = RnGlist
        self.dict_genresInter_idx_UiD = dict_genresInter_idx_UiD
        self.nb_movies = nb_movies
        self.popularity = popularity
        self.DEVICE = DEVICE
        self.exclude_genres = exclude_genres
        self.top_cut = top_cut
        
        
    def __len__(self):
        "Total number of samples. Here one sample corresponds to a new mention in Conversation"
        return len(self.RnGlist)


    def __getitem__(self, index):
        "Generate one sample of data."
        
        # Get list of movies and ratings for user number (=index) 
        ConvID, l_inputs, l_genres, l_targets = self.RnGlist[index]
        
        # Init
        inputs = torch.zeros(self.nb_movies)
        targets = torch.zeros(self.nb_movies)   
        masks_inputs = torch.zeros(self.nb_movies)
        masks_targets = torch.zeros(self.nb_movies)
        genres = torch.zeros(self.nb_movies)
        
        # Inputs
        for uid, rating in l_inputs:
            inputs[uid] = rating
            masks_inputs[uid] = 1
                
        # Targets 
        for uid, rating in l_targets:
            targets[uid] = rating
            masks_targets[uid] = 1
        
        # Genres
        if not self.exclude_genres:
            # Turn list of genres into string
            str_genres = str(l_genres)
            # Try - if no movies of that genres (key error)
            try:
                genres_idx, l_genres_uid = self.dict_genresInter_idx_UiD[str_genres] 
            except:
       #         print('No movie with genres:', str_genres)
                genres_idx = 1
            # If there is a genres...   (no else needed, since already at 0)
            if genres_idx != 1:
                for uid in l_genres_uid:
                    genres[uid] = 1.0
                    
                """normalization and popularity"""
                # Include popularity in genres
                genres = genres * self.popularity
                # Take top 100 movies
                genres_cut = torch.zeros(self.nb_movies)
                genres_cut[genres.topk(self.top_cut)[1]] = genres.topk(self.top_cut)[0]
                genres = genres_cut  
                # Normalize vector
                genres = torch.nn.functional.normalize(genres, dim=0)
        
        
        return (masks_inputs.to(self.DEVICE), masks_targets.to(self.DEVICE)), \
               (inputs.to(self.DEVICE), (genres_idx, genres.to(self.DEVICE))), \
               targets.to(self.DEVICE)








"""

TRAINING AND EVALUATION 

"""



def Train(train_loader, model, criterion, optimizer, completion):
    model.train()
    train_loss = 0
    nb_batch = len(train_loader) * completion / 100
    
   
    print('TRAINING')
     
    for batch_idx, (masks, inputs, targets) in enumerate(train_loader):
        
        # Early stopping
        if batch_idx > nb_batch: 
            print(' *EARLY stopping')
            break
        
        # Print update
        if batch_idx % 10 == 0: 
            print('Batch {:4d} out of {:4.1f}.    Loss on targets: {:.4f}'\
                  .format(batch_idx, nb_batch, train_loss/(batch_idx+1)))  
                        
        # re-initialize the gradient computation
        optimizer.zero_grad()   
        
        # Make prediction            
        pred = model(inputs)
 
        # Evaluate "Masked BCE" to get loss value
        loss = (criterion(pred, targets) * masks[1]).sum()
        assert loss >= 0, 'Getting a negative loss in training - IMPOSSIBLE'
        nb_ratings = masks[1].sum()
        loss /= nb_ratings
        
        loss.backward()
        optimizer.step()

        train_loss += loss
        
    train_loss /= nb_batch
        
    return train_loss.item()





def Eval(valid_loader, model, criterion, completion, topx=100):
    """
    Prediction on targets = to be mentionned movies...
    
    ** Only works with RnGChronoDataset **
    
    """
    model.eval()
    nb_batch = len(valid_loader) * completion / 100    
    
    # METRICS
    # eval_loss (does not depend on qt of movie's mentions, hence an int)
    metrics = {'eval_loss': 0}
    # MetricByMentions objects (depend on qt of movie's mentions)
    metrics_values = ['avrg_rank', 'ndcg', 'recall@1', 'recall@10', 'recall@50']
    # Initializing 
    for m in metrics_values:
        metrics[m] = MetricByMentions(m)


    print('\nEVALUATION')      
      
    with torch.no_grad():
        for batch_idx, (masks, inputs, targets) in enumerate(valid_loader):
            
            # Early stopping 
            if batch_idx > nb_batch or nb_batch == 0: 
                print('EARLY stopping')
                break
            
            # Print Update
            if batch_idx % 10 == 0:
                print('Batch {} out of {}.  Loss:{}'\
                      .format(batch_idx, nb_batch, metrics['eval_loss'] /(batch_idx+1)))
                         
            # Make a pred
            pred = model(inputs)  
            
            # Evaluate "Masked BCE" to get loss value
            loss = (criterion(pred, targets) * masks[1]).sum()
            assert loss >= 0, 'Getting a negative loss in eval - IMPOSSIBLE'
            nb_ratings = masks[1].sum()
            loss = loss / nb_ratings
            metrics['eval_loss'] += loss

            # FOR OTHER METRICS
            # Evaluating each sample seperately, since number of targets vary
            # For each sample in the batch
            for i in range(len(pred)):
                
                # Insure their is at least one target movie 
                # (if not, sample not considered)
                if targets[i].sum() == 0: continue
                
                # Index of targets at 1 (i.e. liked movies) 
                targets_idx = targets[i].nonzero().flatten().tolist()
                
                # Get Ranks for targets (we only care about liked movies)
                avrg_rk, ndcg, re_1, re_10, re_50 = Ranks(pred[i], \
                                                    pred[i][targets_idx], topx)  
                
                # Get the number of inputs mentionned before prediction
                mentions = masks[0][i].sum(dtype=torch.uint8).item()
                
                # Add Ranks' results to appropriate MetricByMentions obj
                metrics['avrg_rank'].Add(avrg_rk, mentions)
                metrics['ndcg'].Add(ndcg, mentions)
                metrics['recall@1'].Add(re_1, mentions)
                metrics['recall@10'].Add(re_10, mentions)
                metrics['recall@50'].Add(re_50, mentions)

    
    metrics['eval_loss'] /= nb_batch   
    metrics['eval_loss'] = metrics['eval_loss'].item()  

    return metrics







"""

METRICS

"""

    

# DCG (Discounted Cumulative Gain)   
 
# Needed to compare rankings when the numbre of item compared are not the same
# and/or when relevance is not binary

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
    

    

def Ranks(all_values, values_to_rank, topx = 0):
    """
    Takes 2 numpy array and return, for all values in values_to_rank,
    the average rank, nDCG and Recall@{1,10,50} for ranks smaller than topx
    """    
    
    qt_values_to_rank = len(values_to_rank)
    
    # If topx not mentionned (no top), it's for all the values
    if topx == 0: topx = len(all_values)
    
    # Initiate ranks
    ranks = np.zeros(qt_values_to_rank)
    
    for i,v in enumerate(values_to_rank):
        ranks[i] = len(all_values[all_values > v]) + 1
        
    ndcg = nDCG(ranks, topx, qt_values_to_rank)
    
    recall_1 = (ranks <= 1).sum() /  qt_values_to_rank
    recall_10 = (ranks <= 10).sum() / qt_values_to_rank
    recall_50 = (ranks <= 50).sum() / qt_values_to_rank
    
    if ranks.sum() == 0: print('warning, should always be at least one rank')
    
    return ranks.mean(), ndcg, recall_1, recall_10, recall_50
    





"""

PRINTS

"""

def PrintResults(metrics, epoch, model, metrics_to_print=['ndcg']):
    """
    Printing different metrics for a specific epoch
    
    Parameters
    ----------
    metrics : dict
        All metrics evaluated for an epoch.
    epoch : int
        The epoch concern with those results.
    model : torch.nn
        The model we are training.
    metrics_to_print : list of str, optional
        List of metrics that will be printed. Correspond to keys from the metrics 
        The default is ['ndcg'].

    Returns
    -------
    None.
    """
    
    # Always print epoch and losses
    print('\nEND EPOCH {:3d}'.format(epoch))
    print('Train Loss on targets: {:.4f}'.format(metrics['train_loss']))
    print('Valid Loss on targets: {:.4f}'.format(metrics['eval_loss']))
    # TODO: Keep this print. If not, remove 'model' from arguments
    print("Parameter g - Avrg: {:.4f} Min: {:.4f} Max: {:.4f}" \
          .format(model.g.data.mean().item(), model.g.data.min().item(), \
                  model.g.data.max().item()))
    
    
    # Print other metrics desired    
    for m in metrics_to_print:   
        print('avrg_'+metrics[m].name+' on targets: {:.4f}'.format(metrics[m].Avrg()))

    





"""

TENSORBOARD

"""



def ToTensorboard(tb, metrics, epoch, metrics_to_track=['ndcg']):
    """
    Adding different metrics to Tensorboard for a specific epoch
    
    Parameters
    ----------
    tb: SummaryWriter obj of Tensorboard
        Instance to write to our Tensorboard
    metrics : dict
        All metrics evaluated for an epoch.
    epoch : int
        The epoch concern with those results.
    metrics_to_track : list of str, optional
        List of metrics that will be tracked. Correspond to keys from the metrics 
        The default is ['ndcg'].

    Returns
    -------
    None.
    """
    
    # Always keep track of losses (in same graph)
    tb.add_scalars('Losses', {'train': metrics['train_loss'], \
                              'valid': metrics['eval_loss']}, epoch)
    
    # Track other metrics desired    
    for m in metrics_to_track:
        # Get the plot
        fig = metrics[m].Plot()   
        # Add to Tensorboard
        tb.add_scalar('avrg_'+metrics[m].name, metrics[m].Avrg(), epoch)
        tb.add_figure('avrg_'+metrics[m].name+'_by_mentions', fig, epoch)
        tb.close()
    





"""

OTHERS

"""


def SaveExperiment(args):
    
    # Set args.id with current GMT time
    args.id = time.asctime(time.gmtime())
    
    # Load Experiement.json
    with open('Experiments.json', 'r') as fp:
        exp = json.load(fp)
    # Add this experiment
    exp[args.id] = args.__dict__
    # Save Experiement.json
    with open('Experiments.json', 'w') as fp:
        json.dump(exp, fp, indent=4, sort_keys=True)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






