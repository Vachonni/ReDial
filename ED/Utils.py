#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:53:48 2018


Classes and functions for ReDial project.


@author: nicholas
"""


import sys
from pathlib import Path 
from torch.utils import data
import torch
import time
import json
    

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
    

from Objects.MetricByMentions import MetricByMentions
from Objects.MetricByMentions import GetMetrics



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
                
                # Get metrics for targets (we only care about liked movies)
                avrg_rk, ndcg, re_1, re_10, re_50 = GetMetrics(pred[i], \
                                                    targets_idx, topx)       
                
                # Get the number of inputs mentionned before prediction
                mentions = masks[0][i].sum(dtype=torch.uint8).item()
                
                # Add metric to appropriate MetricByMentions obj
                metrics['avrg_rank'].Add(avrg_rk, mentions)
                metrics['ndcg'].Add(ndcg, mentions)
                metrics['recall@1'].Add(re_1, mentions)
                metrics['recall@10'].Add(re_10, mentions)
                metrics['recall@50'].Add(re_50, mentions)

    
    metrics['eval_loss'] /= nb_batch   
    metrics['eval_loss'] = metrics['eval_loss'].item()  

    return metrics









"""

PRINTS

"""

def PrintResults(metrics, epoch, model, metrics_to_print=['ndcg']):
    """
    Printing different metrics for a specific epoch
    
    Parameters
    ----------
    metrics : dict of MetricByMentions obj
        All metrics evaluated for an epoch.
    epoch : int
        The epoch concern with those results.
    model : torch.nn.Module
        The model we are training.
    metrics_to_print : list of str, optional
        List of metrics that will be printed. Correspond to keys from the metrics 
        The default is ['ndcg'].

    Returns
    -------
    None.
    """
    
    # Print epoch and losses when training
    if 'train_loss' in metrics:
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

OTHERS

"""


def SaveExperiment(args):
    
    # Path where to save Experiments.json: At top of ReDial folder. 
    # (It differs on personal computer and Compute Canada, see below) 
    # (Can't use args.path_to_ReDial because $SLURM_TMPDIR)
    path_to_Exp = '../'      # Exp are launch in ED or BERT or ... so go up 
    
    # Set args.id with nb of secs since Epoch GMT time + args.a_comment
    # If args.pred_only, add a mention about it
    if args.pred_only:
        args.id = str(int(time.time())) + "__PRED_ONLY" + args.a_comment
    else:
        args.id = str(int(time.time())) + "__" + args.a_comment
    
    # Adapt args.id to add GPU when trained on it
    if args.DEVICE == 'cuda':
        args.id += '_GPU'
        path_to_Exp = ''      # Exp are lunch in ReDial
    
    # Load Experiement.json
    with open(path_to_Exp + 'Experiments.json', 'r') as fp:
        exp = json.load(fp)
    # Add this experiment
    exp[args.id] = args.__dict__
    # Save Experiement.json
    with open(path_to_Exp + 'Experiments.json', 'w') as fp:
        json.dump(exp, fp, indent=4, sort_keys=True)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






