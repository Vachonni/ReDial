#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:02:56 2019


File to see impact of popularity on predictions.

Hence, we change the model to output Softmax(Popularity) for all inputs



Call examples for Prediction on trained model:
    LOCAL: --pred_only --a_comment PRED_POPULARITY_Test --dataValid Test.json --dataPATH /Data/ED/Next 
    CC: python runED.py --pred_only --pre_model 1585927941_ED_Next_genres_3EPOCH.pth --dataPATH /Data/ED/Next --dataValid Test.json --a_comment PREDONLY_Test_Genres_3EPOCH --DEVICE cuda


@author: nicholas
"""




########################
#                      # 
#       IMPORTS        #
#                      # 
########################

import json
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from collections import defaultdict

# Personnal imports
import ED.Utils as Utils
import Settings 
from ED.Arguments import args 
from Objects.MetricByMentions import ToTensorboard




########################
#                      # 
#         INIT         #
#                      # 
########################

# Save experiement's info
Utils.SaveExperiment(args)
# Print args
print(args, '\n')

#TODO: Keep this line?
nb_movies = Settings.nb_movies_ReDial


# Cuda availability check
if args.DEVICE == "cuda" and not torch.cuda.is_available():
    raise ValueError("DEVICE specify a GPU computation but CUDA is not available")
  
# Seed 
if not args.no_seed:
    manualSeed = 1
    # # Python
    # random.seed(manualSeed)
    # Numpy
    np.random.seed(manualSeed)
    # Torch
    torch.manual_seed(manualSeed)
    # Torch with GPU
    if args.DEVICE == "cuda":
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# Create Tensorboard instance 
Path(args.path_to_ReDial + '/runs').mkdir(parents=True, exist_ok=True)
tb = SummaryWriter(log_dir=args.path_to_ReDial + '/runs/' + args.id)





########################
#                      # 
#         MODEL        #
#                      # 
########################
  


class PopModel(torch.nn.Module):
    """
    A model that always output popularity softmaxed
    """

    def __init__(self):
        super(PopModel, self).__init__()
        pass
       
        
    def forward(self, x):
        
        batch_size = len(x[0])          # Quantity of input vectors (1 is genre idx, 2 is genres vector)
        
        pop_softmax = torch.nn.Softmax(dim=0)(Settings.popularity)
        
        pop_resize = pop_softmax.repeat(batch_size, 1)
        
        return pop_resize



model = PopModel()
    

print('******* Loading GENRES dict from *******', args.genres_dict)
# Format {"['genres']": [ idx, [movies ReD_or_id INTERSECTION of genres from key] ]}    
genres_Inter = json.load(open(args.genres_dict))    

    
# CRITERION
if args.loss_fct == 'BCEWLL':
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
elif args.loss_fct == 'BCE':
    criterion = torch.nn.BCELoss(reduction='none')








########################
#                      # 
#         DATA         #
#                      # 
########################


# LOAD DATA 
#TODO:  ADJUST   R (ratings) - Format [ [UserID, [movies uID], [ratings 0-1]] ]   
print('******* Loading SAMPLES from *******', \
      args.path_to_ReDial+args.dataPATH+'/'+args.dataTrain)

if not args.pred_only:    
    train_data = json.load(open(args.path_to_ReDial+args.dataPATH+'/'+args.dataTrain))
valid_data = json.load(open(args.path_to_ReDial+args.dataPATH+'/'+args.dataValid))
# TODO: STILL USEFULL? Use only samples where there is a genres mention
valid_g_data = [[c,m,g,tbm] for c,m,g,tbm in valid_data if g != []]
if args.DEBUG: 
    train_data = train_data[:128]
    valid_data = valid_data[:128]



######## CREATING DATASET ListRatingDataset 
print('******* Creating torch datasets *******')
if not args.pred_only:
    train_dataset = Utils.RnGChronoDataset(train_data, genres_Inter, \
                                           nb_movies, Settings.popularity, args.DEVICE, \
                                           args.exclude_genres, args.no_popularity, 
                                           args.top_cut)
valid_dataset = Utils.RnGChronoDataset(valid_data, genres_Inter, \
                                       nb_movies, Settings.popularity, args.DEVICE,  \
                                       args.exclude_genres, args.no_popularity, \
                                       args.top_cut)
    

######## CREATE DATALOADER
print('******* Creating dataloaders *******\n\n')    
kwargs = {}
if(args.DEVICE == "cuda"):
    kwargs = {'num_workers': 0, 'pin_memory': False}
if not args.pred_only:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,\
                                               shuffle=True, drop_last=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch,\
                                           shuffle=True, drop_last=False, **kwargs)    
# TODO: STILL USEFULL? For PredRaw - Loader of only 1 sample (user) 
valid_bs1_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, **kwargs)
##TODO: For PredChrono
#valid_chrono_loader = torch.utils.data.DataLoader(valid_chrono_dataset, batch_size=args.batch, shuffle=True, **kwargs)    








########################
#                      # 
#        TRAIN         #
#                      # 
########################


metrics_by_epoch = defaultdict(list) 

# For saving and patience
best_ndcg = 0
last_improvement = 0 


if args.DEBUG: args.epoch = 1
for epoch in range(args.epoch):

    print('\n\n\n\n     ==> Epoch:', epoch, '\n')
    
    # Train 
    if not args.pred_only:
        train_loss = Utils.Train(train_loader, model, criterion, optimizer, \
                                 args.completionTrain)
    
    # Evaluate
    metrics = Utils.Eval(valid_loader, model, criterion, args.completionEval, args.topx)
    
    if not args.pred_only:
        # Put train_loss with all the metrics together
        metrics['train_loss'] = train_loss
    
    # Print results 
    Utils.PrintResults(metrics, epoch, model, ['ndcg', 'recall@1', 'recall@10', 'recall@50'])
    
    # Add results to Tensorboard 
    ToTensorboard(tb, metrics, epoch, model, ['ndcg', 'recall@1', 'recall@10', 'recall@50'])
    
    
    # SAVING AND PATIENCE
    if not args.pred_only:
        # Keep track of metrics by epoch
        metrics_by_epoch['train_loss'].append(metrics['train_loss'])
        metrics_by_epoch['eval_loss'].append(metrics['eval_loss'])
        metrics_by_epoch['avrg_ndcg'].append(metrics['ndcg'].Avrg())    
        
        # If model improves
        if metrics['ndcg'].Avrg() > best_ndcg:
            best_ndcg = metrics['ndcg'].Avrg()
            last_improvement = 0
            # Save model
            print('\n\n   Saving...')
            state = {
                    'epoch': epoch,
                    'metrics_by_epoch': metrics_by_epoch,
                    'metrics': metrics,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'layers': args.layers,
                    'activations': args.activations,
                    'last_layer_activation': args.last_layer_activation,
                    'loss_fct': args.loss_fct,
                    'g_type': args.g_type
                    }
            Path(args.path_to_ReDial + '/Results').mkdir(parents=True, exist_ok=True)
            torch.save(state, args.path_to_ReDial+'/Results/'+args.id+'.pth')
            print('......saved.')
        
        # Stop if reached patience    
        elif last_improvement >= args.patience:
            print('--------------------------------------------------------------------------------')
            print('-                               STOPPED TRAINING                               -')
            print('--------------------------------------------------------------------------------')
            break
        
        else:
            last_improvement += 1

    





















































    
    







































