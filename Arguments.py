#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed March 11 13:36:45 2020


List of argumnents usable with parser 


@author: nicholas
"""


import argparse


parser = argparse.ArgumentParser(description='Train an EncoderDecoder Recommender and Predict')


# parser.add_argument('--id', type=str, metavar='', default='test', help='ID of experience. Will be used when saving file.')



# Path
parser.add_argument('--path_to_ReDial', type=str, metavar='', default=None, \
                    help='Path to ReDial folder. See PATH MANAGEMENT below')
    
    
# Data
parser.add_argument('--dataPATH', type=str, metavar='', default='/Data/EncoderDecoder', \
                    help='Path from path_to_ReDial to datasets to train on')
parser.add_argument('--dataTrain', type=str, metavar='', default='ED_next_train.json', \
                    help='File name of Dataset to train on')
parser.add_argument('--dataValid', type=str, metavar='', default='ED_next_valid.json', \
                    help='File name of Dataset to for validation')
parser.add_argument('--exclude_genres', default=False, action='store_true', \
                    help='If arg added, genres not used in input (Dataset part empty for genres)')


    
# Training
parser.add_argument('--lr', type=float, metavar='', default=0.001, help='Learning rate')
parser.add_argument('--batch', type=int, metavar='', default=64, help='Batch size')
parser.add_argument('--epoch', type=int, metavar='', default=100, help='Number of epoch')
parser.add_argument('--loss_fct', type=str, metavar='', default='BCE', \
                    choices=['BCEWLL', 'BCE'], help='Loss function')
parser.add_argument('--patience', type=int, metavar='', default=5, \
                    help='number of epoch to wait without improvement in valid_loss before ending training')
parser.add_argument('--completionTrain', type=float, metavar='', default=100, \
                    help='% of data used during 1 training epoch')
parser.add_argument('--completionEval', type=float, metavar='', default=100, \
                    help='% of data used for evaluation')
parser.add_argument('--EARLY', default=False, action='store_true', \
                    help="If arg added, Train at 10% and Eval at 10%")



# Model
parser.add_argument('--layer1', type=int, metavar='', default=323, \
                    help='Integers corresponding to the first hidden layer size')
parser.add_argument('--layer2', type=int, metavar='', default=0, \
                    help='Integers corresponding to the second hidden layer size. 0 if none.')
parser.add_argument('--activations', type=str, metavar='', default='relu', \
                    choices=['relu', 'sigmoid'],\
                    help='Activations in hidden layers of the model')
parser.add_argument('--last_layer_activation', type=str, metavar='', default='softmax', \
                    choices=['none', 'sigmoid', 'softmax'],\
                    help='Last layer activation of the model')    
parser.add_argument('--g_type', type=str, metavar='', default='none', \
                    choices=['none', 'fixed', 'one', 'genres', 'unit'], \
                    help="Parameter(s) learned for genres inputs. None: no genres, Fixed: no learning, \
                    One: one global parameter, Genres: one parameter by genres, Unit:one parameter per movie,...")
parser.add_argument('--preModel', type=str, metavar='', default='none', \
                    help='Path from path_to_ReDial to a pre-trained model to start with. \
                    Should include a GenresWrapper of same type')
                    
                    
                    
# ...for Pred file
parser.add_argument('--completionPredChrono', type=float, metavar='', default=100, \
                    help='% of data used for prediction')
parser.add_argument('--M1_path', type=str, metavar='', default='none', \
                    help='Path from path_to_ReDial to a Model 1. Will pred with + without genres')
parser.add_argument('--M1_label', type=str, metavar='', default='none', \
                    help='Label for Model 1')
parser.add_argument('--M2_path', type=str, metavar='', default='none', \
                    help='Path from path_to_ReDial to a Model 2')
parser.add_argument('--M2_label', type=str, metavar='', default='none', \
                    help='Label to a Model 2')


    
# Genres 
parser.add_argument('--genresDict', type=str, metavar='', default='genres_inter_IDX_2_ReD_or_id.json', \
                    help='File name of Dict of genres')
parser.add_argument('--top_cut', type=int, metavar='', default=100, \
                    help='number of movies in genres vector (for torch Dataset)')


    
# Metrics
parser.add_argument('--topx', type=int, metavar='', default=0, \
                    help='for NDCG mesure, size of top ranks considered. \
                         If 0, we consider all the values ranked')


    
# Others
parser.add_argument('--no_seed', default=False, action='store_true', \
                    help="If arg added, random always give the same")

parser.add_argument('--DEVICE', type=str, metavar='', default='cpu', choices=['cuda', 'cpu'], \
                    help="Type of machine to run on")

parser.add_argument('--DEBUG', default=False, action='store_true', \
                    help="If arg added, reduced dataset and epoch to 1 for rapid debug purposes")



args = parser.parse_args()



# PATH MANAGEMENT FOR SAVING AND LOADING
if args.path_to_ReDial == None:
    print('\n\n\n\n       *** No path_to_ReDial specified as args *** ')
    from pathlib import Path
    import sys
    path = Path(sys.executable)
    # If using cpu, assume at root of user's machine
    if args.DEVICE == 'cpu':
        args.path_to_ReDial = str(path.home()) + '/ReDial'
    # If not, assume Compute Canada, hence in scratch
    else:
        args.path_to_ReDial = str(path.home()) + '/scratch/ReDial'
    print(f'       *** path_to_ReDial establish as {args.path_to_ReDial} *** \n\n\n\n')

    

# ASSERTION
if args.loss_fct == 'BCE':
    assert args.last_layer_activation != 'none','Need last layer activation with BCE'
if args.loss_fct == 'BCEWLL':
    assert args.last_layer_activation == 'none',"Last layer activation must be 'none' with BCEWLL"

assert 0 <= args.completionTrain <=100,'completionTrain should be in [0,100]'
assert 0 <= args.completionEval <=100,'completionPred should be in [0,100]'


# CONVERSION
# (bunch of hyper-parameters group under a name for efficiency when running)
if args.EARLY:
    args.completionTrain = 10 
    args.completionEval = 50
    
    