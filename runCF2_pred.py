#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:15:28 2019


Basic Transformer Recommender - 
Predictions with models of type Ratings and Genres with Chronological Data


@author: nicholas
"""


########  IMPORTS  ########


import sys
import torch
import pandas as pd
import numpy as np

# Personnal imports
import CF2.Models as Models
import CF2.Utils as Utils



#######################
###                 ###
###       ARGS      ###
###                 ###
#######################

import argparse

parser = argparse.ArgumentParser(description='CF2 Prediction')
    

parser.add_argument('--data_path', type=str, metavar='', default='./Data/CF2/', \
                    help='Path to datasets')
parser.add_argument('--data_pred', type=str, metavar='', default='Test_new.json', \
                    help='File to make predictions on')      
parser.add_argument('--user_RT', type=str, metavar='', default='items_full_kb.pth', \
                    help='User RT')      
parser.add_argument('--item_RT', type=str, metavar='', default='users_nlg.pth', \
                    help='Item RT')     
    
parser.add_argument('--model_type', type=str, metavar='', default='learned', \
                    help='Learned or fixed latent representations')    
parser.add_argument('--model_name', type=str, metavar='', \
                    default='CF2_2BERTDot_40_15e5_b32_fullkb_AVRG_TestSet', \
                    help='Complete path to model')  
    
parser.add_argument('--DEVICE', type=str, metavar='', default='cuda', \
                    help='cuda ou cpu')
    
    
args = parser.parse_args()


         
def main():                        
                    
    
    
    ########################
    #                      # 
    #         INIT         #
    #                      # 
    ########################
    
    # Print agrs that will be used
    print(sys.argv)
    
    # Cuda availability check
    if args.DEVICE == "cuda" and not torch.cuda.is_available():
        raise ValueError("DEVICE specify a GPU computation but CUDA is not available")
      
    # Seed 
    manualSeed = 1
    # Python
  #  random.seed(manualSeed)
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
    
    # Global variable for runOrion.py (NDCGs for one model)
    NDCGs_1model = -1
    
    
    
    
    ########################
    #                      # 
    #        MODEL         #
    #                      # 
    ########################
    
    
    # Create basic model and load learned parameters
    print('\n******* Loading model *******') 
    
    if args.model_type == 'learned':
        model = Models.Train2BERT('Train2BERTDotProduct') 
    else:
        model = Models.MLPLarge()
  
    model = model.to(args.DEVICE)      
    
    model_path = './Results/Models/' + args.model_name + '/model.pth'   
    
    checkpoint = torch.load(model_path, map_location=args.DEVICE)
    
    model.load_state_dict(checkpoint['state_dict'])



    
    
    ########################
    #                      # 
    #         DATA         #
    #                      # 
    ########################    
    
    
    ######## LOAD DATA 
    
    
    print('\n******* Loading PRED samples from *******', args.data_path + args.data_pred)
    df_pred = pd.read_csv(args.data_path + args.data_pred)
    # Turn DataFrame into an numpy array (easier iteration)
    pred_data = df_pred.values
    
    print('\n******* Loading RT *******', args.data_path + args.item_RT)
    # LOAD RT - According to the model
    if args.model_type == 'learned':
        # Load Relational Tables (RT) of BERT ready inputs for users and items. Type: dict of torch.tensor.
        user_RT = np.load(args.data_path + 'RT/BERTInput/' + args.user_RT, allow_pickle=True).item()
        item_RT = np.load(args.data_path + 'RT/BERTInput/' + args.item_RT, allow_pickle=True).item()
    else:
        # Load Relational Tables (RT) of BERT_avrg for users and items. Type: torch.tensor.
        # map_location is CPU because Dataset with num_workers > 0 should not return CUDA.
        user_RT = torch.load(args.data_path + 'RT/PoolerEmbed/' + args.user_RT, map_location='cpu')
        item_RT = torch.load(args.data_path + 'RT/PoolerEmbed/' + args.item_RT, map_location='cpu')  
    

    if args.DEBUG: 
        pred_data = pred_data[:128]
    
        
    
    
    
    
    ##############################
    #                            # 
    #         PREDICTION         #
    #                            # 
    ##############################    
    
          
    # Make predictions (returns dictionaries)
    print("\n\nPrediction Chronological...")
    avrg_rank, MRR, RR, RE_1, RE_10, RE_50, NDCG = \
            Utils.Prediction(pred_data, model, user_RT, item_RT, \
                             100, 'min', \
                             args.DEVICE, 100)   
    
    # Print results
    print("\n\n\n\n  ====> RESULTS <==== ")
    print("\n  ==> By qt_of_movies_mentioned, on to be mentionned movies <==\n")
        
    # List of metrics to evaluate and graph
    #   Possible values: avrg_rank, MRR, RR, RE_1, RE_10, RE_50, NDCG 
    graphs_data = [avrg_rank, RE_1, RE_10, RE_50, MRR, NDCG]  
    graphs_titles = ['AVRG_RANK', 'RE_1', 'RE_10', 'RE_50', 'MRR', 'NDCG'] 
    
    # Evaluate + graph
    for i in range(len(graphs_titles)):
        avrgs = Utils.ChronoPlot(graphs_data[i], graphs_titles[i], args.logInfosPATH, '_'+args.trial_id)
        if graphs_titles[i] == 'NDCG':
            NDCGs_1model = avrgs
    
    return NDCGs_1model 


#%%
    
if __name__ == '__main__':
    main()



























































