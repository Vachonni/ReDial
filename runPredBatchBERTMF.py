#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:53 2019



For a single model loaded once, get the predictions for a user. 

A user corresponds to a .csv file and turns it into a databunch. This .csv file is formated for fast_bert and
texts corresponds to one users and all possible items.

One file is treated by RankUser, a method of learner.   

Results are MetricsByMentions objects, saved in same Result folder were the 
model came from.




        ***    So big, default is for Test files    ***
        



----
BERT adapted for recommendation

From FAST-BERT example at: 
    https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384
    Up to date code is in: fast_bert-1.4.2.tar.gz
        
    When using Bert for Recommendation:
        use ONE label column in the data with header 'ratings' and 
        label_col = ['ratings']. 
        'ratings' column shoul have all same number of 
        examples (fill with (-1, 0) if necessary).
    
    
    


@author: nicholas

"""

import time
import json
from pathlib import Path
import random
import numpy as np
import torch
# import apex
import logging


from BERTMF.data_reco_MF import BertDataBunch
from BERTMF.learner_reco_MF import BertLearner      
from torch.utils.tensorboard import SummaryWriter    
from Objects.MetricByMentions import ToTensorboard, MetricByMentions


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()





#######################
###                 ###
###       ARGS      ###
###                 ###
#######################

import argparse

parser = argparse.ArgumentParser(description='Bert for recommendation')

parser.add_argument('--a_comment',type=str, metavar='', default='',\
                    help='Comment to add to name of Results and Tensorboards')
    
    
parser.add_argument('--log_path', type=str, metavar='', default='.',\
                    help='Path where all infos will be saved.')
parser.add_argument('--data_path', type=str, metavar='', default='./Data/BERTMF/Test/', \
                    help='Path to datasets')
parser.add_argument('--dataPred', type=str, metavar='', default='Test.csv', \
                    help='File to make predictions on')    
    
    
parser.add_argument('--user_start', type=int, metavar='', default=0, \
                    help='First user treated (i.e. files loaded)')
parser.add_argument('--user_stop', type=int, metavar='', default=6925, \
                    help='Last user treated (i.e. files loaded)')    
    
    
parser.add_argument('--pre_model', type=str, metavar='', default=None, \
                    help='Folder where all files for model are saved')    
  
    
parser.add_argument('--DEVICE', type=str, metavar='', default='cuda', \
                    help='cuda ou cpu')

    
args = parser.parse_args()




#######################
###                 ###
###  EXPERIENCE_ID  ###
###                 ###
#######################


# Set args.id with nb of secs since Epoch GMT time + args.a_comment
exp_id = str(int(time.time())) + "_BERT_" + args.a_comment

# Load Experiement.json
with open('Experiments.json', 'r') as fp:
    exp = json.load(fp)

# Add this experiment
exp[exp_id] = args.__dict__
# Save Experiement.json
with open('Experiments.json', 'w') as fp:
    json.dump(exp, fp, indent=4, sort_keys=True)  
    



######################
###                ###
###      SEED      ###
###                ###
######################

manualSeed = 1
# Python
random.seed(manualSeed)
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
            
            


######################
###                ###
###      DATA      ###
###                ###
######################




LABEL_PATH = Path(args.data_path)    # path for labels file
MODEL_PATH = Path(args.log_path)     # path for model artifacts to be stored
LOG_PATH = Path(args.log_path)       # path for log files to be stored

# Insure MODEL_PATH and LOG_PATH exit
MODEL_PATH.mkdir(exist_ok=True)

# Identify metrics to track
metrics = ['ndcg', 'recall@1', 'recall@10', 'recall@50']

# Init variable to store the results (dict of metric:MetricsByMentions)
results = {}           
for m in metrics:
    results[m] = MetricByMentions(m)


# Treat users one by one
for u in range(args.user_start, args.user_stop):
    
    user_folder = 'User' + str(u)
    DATA_PATH = Path(args.data_path, 'Test', user_folder)     # path for data files 
    
    logger.info('\n Creating databunch of {}', user_folder)
    
    databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                              tokenizer='bert-base-uncased',
                              train_file='',                  # No pre-process, no dl
                              val_file=args.dataPred,
                              label_file='labels.csv',
                              text_col='text',
                              label_col=['ratings'],
                              batch_size_per_gpu=8,
                              max_seq_length=512,
                              multi_gpu=True,
                              multi_label=True,
                              model_type='bert',
                              clear_cache=False,
                              no_cache=False)




######################
###                ###
###    LEARNER     ###
###                ###
######################


    device_cuda = torch.device(args.DEVICE)

    
    logger.info('\n Creating learner')
    
    # Set pretrained model to use.
    model_to_start = Path(args.log_path, 'Results', args.pre_model)
    
    learner = BertLearner.from_pretrained_model(
    						databunch,
    						pretrained_path=model_to_start,
    						metrics=metrics,
    						device=device_cuda,
    						logger=logger,
    						output_dir=MODEL_PATH,
    						finetuned_wgts_path=None,
    						warmup_steps=500,
    						multi_gpu=True,
    						is_fp16=False,
    						multi_label=True,
    						logging_steps=0)

    # Add experience id argument to the learner instance
    learner.exp_id = exp_id




######################
###                ###
###     PRED       ###
###                ###
######################
     

     
    # Create SummaryWriter for Tensorboard       
    tensorboard_dir = Path(args.log_path, 'runs', exp_id)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)    
    tb_writer = SummaryWriter(tensorboard_dir)
        
    # Get user's metrics    
    results_this_user = learner.RankUser()
    
    # For each metric, combine resuls of this user all users
    for metric_name, metric_by_mentions in results_this_user.items():
        results[metric_name].Combine(metric_by_mentions)
    
    # # Add results to tensorboard
    # ToTensorboard(tb_writer, results, 0, learner.model, metrics)
    # for key, value in results.items():
    #     if key == 'train_loss' or key == 'eval_loss': continue
    #     logger.info("{} : {}: ".format(key, value.Avrg()))




######################
###                ###
###     SAVE       ###
###                ###
######################


name_file = 'MetricsByMentions_' + args.user_start + '_' + args.user_stop + '.pth'
torch.save(results, Path(model_to_start, name_file))





































