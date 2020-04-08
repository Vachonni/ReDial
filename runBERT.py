#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:53 2019



BERT adapted for recommendation


From FAST-BERT example at: 
    https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384
    Up to date code is in: fast_bert-1.4.2.tar.gz
    
    
    When using Bert for Recommendation:
        use ONE label column in the data with header 'ratings' and 
        label_col = ['ratings']. 
        'ratings' column shoul have all same number of 
        examples (fill with (-1, 0) if necessary).
    
    
    
Call examples for Prediction on trained model:
     python runBERT.py --pre_model 1585326608_BERT_Next_NL_lr24e-4 --data_path ./Data/BERT_no_empty_input/Next/TextNLGenres/ --dataPred Test.csv --a_comment PREDONLY_Test_NL_24_NO_EMPTY



@author: nicholas

"""

import time
import json
from pathlib import Path
import random
import numpy as np
import torch
# import apex





import argparse

parser = argparse.ArgumentParser(description='Bert for recommendation')

parser.add_argument('--a_comment',type=str, metavar='', default='',\
                    help='Comment to add to name of Results and Tensorboards')
    
parser.add_argument('--log_path', type=str, metavar='', default='.',\
                    help='Path where all infos will be saved.')
parser.add_argument('--data_path', type=str, metavar='', default='./Data/BERT/Next/', \
                    help='Path to datasets')
parser.add_argument('--dataPred', type=str, metavar='', default='Val.csv', \
                    help='File to make predictions on')    

parser.add_argument('--pre_model', type=str, metavar='', default=None, \
                    help='Folder where all files for model are saved')    
parser.add_argument('--epoch', type=int, metavar='', default=1, \
                    help='Qt of epoch')
parser.add_argument('--lr', type=float, metavar='', default=6e-5*4, \
                    help='Initial learning rate')

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

from BERT.data_reco import BertDataBunch


DATA_PATH = Path(args.data_path)     # path for data files (train and val)
LABEL_PATH = Path(args.data_path)    # path for labels file
MODEL_PATH = Path(args.log_path)     # path for model artifacts to be stored
LOG_PATH = Path(args.log_path)       # path for log files to be stored

# Insure MODEL_PATH and LOG_PATH exit
MODEL_PATH.mkdir(exist_ok=True)


databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='Train.csv',
                          val_file=args.dataPred,
                          label_file='labels.csv',
                          text_col='text',
                          label_col=['ratings'],
                          batch_size_per_gpu=8,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=True,
                          model_type='bert',
                          clear_cache=True,
                          no_cache=True)




######################
###                ###
###    LEARNER     ###
###                ###
######################


from BERT.learner_reco import BertLearner


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
#logger.setLevel('INFO')

logger.info('will my logger print?')

device_cuda = torch.device(args.DEVICE)
metrics = ['ndcg', 'recall@1', 'recall@10', 'recall@50']

print('hello')


# Set pretrained model to use.
# If nothing specified, use 'bert-base-uncased'
if args.pre_model == None: model_to_start = 'bert-base-uncased'
else: model_to_start = Path(args.log_path, 'Results', args.pre_model)


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


#%%

######################
###                ###
###     TRAIN      ###
###                ###
######################

print('hello again')

# If no fine-tuned model to start with, train
if args.pre_model == None:
    
    learner.fit(epochs=args.epoch,
    			lr=args.lr,
    			validate=True,        	# Evaluate the model after each epoch
    			schedule_type="warmup_cosine",
    			optimizer_type="lamb")


#%%

######################
###                ###
###     PRED       ###
###                ###
######################


# If we have a fine_tuned model to start with, pred
elif args.pre_model != None:
    
    from torch.utils.tensorboard import SummaryWriter    
    from Objects.MetricByMentions import ToTensorboard
     
    # Create SummaryWriter for Tensorboard       
    tensorboard_dir = Path(args.log_path, 'runs', exp_id)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)    
    tb_writer = SummaryWriter(tensorboard_dir)
        
    # Get results    
    results = learner.validate()
    
    # Add results to tensorboard
    ToTensorboard(tb_writer, results, 0, learner.model, metrics)
    for key, value in results.items():
        if key == 'train_loss' or key == 'eval_loss': continue
        logger.info("{} : {}: ".format(key, value.Avrg()))











































