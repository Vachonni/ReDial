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
parser.add_argument('--epoch', type=int, metavar='', default=30, \
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
                          val_file='Val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=['ratings'],
                          batch_size_per_gpu=8,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=True,
                          model_type='bert')




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

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
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

learner.fit(epochs=args.epoch,
			lr=args.lr,
			validate=True,        	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")


#%%

######################
###                ###
###     SAVE       ###
###                ###
######################


# learner.save_model()


#%%


texts = [
        'I really love the Netflix original movies',
		 'Jerk me jolly. I have a big penis, not to mention the species is thriving.',
         'People watching Netflix movies should die',
         'You are a big hairy ape like mamith.'
         ]
predictions = learner.predict_batch(texts)


for i in range(len(predictions)):
    print('\n\n',texts[i],'\n', predictions[i][:5])










































