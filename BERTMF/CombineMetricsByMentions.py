#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:45:22 2020



File to combine metrics_by_mentions from different files



@author: nicholas
"""



import os
import torch
from torch.utils.tensorboard import SummaryWriter    
from pathlib import Path
import argparse


from Objects.MetricByMentions import ToTensorboard




parser = argparse.ArgumentParser(description='Combine MetricsByMentions')

parser.add_argument('--model_id',type=str, metavar='', default='',\
                    help='model_id')
parser.add_argument('--epoch',type=str, metavar='', default='',\
                    help='epoch of the model_id we combine results for')    
       
args = parser.parse_args()




# Get the files

# Path to the model. Need to update model_id and epoch
path = Path('/Users/nicholas/ReDial/Results', args.model_id, args.epoch)

files = []
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)) and 'MetricsByMentions_' in i:
        files.append(i)

print(files)
        
        
        
        
# Init variable to store the results (dict of metric:MetricsByMentions)
results = {}           
        
    
# Treat all files
for file in files:
    
    # Load file
    results_this_file = torch.load(str(path)+'/'+file)
    
    # Update results...
    # ...if first file
    if results == {}:
        results = results_this_file
    # ...if not empty, combine every metric with global results
    for metric_name, metric_by_mentions in results_this_file.items():
        results[metric_name].Combine(metric_by_mentions)
        
    
        
# Print results
for key, value in results.items():
    print("Avrg {} = {}: ".format(key, value.Avrg()))

print('Recall@50 - Averages by mentions', results['recall@50'].AvrgByMentions())


# Results to tensorboard

# Create SummaryWriter for Tensorboard       
tensorboard_dir = Path('/Users/nicholas/ReDial/runs', args.model_id, args.epoch)
tensorboard_dir.mkdir(parents=True, exist_ok=True)    
tb_writer = SummaryWriter(tensorboard_dir)

metrics = list(results.keys())        
ToTensorboard(tb_writer, results, 0, None, metrics)
