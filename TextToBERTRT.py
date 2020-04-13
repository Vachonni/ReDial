#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 07:45:39 2019


Turning users' and items' text representation into BERT...:
    - Inputs
    - pooler embeddings


@author: nicholas
"""



import torch
import json
from transformers import BertForSequenceClassification #, BertConfig

from Objects.BERTPreProcessor import BERTPreProcessor 
from Settings import ReD_id_2_ReD_or_id



# Variables according to local or Compute Canada
if torch.cuda.is_available():
    DEVICE = 'cuda'
    path_to_ReDial = '/home/vachonni/scratch/ReDial'
else:
    DEVICE = 'cpu'
    path_to_ReDial = '/Users/nicholas/ReDial' 
print('DEVICE = ', DEVICE )



# LOADING


# KB_users
with open(path_to_ReDial + '/Data/PreProcessed/KB_users.json', 'r') as fp:
    KB_users = json.load(fp)


# KB_items
with open(path_to_ReDial + '/Data/PreProcessed/KB_IMDB_movies.json', 'r') as fp:
    KB_items = json.load(fp)


# BERT models
# # First, load config and adjust it, because we'll need to output hidden layers, 
# # which is not the default case for BertForSequenceClassification
# config = BertConfig.from_pretrained( \
#                 path_to_ReDial + '/Results/1586382856_BERT_Next_NL_G_lr5e-4', \
#                 output_hidden_states = True)
# Load BERT for users


# BERT_users = BertForSequenceClassification.from_pretrained( \
#                 path_to_ReDial + '/Results/1586382856_BERT_Next_NL_G_lr5e-4') #, \
# #                config=config)
# BERT_users.to(DEVICE)
# BERT_users.eval()


# Load BERT models for items
BERT_items_full_kb = BertForSequenceClassification.from_pretrained( \
                path_to_ReDial + '/Results/1586650446_BERT_Items_FullKB_lr5e-4_160epoch') #, \
#                config=config)
BERT_items_full_kb.to(DEVICE)
BERT_items_full_kb.eval()
BERT_items_tga = BertForSequenceClassification.from_pretrained( \
                path_to_ReDial + '/Results/1586656383_BERT_Items_TGA_lr5e-4_160epoch') #, \
#                config=config)
BERT_items_tga.to(DEVICE)
BERT_items_tga.eval()
BERT_items_title = BertForSequenceClassification.from_pretrained( \
                path_to_ReDial + '/Results/1586699552_BERT_Items_Title_lr5e-4_160epoch') #, \
#                config=config)
BERT_items_title.to(DEVICE)
BERT_items_title.eval()
print('Models loaded')




if __name__ == '__main__':
    
    
    # USERS   
    
    
    # # Get the BERT processor for users    
    # bert_user_prepro = BERTPreProcessor(BERT_users)
    
    # # Inputs. Init dict of dict of torch tensors
    # users_raw_inputs = {}
    # users_nl_inputs = {}
    # users_nlg_inputs = {}
    
    # # Embeddings. Init torch tensors
    # qt_users = len(KB_users)
    # users_raw_embed = torch.empty(qt_users,768)
    # users_nl_embed = torch.empty(qt_users,768)
    # users_nlg_embed = torch.empty(qt_users,768)
    
    
    # # Treat all users
    # for user_id, user_id_values in KB_users.items():
        
    #     # print update
    #     if int(user_id) % 1000 == 0: print(f'Treating user {user_id}')
        
    #     # Into BERT inputs
    #     users_raw_inputs[user_id] = \
    #         bert_user_prepro.TextToBERTInp1Speaker(user_id_values['text_raw'])
    #     users_nl_inputs[user_id] = \
    #         bert_user_prepro.TextToBERTInp1Speaker(user_id_values['text_nl'])
    #     users_nlg_inputs[user_id] = \
    #         bert_user_prepro.TextToBERTInp1Speaker(user_id_values['text_nlg'])

    #     # Into BERT pooler embeddings
    #     users_raw_embed[int(user_id)] = \
    #         bert_user_prepro.TextToBERTPooler(user_id_values['text_raw'])
    #     users_nl_embed[int(user_id)] = \
    #         bert_user_prepro.TextToBERTPooler(user_id_values['text_nl'])
    #     users_nlg_embed[int(user_id)] = \
    #         bert_user_prepro.TextToBERTPooler(user_id_values['text_nlg'])

    
    # # save
    # torch.save(users_raw_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/users_raw.pth')
    # torch.save(users_nl_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/users_nl.pth')
    # torch.save(users_nlg_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/users_nlg.pth')
    # torch.save(users_raw_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/users_raw.pth')
    # torch.save(users_nl_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/users_nl.pth')
    # torch.save(users_nlg_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/users_nlg.pth')
        
    

    
    
    # ITEMS   
    
    
    # Get the BERT processor for users    
    bert_item_kb_prepro = BERTPreProcessor(BERT_items_full_kb)
    bert_item_tga_prepro = BERTPreProcessor(BERT_items_tga)
    bert_item_title_prepro = BERTPreProcessor(BERT_items_title)
    
    # Inputs. Init dict of dict of torch tensors
    items_full_kb_inputs = {}
    items_tga_inputs = {}
    items_title_inputs = {}
    
    # Embeddings. Init torch tensors
    qt_items = len(KB_items)
    items_full_kb_embed = torch.empty(qt_items,768)
    items_tga_embed = torch.empty(qt_items,768)
    items_title_embed = torch.empty(qt_items,768)
    
    
    # Treat all items
    for ReD_id, item_id_values in KB_items.items():
        
        # From ReD_id to ReD_or_id
        ReD_or_id = ReD_id_2_ReD_or_id[ReD_id]
        
        # print update
        if ReD_or_id % 1000 == 0: print(f'Treating user {ReD_or_id}')
        
        tga = item_id_values['title'] + \
              '. Genres: ' + str(item_id_values['genres']) + \
              '. Actors: ' + str(item_id_values['actors'])
        
        # Into BERT inputs
        items_full_kb_inputs[str(ReD_or_id)] = \
            bert_item_kb_prepro.TextToBERTInp1Speaker(str(item_id_values))
        items_tga_inputs[str(ReD_or_id)] = \
            bert_item_tga_prepro.TextToBERTInp1Speaker(tga)
        items_title_inputs[str(ReD_or_id)] = \
            bert_item_title_prepro.TextToBERTInp1Speaker(item_id_values['title'])

        # Into BERT pooler embeddings
        items_full_kb_embed[ReD_or_id] = \
            bert_item_kb_prepro.TextToBERTPooler(str(item_id_values))
        items_tga_embed[ReD_or_id] = \
            bert_item_tga_prepro.TextToBERTPooler(tga)
        items_title_embed[ReD_or_id] = \
            bert_item_title_prepro.TextToBERTPooler(item_id_values['title'])

    
    # save
    torch.save(items_full_kb_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/items_full_kb.pth')
    torch.save(items_tga_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/items_tga.pth')
    torch.save(items_title_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/items_title.pth')
    torch.save(items_full_kb_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/items_full_kb.pth')
    torch.save(items_tga_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/items_tga.pth')
    torch.save(items_title_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/items_title.pth')
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    