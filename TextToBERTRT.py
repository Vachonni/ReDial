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
from transformers import BertForSequenceClassification 

from BERT.data_reco import BertDataBunch
from Settings import ReD_id_2_ReD_or_id




# Set variables according to local or Compute Canada
if torch.cuda.is_available():
    DEVICE = 'cuda'
    path_to_ReDial = '/home/vachonni/scratch/ReDial'
else:
    DEVICE = 'cpu'
    path_to_ReDial = '/Users/nicholas/ReDial' 
print('DEVICE = ', DEVICE )



######################
###                ###
###     Data       ###
###                ###
######################

# Load knowledge bases

# KB_users
with open(path_to_ReDial + '/Data/PreProcessed/KB_users.json', 'r') as fp:
    KB_users = json.load(fp)

# KB_items
with open(path_to_ReDial + '/Data/PreProcessed/KB_IMDB_movies.json', 'r') as fp:
    KB_items = json.load(fp)



######################
###                ###
###   DataBunch    ###
###                ###
######################

###        ****    WARNING    ****
### THESE PATHS WON'T BE USED FOR DATA INPUT (but needed in databunch creation)
### Only texts from KB will be used.
DATA_PATH = path_to_ReDial + '/Data/BERT/Next/TextRaw/'     # path for data files (train and val)
LABEL_PATH = path_to_ReDial + '/Data/BERT/Next/TextRaw/'    # path for labels file


databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='Train.csv',   
                          val_file='Val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=['ratings'],
                          batch_size_per_gpu=1,
                          max_seq_length=512,
                          multi_gpu=False,
                          multi_label=True,
                          model_type='bert',
                          clear_cache=True,
                          no_cache=True)



######################
###                ###
###   Function     ###
###                ###
######################


def TextToBERTRT(model, databunch, text):
    """
    From a string 
    To a BERT input as vector and 
    pooler output of a BertForSequenceClassification model.       
    """

    # Get the dataloader for this text
    dataloader = databunch.get_dl_from_texts([text])

    # Get the batch from this dataloader
    batch = next(iter(dataloader))
    
    # Turn the batch into proper BERT input        
    with torch.no_grad():
        input_to_bert = {'input_ids':      batch[0],
                         'attention_mask': batch[1],
                         'token_type_ids': batch[2]}

    # Put this input into BERT for classification model and get pooler output
    pooler_output = model.bert(**input_to_bert)[1].detach()


    return input_to_bert, pooler_output








######################
###                ###
###      Main      ###
###                ###
######################

if __name__ == '__main__':
  
    
    
    # USERS   
    
    
    # Load BERT models for users    
    BERT_users_raw = BertForSequenceClassification.from_pretrained( \
                       path_to_ReDial + '/Results/1585354803_BERT_Next_Raw_lr96e-4') 
    BERT_users_raw.to(DEVICE)
    BERT_users_raw.eval()
    BERT_users_nl = BertForSequenceClassification.from_pretrained( \
                       path_to_ReDial + '/Results/1586295618_BERT_Next_NL_lr86e-4') 
    BERT_users_nl.to(DEVICE)
    BERT_users_nl.eval()    
    BERT_users_nlg = BertForSequenceClassification.from_pretrained( \
                       path_to_ReDial + '/Results/1586382856_BERT_Next_NL_G_lr5e-4') 
    BERT_users_nlg.to(DEVICE)
    BERT_users_nlg.eval()
    print('Bert for users models loaded')

    
    # Inputs. Init dict of dict of torch tensors
    users_raw_inputs = {}
    users_nl_inputs = {}
    users_nlg_inputs = {}
    
    # Embeddings. Init torch tensors
    qt_users = len(KB_users)
    users_raw_embed = torch.empty(qt_users,768)
    users_nl_embed = torch.empty(qt_users,768)
    users_nlg_embed = torch.empty(qt_users,768)
    
    
    # Treat all users
    for user_id, user_id_values in KB_users.items():
        
        # print update
        if int(user_id) % 1000 == 0: print(f'Treating user {user_id}')
        
        # String to BERT RT
        users_raw_inputs[user_id], users_raw_embed[int(user_id)] = \
            TextToBERTRT(BERT_users_raw, databunch, user_id_values['text_raw'])
        users_nl_inputs[user_id], users_nl_embed[int(user_id)] = \
            TextToBERTRT(BERT_users_nl, databunch, user_id_values['text_nl'])
        users_nlg_inputs[user_id], users_nlg_embed[int(user_id)] = \
            TextToBERTRT(BERT_users_nlg, databunch, user_id_values['text_nlg'])

    
    # save
    torch.save(users_raw_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/users_raw.pth')
    torch.save(users_nl_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/users_nl.pth')
    torch.save(users_nlg_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/users_nlg.pth')
    torch.save(users_raw_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/users_raw.pth')
    torch.save(users_nl_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/users_nl.pth')
    torch.save(users_nlg_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/users_nlg.pth')

    # del to free space
    del(BERT_users_raw)
    del(BERT_users_nl)
    del(BERT_users_nlg)


    
    
    # ITEMS   
    
    
    # Load BERT models for items
    BERT_items_full_kb = BertForSequenceClassification.from_pretrained( \
                    path_to_ReDial + '/Results/1586650446_BERT_Items_FullKB_lr5e-4_160epoch') 
    BERT_items_full_kb.to(DEVICE)
    BERT_items_full_kb.eval()
    BERT_items_tga = BertForSequenceClassification.from_pretrained( \
                    path_to_ReDial + '/Results/1586656383_BERT_Items_TGA_lr5e-4_160epoch')
    BERT_items_tga.to(DEVICE)
    BERT_items_tga.eval()
    BERT_items_title = BertForSequenceClassification.from_pretrained( \
                    path_to_ReDial + '/Results/1586699552_BERT_Items_Title_lr5e-4_160epoch') 
    BERT_items_title.to(DEVICE)
    BERT_items_title.eval()
    print('Bert for items models loaded')
        
    

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
        items_full_kb_inputs[str(ReD_or_id)], items_full_kb_embed[ReD_or_id] = \
            TextToBERTRT(BERT_items_full_kb, databunch, str(item_id_values))
        items_tga_inputs[str(ReD_or_id)], items_tga_embed[ReD_or_id] = \
            TextToBERTRT(BERT_items_tga, databunch, tga)
        items_title_inputs[str(ReD_or_id)], items_title_embed[ReD_or_id] = \
            TextToBERTRT(BERT_items_title, databunch, item_id_values['title'])

    
    # save
    torch.save(items_full_kb_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/items_full_kb.pth')
    torch.save(items_tga_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/items_tga.pth')
    torch.save(items_title_inputs, path_to_ReDial + '/Data/CF2/RT/BERTInput/items_title.pth')
    torch.save(items_full_kb_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/items_full_kb.pth')
    torch.save(items_tga_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/items_tga.pth')
    torch.save(items_title_embed, path_to_ReDial + '/Data/CF2/RT/PoolerEmbed/items_title.pth')
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    