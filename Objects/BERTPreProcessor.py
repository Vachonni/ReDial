#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:33:39 2020


The BERTPreProcessor object.



    ***  ONLY FOR   BertForSequenceClassification   TYPE ***



Takes a BERT model and a related tokenizer.

Has different methods to get text into:
    - BERT input format for batches of 1. Used in TextToBERTRT to get RT one at a time
    - BERT input for batch > 1 and one speaker. Ready to use in regular batch training
    - BERT pooler output
    - **pending** BERT last layer output averaged 
    - ...
    

@author: nicholas
"""


import torch
from transformers import BertTokenizer
 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE = ', DEVICE )





class BERTPreProcessor():
    
    
    def __init__(self, model, 
                 tokenizer_path='bert-base-uncased'):
                
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = model

        

        
    def TextToBERTInpBatch1(self, text, max_length=512):
        """
        From a string to a BERT input.
        The output is in a batch 1 format. (each element one line matrix format, not vector)        
        """
        
        
        # text can't be empty
        if text == '': text = ' '
        

        input_dict = {}
        encoded_dict = self.tokenizer.encode_plus(text, max_length=max_length)
        
        # Get the actual lenght of the input
        length = len(encoded_dict['input_ids'])
        qt_to_add = max_length - length 
        
        # Prepare inputs we'll add ('PAD' = token 100, with a special mask token)
        PAD_to_add = [100] * qt_to_add
        masks_to_add = [1] * qt_to_add
        
        # Add them
        encoded_dict['input_ids'] = encoded_dict['input_ids'] + PAD_to_add
        encoded_dict['special_tokens_mask'] = encoded_dict['special_tokens_mask'] + \
                                              masks_to_add                                    
                                          
        # Turn into torch tensors and add 'inputs_ids' to input_dict
        input_dict['input_ids'] = torch.tensor([encoded_dict['input_ids']]).long().to(DEVICE)
        encoded_dict['special_tokens_mask'] = torch.tensor([encoded_dict['special_tokens_mask']])
        
        # Add 'attention mask' tokens (it's the inverse of 'special_tokens_mask')
        input_dict['attention_mask'] = ((encoded_dict['special_tokens_mask'] -1) * -1).float().to(DEVICE)
        
        
        # Add token_type and position_ids 
        input_dict['token_type_ids'] = torch.zeros(1, max_length).long().to(DEVICE)
        input_dict['position_ids'] = torch.arange(max_length).unsqueeze(0).to(DEVICE)
    
        
        return input_dict
    
    
    
    
    
    def TextToBERTInp1Speaker(self, text, max_length=512):
        """
        From a string to a BERT input flatten and without 'token_type_ids' and
        'position_ids'. 
        Good for one speaker inputs (because creates smaller data and anyway, 
        BERT reverts to this setting).
        Used for CF2 RT.
        The output is for batch > 1. (each element one vector)        
        """
    
        complete_input = self.TextToBERTInpBatch1(text)
        
        # Remove 'token_type_ids' and 'position_ids'
        complete_input.pop('token_type_ids')
        complete_input.pop('position_ids')
        
        # Flatten the tensors in complete_input
        flatten_input = {k:v[0] for k,v in complete_input.items()}

        return flatten_input
        
    
    
    
    
    
    # def TextToBERTAvrg(self, text, max_length=512):
    #     """
    #     From a string to BERT last hidden layer averaged 
    #     """
        
    #     # Get text input a BERT ready input
    #     input_to_bert = self.TextToBERTInpBatch1(text, max_length)
    #     # Pass input through model. Take last hidden layer as output (idx 0)
    #     last_hidden_layer = self.model(**input_to_bert)[0]
    #     avrg_last_hidden_layer = last_hidden_layer.mean(dim=1)
        
        
    #     return avrg_last_hidden_layer
    
    
    
    
    
    
    def TextToBERTPooler(self, text, max_length=512):
        """
        From a string to BERT pooler output
        """

        # Need the output of pooler layer. Not available by default for 
        # BertForSequenceClassification. Hence, we need to get access to 
        # the submodels.
        generator_of_models_modules = self.model.named_children()
        dict_of_models_modules = {}
        for name, module in generator_of_models_modules:
            dict_of_models_modules[name] = module

        # Get text input a BERT ready input
        input_to_bert = self.TextToBERTInpBatch1(text, max_length)
        # Pass input through 'bert' model (excludes the Dropout and \
        # classification layer. Take pooler output (idx 1, just like regular 
        # Bert Model)
        pooler_output = dict_of_models_modules['bert'](**input_to_bert)[1].detach()

        
        return pooler_output       
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    