#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:52:13 2019


Models used for CF2


@author: nicholas
"""


import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
    



def DotProduct(tensor1, tensor2):
    
    logits = (tensor1 * tensor2).sum(dim=1)
    pred = torch.sigmoid(logits)
    
    return pred, logits





class MLP(nn.Module):
    """
    Input:
        user: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
        item: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
    Output:
        A tensor of shape (batch, 1) representing the predicted ratings of each user-item pair (+ the logits)
        
    """


    def __init__(self, input_size=2*768, hidden_size=512, output_size=1):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
          nn.Linear(input_size ,hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size ,output_size),
        )
        
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[2].weight)
       
        
    def forward(self, user, item):
        
        # Concatenate user and item
        user_item = torch.cat((user, item), dim = -1)
        
        # Make a prediction
        logits = self.model(user_item).squeeze()
        pred = torch.sigmoid(logits)

        return pred, logits






class MLPLarge(nn.Module):
    """
    Input:
        user: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
        item: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
    Output:
        A tensor of shape (batch, 1) representing the predicted ratings of each user-item pair (+ the logits)
        
    """


    def __init__(self):
        super(MLPLarge, self).__init__()
        
        self.model = nn.Sequential(
          nn.Linear(2*768 ,512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),   
          nn.Linear(256, 128),
          nn.ReLU(), 
          nn.Linear(128, 64),
          nn.ReLU(),  
          nn.Linear(64 ,1),
        )
        
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[2].weight)
        nn.init.xavier_uniform_(self.model[4].weight)
        nn.init.xavier_uniform_(self.model[6].weight)
        nn.init.xavier_uniform_(self.model[8].weight)
    
        
        
    def forward(self, user, item):
        
        # Concatenate user and item
        user_item = torch.cat((user, item), dim = -1)
        
        # Make a prediction
        logits = self.model(user_item).squeeze()
        pred = torch.sigmoid(logits)

        return pred, logits
    
    
    
    
    
    
    

class MLPLargeSELU(nn.Module):
    """
    Input:
        user: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
        item: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
    Output:
        A tensor of shape (batch, 1) representing the predicted ratings of each user-item pair (+ the logits)
        
    """


    def __init__(self):
        super(MLPLargeSELU, self).__init__()
        
        self.model = nn.Sequential(
          nn.Linear(2*768 ,512),
          nn.SELU(),
          nn.Linear(512, 256),
          nn.SELU(),   
          nn.Linear(256, 128),
          nn.SELU(), 
          nn.Linear(128, 64),
          nn.SELU(),  
          nn.Linear(64 ,1),
        )
        
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[2].weight)
        nn.init.xavier_uniform_(self.model[4].weight)
        nn.init.xavier_uniform_(self.model[6].weight)
        nn.init.xavier_uniform_(self.model[8].weight)
    
        
        
    def forward(self, user, item):
        
        # Concatenate user and item
        user_item = torch.cat((user, item), dim = -1)
        
        # Make a prediction
        logits = self.model(user_item).squeeze()
        pred = torch.sigmoid(logits)

        return pred, logits
    
    
    
    
    
    

class MLPXLarge(nn.Module):
    """
    Input:
        user: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
        item: A tensor of shape (batch, x)        (e.g: BERT average representation (batch, 768))
    Output:
        A tensor of shape (batch, 1) representing the predicted ratings of each user-item pair (+ the logits)
        
    """


    def __init__(self):
        super(MLPXLarge, self).__init__()
        
        self.model = nn.Sequential(
          nn.Linear(2*768 ,512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),   
          nn.Linear(256, 128),
          nn.ReLU(), 
          nn.Linear(128, 64),
          nn.ReLU(),  
          nn.Linear(64, 32),
          nn.ReLU(),  
          nn.Linear(32, 16),
          nn.ReLU(),  
          nn.Linear(16, 8),
          nn.ReLU(),
          nn.Linear(8 ,1),
        )
        
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[2].weight)
        nn.init.xavier_uniform_(self.model[4].weight)
        nn.init.xavier_uniform_(self.model[6].weight)
        nn.init.xavier_uniform_(self.model[8].weight)
        nn.init.xavier_uniform_(self.model[10].weight)
        nn.init.xavier_uniform_(self.model[12].weight)
        nn.init.xavier_uniform_(self.model[14].weight)       
        
        
        
    def forward(self, user, item):
        
        # Concatenate user and item
        user_item = torch.cat((user, item), dim = -1)
        
        # Make a prediction
        logits = self.model(user_item).squeeze()
        pred = torch.sigmoid(logits)

        return pred, logits





class NeuMF(nn.Module):
    
    def __init__(self):
        super(NeuMF, self).__init__()
        
        self.NCF = nn.Sequential(
          nn.Linear(2*768 ,512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),   
          nn.Linear(256, 128)
        )
        
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[2].weight)
        nn.init.xavier_uniform_(self.model[4].weight)

        
        self.model = nn.Sequential(
          nn.Linear(768+128, 512),
          nn.ReLU(),
          nn.Linear(512, 128),
          nn.ReLU(),   
          nn.Linear(128, 1)
        )
        
        nn.init.xavier_uniform_(self.model[0].weight)
        nn.init.xavier_uniform_(self.model[2].weight)
        nn.init.xavier_uniform_(self.model[4].weight)

        
        
        
    def forward(self, user, item):
        
        # Concatenate user and item
        user_item = torch.cat((user, item), dim = -1)
        
        # Linear combination
        linear = DotProduct(user, item)
        
        # Non-linear combination
        non_linear = self.NCF(user_item)
        
        # Combine linear and non-linear
        linear_non_linear = torch.cat((linear, non_linear), dim = -1)
        
        # Make a prediction
        logits = self.model(linear_non_linear).squeeze()
        pred = torch.sigmoid(logits)

        return pred, logits







class TrainBERT(nn.Module):
    """ 
    A Model that takes in 2 BERT_input: user and item.
    
    Passed each through the SAME BERT_Model. 
    
    Averages the last_hidden_layer.
    
    Passed it through MLP model or DotProduct (depending on model)
    to get a prediction (and logits)
    """
    
    
    def __init__(self, model, input_size=2*768, hidden_size=512, output_size=1):
        super(TrainBERT, self).__init__()
        
        if model == 'TrainBERTDotProduct':
            self.merge = DotProduct
        elif model == 'TrainBERTMLP':
            self.merge = MLP(input_size, hidden_size, output_size)
        
        self.BERT = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        
        
    def forward(self, user, item):
        
        # # Get user's BERT_avrg value
        # user_last_hidden_layer = self.BERT(**user)[0]
        # user_avrg_last_hidden_layer = user_last_hidden_layer.mean(dim=1)

        # # Get item's BERT_avrg value
        # item_last_hidden_layer = self.BERT(**item)[0]
        # item_avrg_last_hidden_layer = item_last_hidden_layer.mean(dim=1)    
        
        
        
        """ Trying with Pooler """
        
        # Get user's BERT_avrg value
        user_avrg_last_hidden_layer = self.BERT(**user)[1]

        # Get item's BERT_avrg value
        item_avrg_last_hidden_layer = self.BERT(**item)[1]
      
        
        """  """
        
        
        # Return pred and logits, according to matching factor
        return self.merge(user_avrg_last_hidden_layer, item_avrg_last_hidden_layer)





class Train2BERT(nn.Module):
    """ 
    A Model that takes in 2 BERT_input: user and item.
    
    Passed each through the EACH IT'S OWN BERT_Model. 
    
    Averages the last_hidden_layer.
    
    Passed it through MLP model or DotProduct (depending on model)
    to get a prediction (and logits)
    """
    
    
    def __init__(self, model, input_size=2*768, hidden_size=512, output_size=1):
        super(Train2BERT, self).__init__()
        
        
        if model == 'TrainBERTDotProduct' or  model == 'Train2BERTDotProduct':
            self.merge = DotProduct
        elif model == 'TrainBERTMLP' or model == 'Train2BERTMLP':
            self.merge = MLP(input_size, hidden_size, output_size)
        
        
        self.BERT_user = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.BERT_item = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        
        
    def forward(self, user, item):
        
        # # Get user's BERT_avrg value
        # user_last_hidden_layer = self.BERT_user(**user)[0]
        # user_avrg_last_hidden_layer = user_last_hidden_layer.mean(dim=1)

        # # Get item's BERT_avrg value
        # item_last_hidden_layer = self.BERT_item(**item)[0]
        # item_avrg_last_hidden_layer = item_last_hidden_layer.mean(dim=1)    
        
        
        
        """ Trying with Pooler """
        
        # Get user's BERT_avrg value
        user_avrg_last_hidden_layer = self.BERT_user(**user)[1]

        # Get item's BERT_avrg value
        item_avrg_last_hidden_layer = self.BERT_item(**item)[1]
      
        
        """  """
        
        
        # Return pred and logits, according to matching factor
        return self.merge(user_avrg_last_hidden_layer, item_avrg_last_hidden_layer)



