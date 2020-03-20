#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:33:37 2018


NVIDIA Auto-Encoder. Needs adaptation to be AutoRec.


@author: nicholas
"""

# Copyright (c) 2017 NVIDIA Corporation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import json

from Arguments import args 



def activation(input, kind):
  #print("Activation: {}".format(kind))
  if kind == 'selu':
    return F.selu(input)
  elif kind == 'relu':
    return F.relu(input)
  elif kind == 'relu6':
    return F.relu6(input)
  elif kind == 'sigmoid':
    return torch.sigmoid(input)   #F.Sigmoid depricated
  elif kind == 'tanh':
    return F.tanh(input)
  elif kind == 'elu':
    return F.elu(input)
  elif kind == 'lrelu':
    return F.leaky_relu(input)
  elif kind == 'swish':
    return input*F.sigmoid(input)
  elif kind == 'softmax':
    return F.softmax(input, dim=1)
  elif kind == 'none':
    return input
  else:
    raise ValueError('Unknown non-linearity type')





class AsymmetricAutoEncoder(nn.Module):
  def __init__(self, layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.0, \
               last_layer_activations=True, lla='none'):
    """
    Describes an AutoEncoder model with only ONE LAYER AS DECODER
    :param layer_sizes: Encoder network description. Should start with feature size (e.g. dimensionality of x).
    For example: [10000, 1024, 512] will result in:
      - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
      - decoder 2 layers: 512x10000.
    :param nl_type: (default 'selu') Type of no-linearity
    :param is_constrained: (default: True) Should constrain decoder weights
    :param dp_drop_prob: (default: 0.0) Dropout drop probability
    :param last_layer_activations: (default: True) Whether to apply activations on last decoder layer
    """

    super(AsymmetricAutoEncoder, self).__init__()
    self._dp_drop_prob = dp_drop_prob
    self._last_layer_activations = last_layer_activations
    self.lla = lla
    if dp_drop_prob > 0:
      self.drop = nn.Dropout(dp_drop_prob)
    self._last = 0 # Always only one layer of decode
    self._nl_type = nl_type

    # Create weight matrices for encoder, Xavier initialization, Create bias parameters
    self.encode_w = nn.ParameterList(
      [nn.Parameter(torch.rand(layer_sizes[i + 1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)])
    for ind, w in enumerate(self.encode_w):
      weight_init.xavier_uniform_(w)
    self.encode_b = nn.ParameterList(
      [nn.Parameter(torch.zeros(layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)])

    # reversed_enc_layers = list(reversed(layer_sizes))

    self.is_constrained = is_constrained
    if not is_constrained:
      self.decode_w = nn.ParameterList(
      [nn.Parameter(torch.rand(layer_sizes[0], layer_sizes[-1]))])
      for ind, w in enumerate(self.decode_w):
        weight_init.xavier_uniform_(w)
    self.decode_b = nn.ParameterList(
      [nn.Parameter(torch.zeros(layer_sizes[0]))])

    print("******************************")
    print("******************************")
    print(layer_sizes)
    print("Activation function:", self._nl_type)
    print("Dropout drop probability: {}".format(self._dp_drop_prob))
    print("Encoder pass:")
    for ind, w in enumerate(self.encode_w):
      print(w.data.size())
      print(self.encode_b[ind].size())
    print("Decoder pass:")
    if self.is_constrained:
      print('Decoder is constrained')
      for ind, w in enumerate(list(reversed(self.encode_w))):
        print(w.transpose(0, 1).size())
        print(self.decode_b[ind].size())
    else:
      for ind, w in enumerate(self.decode_w):
        print(w.data.size())
        print(self.decode_b[ind].size())
    print("******************************")
    print("******************************")


  def encode(self, x):
    for ind, w in enumerate(self.encode_w):
      x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    if self._dp_drop_prob > 0: # apply dropout only on code layer
      x = self.drop(x)
    return x


  def decode(self, z):
    if self.is_constrained:
      for ind, w in enumerate(list(reversed(self.encode_w))): # constrained autoencode re-uses weights from encoder
        z = activation(input=F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[ind]),
                     # last layer or decoder should not apply non linearities
                     kind=self._nl_type if ind!=self._last or self._last_layer_activations else 'none')
        #if self._dp_drop_prob > 0 and ind!=self._last: # and no dp on last layer
        #  z = self.drop(z)
    else:
      for ind, w in enumerate(self.decode_w):
        z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
                     # last layer or decoder should not apply non linearities
                     kind=self._nl_type if ind!=self._last or self._last_layer_activations else 'none')
        #if self._dp_drop_prob > 0 and ind!=self._last: # and no dp on last layer
        #  z = self.drop(z)

    return z


  def forward(self, x):
    return activation(self.decode(self.encode(x)), self.lla)


    

    
    
class GenresWrapperChrono(nn.Module):
    """
    Wraps the model_pre with g parameter(s) (for genres input)
    """
    
    def __init__(self, model_pre, g_type, genres_Inter):
        super(GenresWrapperChrono, self).__init__()
        self.model_pre = model_pre
        self.g_type = g_type
        # size of one input
        size = model_pre.encode_w[0].size(1)
        if self.g_type == 'none':
            self.g = nn.Parameter(torch.zeros(1,size), requires_grad=False)        
        if self.g_type == 'fixed':
            self.g = nn.Parameter(torch.ones(1,size), requires_grad=False)
        if self.g_type == 'one':
            self.g = nn.Parameter(torch.rand(1)/10)
        if self.g_type == 'genres':
            self.dict_genresInter_idx_UiD = genres_Inter
            self.g = nn.Parameter(torch.rand(1, len(self.dict_genresInter_idx_UiD))/10)            
        if self.g_type == 'unit':
            # Init g_i parameters that will weight the genres inputs, one by movie
        #   self.g = nn.Parameter(torch.rand(1,size)/10)     # Random init
            self.g = nn.Parameter(torch.zeros(1,size)+0.01)  # Constant init
        
        
    def forward(self, inputs):
        if self.g_type in ['none', 'fixed', 'one', 'unit']:
            x = inputs[0] + self.g * inputs[1][1]
        if self.g_type == 'genres':
            # Prepare the one_hot_matrix (of same cuda type than self.g)
            one_hot_mat = self.g.new_zeros(inputs[0].size(0), \
                                           len(self.dict_genresInter_idx_UiD))
            for i, g_idx in enumerate(inputs[1][0]):
                one_hot_mat[i, g_idx] = 1
            # Get the genres for each sample * top 100 normalized movies genres values
            x = inputs[0] + (self.g * one_hot_mat).sum(1, keepdim=True) * inputs[1][1]                
        
        return self.model_pre(x)
    
    
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    