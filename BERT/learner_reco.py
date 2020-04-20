#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:17:01 2019


Adapting file fast-bert.modeling.py for recommendation because:
    1- Just to insure that the right model is imported (from modeling_reco now)
    2- To manage the right config of imported pre-trained model

@author: nicholas
"""

from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures

# *** CHANGE ***
from BERT.modeling_reco import BertForMultiLabelSequenceClassification, XLNetForMultiLabelSequenceClassification, RobertaForMultiLabelSequenceClassification, DistilBertForMultiLabelSequenceClassification
import json
# *** CHANGE ***

from pathlib import Path

from torch.optim.lr_scheduler import _LRScheduler, Optimizer
from transformers import AdamW, ConstantLRSchedule

from torch.utils.tensorboard import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer, 
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                                  DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)

from pytorch_lamb import Lamb

from transformers import WarmupCosineSchedule, WarmupConstantSchedule, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule

MODEL_CLASSES = {
    'bert': (BertConfig, (BertForSequenceClassification, BertForMultiLabelSequenceClassification), BertTokenizer),
    'xlnet': (XLNetConfig, (XLNetForSequenceClassification, XLNetForMultiLabelSequenceClassification), XLNetTokenizer),
    'xlm': (XLMConfig, (XLMForSequenceClassification, XLMForSequenceClassification), XLMTokenizer),
    'roberta': (RobertaConfig, (RobertaForSequenceClassification, RobertaForMultiLabelSequenceClassification), RobertaTokenizer),
    'distilbert': (DistilBertConfig, (DistilBertForSequenceClassification, DistilBertForMultiLabelSequenceClassification), DistilBertTokenizer)
}


from transformers import BertForSequenceClassification
from fast_bert.bert_layers import BertLayerNorm
from fastprogress.fastprogress import master_bar, progress_bar
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

from fastai.torch_core import *
from fastai.callback import *

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
except:
    from fast_bert.bert_layers import BertLayerNorm as FusedLayerNorm

SCHEDULES = {
            None:       ConstantLRSchedule,
            "none":     ConstantLRSchedule,
            "warmup_cosine": WarmupCosineSchedule,
            "warmup_constant": WarmupConstantSchedule,
            "warmup_linear": WarmupLinearSchedule,
            "warmup_cosine_hard_restarts": WarmupCosineWithHardRestartsSchedule
        }



from Objects.MetricByMentions import MetricByMentions
from Objects.MetricByMentions import GetMetrics
from Objects.MetricByMentions import ToTensorboard
from Settings import nb_movies_ReDial, nb_users_ReDial




class BertLearner(object):
    
    @staticmethod
    def from_pretrained_model(dataBunch, pretrained_path, output_dir, metrics, device, logger, finetuned_wgts_path=None, 
                              multi_gpu=True, is_fp16=True, loss_scale=0, warmup_steps=0, fp16_opt_level='O1',
                              grad_accumulation_steps=1, multi_label=False, max_grad_norm=1.0, adam_epsilon=1e-8, 
                              logging_steps=100, items=False):
        
        model_state_dict = None
        
        model_type = dataBunch.model_type
        
        config_class, model_class, _ = MODEL_CLASSES[model_type]
        
# *** CHANGE ***
# If in recommender case        
        if dataBunch.labels == ['ratings']:
            if not items:
                config = config_class.from_pretrained(pretrained_path, num_labels=nb_movies_ReDial)
            elif items:
                config = config_class.from_pretrained(pretrained_path, num_labels=nb_users_ReDial)
# If multi-label
        else:
            config = config_class.from_pretrained(pretrained_path, num_labels=len(dataBunch.labels))
# *** CHANGE ***
   
            
        if finetuned_wgts_path:
            model_state_dict = torch.load(finetuned_wgts_path)
        else:
            model_state_dict = None
        
        if multi_label == True:
            model = model_class[1].from_pretrained(pretrained_path, config=config, state_dict=model_state_dict)
        else:
            model = model_class[0].from_pretrained(pretrained_path, config=config, state_dict=model_state_dict)
        
        model.to(device)
    
            
        return BertLearner(dataBunch, model, pretrained_path, output_dir, metrics, device, logger,
                           multi_gpu, is_fp16, loss_scale, warmup_steps, fp16_opt_level, grad_accumulation_steps, 
                           multi_label, max_grad_norm, adam_epsilon, logging_steps, items)
             
        
    def __init__(self, data: BertDataBunch, model: nn.Module, pretrained_model_path, output_dir, metrics, device,logger,
                 multi_gpu=True, is_fp16=True, loss_scale=0, warmup_steps=0, fp16_opt_level='O1',
                 grad_accumulation_steps=1, multi_label=False, max_grad_norm=1.0, adam_epsilon=1e-8, logging_steps=100, items=False):
        
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        

        self.multi_label = multi_label
        self.data = data
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.metrics = metrics
        self.multi_gpu = multi_gpu
        self.is_fp16 = is_fp16
        self.fp16_opt_level = fp16_opt_level
        self.adam_epsilon = adam_epsilon
        self.loss_scale = loss_scale
        self.warmup_steps = warmup_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.device = device
        self.logger = logger
        self.layer_groups = None
        self.optimizer = None
        self.bn_types = (BertLayerNorm, FusedLayerNorm)
        self.n_gpu = 0
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.max_steps = -1
        self.weight_decay = 0.0
        self.model_type = data.model_type
# *** CHANGE ***   To save when NDCG improves
        self.best_NDCG = 0
        self.items = items
# *** CHANGE ***       
        
        self.output_dir = output_dir
        
        
        if self.multi_gpu:
            self.n_gpu = torch.cuda.device_count()
        
    
    
    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer group `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not isinstance(l, self.bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)
        self.optimizer = None
    
                
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
            
    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True
    
    def freeze(self)->None:
        "Freeze up to last layer group."
        assert(len(self.layer_groups)>1)
        self.freeze_to(-1)
        self.optimizer = None

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)
        self.optimizer = None
    
    def bert_clas_split(self) -> List[nn.Module]:
        "Split the BERT `model` in groups for differential learning rates."
        if self.model.module:
            model = self.model.module
        else:
            model = self.model
        
        bert = model.bert
        
        embedder = bert.embeddings
        pooler = bert.pooler
        
        encoder = bert.encoder
        
        classifier = [model.dropout, model.classifier]
        
        n = len(encoder.layer)//3
        
        groups = [[embedder], list(encoder.layer[:n]), list(encoder.layer[n:2*n]), list(encoder.layer[2*n:]), [pooler], classifier]
        return groups
    
    
    def split(self, split_on:SplitFuncOrIdxList)->None:
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on()
        self.layer_groups = split_model(self.model, split_on)
        return self
    

    
    def get_optimizer(self, lr, t_total, schedule_type='warmup_linear', optimizer_type='lamb'):   
        
        
        
        # Prepare optimiser and schedule 
        no_decay = ['bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay },
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        if optimizer_type == 'lamb':
            optimizer = Lamb(optimizer_grouped_parameters, lr=lr, eps=self.adam_epsilon)
        elif optimizer_type == 'adamw':
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=self.adam_epsilon)
        
        
        schedule_class = SCHEDULES[schedule_type]

        scheduler = schedule_class(optimizer, warmup_steps=self.warmup_steps, t_total=t_total)
        
        return optimizer, scheduler
    
    
    ### Train the model ###    
    def fit(self, epochs, lr, validate=True, schedule_type="warmup_cosine", optimizer_type='lamb'):
        
        tensorboard_dir = Path(self.output_dir, 'runs', self.exp_id)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        print(tensorboard_dir)
        
        
        # Train the model
        tb_writer = SummaryWriter(tensorboard_dir)

        train_dataloader = self.data.train_dl
        if self.max_steps > 0:
            t_total = self.max_steps
            self.epochs = self.max_steps // len(train_dataloader) // self.grad_accumulation_steps + 1
        else:
            t_total = len(train_dataloader) // self.grad_accumulation_steps * epochs

        # Prepare optimiser and schedule 
        optimizer, _ = self.get_optimizer(lr, t_total, 
                                                  schedule_type=schedule_type, optimizer_type=optimizer_type)
        
        
        # get the base model if its already wrapped around DataParallel
        if hasattr(self.model, 'module'):
            self.model = self.model.module
        
        if self.is_fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError('Please install apex to use fp16 training')
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.fp16_opt_level)
        
        schedule_class = SCHEDULES[schedule_type]

        scheduler = schedule_class(optimizer, warmup_steps=self.warmup_steps, t_total=t_total)
        
        # Parallelize the model architecture
        if self.multi_gpu == True:
            self.model = torch.nn.DataParallel(self.model)
        
        # Start Training
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataloader.dataset))
        self.logger.info("  Num Epochs = %d", epochs)
        self.logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       self.data.train_batch_size * self.grad_accumulation_steps)
        self.logger.info("  Gradient Accumulation steps = %d", self.grad_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step =  0
        epoch_step = 0
        tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()
        pbar = master_bar(range(epochs))

        for epoch in pbar:
            epoch_step = 0
            epoch_loss = 0.0
            for step, batch in enumerate(progress_bar(train_dataloader, parent=pbar)):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                
                if self.model_type in ['bert', 'xlnet']:
                    inputs['token_type_ids'] = batch[2]
                    
                outputs = self.model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training
                if self.grad_accumulation_steps > 1:
                    loss = loss / self.grad_accumulation_steps

                if self.is_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                tr_loss += loss.item()
                epoch_loss += loss.item() 
                if (step + 1) % self.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    
                    self.model.zero_grad()
                    global_step += 1
                    epoch_step += 1

                    # Evaluate model during epoch, at specified frequency
                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        if validate:
                            # evaluate model
                            results = self.validate()
                            # add train_loss to results 
                            results['train_loss'] = (tr_loss - logging_loss)/self.logging_steps
                            # Add results to tensorboard
                            ToTensorboard(tb_writer, results, global_step, self.model, self.metrics)
                            for key, value in results.items():
                                if key == 'train_loss' or key == 'eval_loss': continue
                                self.logger.info("eval_{} after step {}: {}: ".format(key, global_step, value.Avrg()))
                        
                        # Log metrics
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        self.logger.info("lr after step {}: {}".format(global_step, scheduler.get_lr()[0]))
                        self.logger.info("train_loss after step {}: {}".format(global_step, (tr_loss - logging_loss)/self.logging_steps))
                        
                        logging_loss = tr_loss
            
            # If evaluate the model after every epoch
            if validate:
                # evaluate model
                results = self.validate()
                # add train_loss to model
                results['train_loss'] = epoch_loss/epoch_step
                 # Add results to tensorboard
                ToTensorboard(tb_writer, results, epoch + 1, self.model, self.metrics)
                for key, value in results.items():
                    if key == 'train_loss' or key == 'eval_loss': continue
                    self.logger.info("eval_{} after epoch {}: {}: ".format(key, (epoch + 1), value.Avrg()))

# *** CHANGE ***   To save when NDCG improves
                actual_NDCG = results['ndcg'].Avrg()
                if actual_NDCG > self.best_NDCG:
                   self.logger.info("NDCG Improved. Saving...")
                   self.results_to_save = results
                   self.save_model()                 
                   self.logger.info("\n                       ...saved")
                   self.best_NDCG = actual_NDCG 
# *** CHANGE ***                    
                
                
            # Log metrics
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch + 1)
            self.logger.info("lr after epoch {}: {}".format((epoch + 1), scheduler.get_lr()[0]))
            self.logger.info("train_loss after epoch {}: {}".format((epoch + 1), epoch_loss/epoch_step))  
            self.logger.info("\n")
            
        tb_writer.close()
        
        return global_step, tr_loss / global_step   
    
    
    
    
    
    
    ### Evaluate the model    
    def validate(self):
        """
        Evaluate model on validation data. 

        Returns
        -------
        results: a dict of eval_loss as 'loss' and all other keys
        ('avrg_rank', 'ndcg', 'recall@1', 'recall@10', 'recall@50')
        as MetricsByMentions objects
        """
        
        self.logger.info("Running evaluation")
        
        self.logger.info("  Num examples = %d", len(self.data.val_dl.dataset))
        self.logger.info("  Batch size = %d", self.data.val_batch_size)
        
        all_logits = None
        all_labels = None
        
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        preds = None
        out_label_ids = None
        
        
        # GET PREDICTIONS
        for step, batch in enumerate(progress_bar(self.data.val_dl)):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                
                if self.model_type in ['bert', 'xlnet']:
                    inputs['token_type_ids'] = batch[2]
                    
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
            
                
                eval_loss += tmp_eval_loss.mean().item()
            
            nb_eval_steps += 1
            nb_eval_examples += inputs['input_ids'].size(0)
                
# *** CHANGE ***            
            # Treat the recommender case (a 3D input (batch, nb of ratings, itemid+rating))
            if len(inputs['labels'].shape) == 3:
                logits = logits.softmax(dim=1)
# *** CHANGE ***
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), 0)
            
            # First batch
            if all_labels is None:
# *** CHANGE ***
                # Treat the recommender case (a 3D input (batch, nb of ratings, itemid+rating))
                if len(inputs['labels'].shape) == 3:
                    l_qt_movies_mentionned = []
                    ratings = torch.zeros_like(logits)
                    for i, list_itemid_rating in enumerate(inputs['labels']):
                        for (itemid, rating) in list_itemid_rating:
                            # If  itemid is -2, it's number of movies mentioned indicator
                            if itemid == -2: 
                                l_qt_movies_mentionned.append(rating)
                                continue
                            ratings[i, itemid] = rating
                    all_labels = ratings
                # Other cases
                else:
                    all_labels = inputs['labels']
            # ...after first batch 
            else:   
                # Treat the recommender case (a 3D input (batch, nb of ratings, itemid+rating))
                if len(inputs['labels'].shape) == 3:
                    ratings = torch.zeros_like(logits)
                    for i, list_itemid_rating in enumerate(inputs['labels']):
                        for (itemid, rating) in list_itemid_rating:
                            # If  itemid is -2, it's number of movies mentioned indicator
                            if itemid == -2: 
                                l_qt_movies_mentionned.append(rating.item())
                                continue
                            ratings[i, itemid] = rating
                    all_labels = torch.cat((all_labels, ratings), 0)
                # Other cases
                else:
                    all_labels =  torch.cat((all_labels, inputs['labels']), 0)
# *** CHANGE ***
 
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        
        
        
        # GET METRICS
        # TODO: Here:
        #   - all_logits are the Softmax predictions by line
        #   - all_labels are ratings for every possible movies (most at 0, of course)
        
        # Get loss
        eval_loss = eval_loss / nb_eval_steps
        results = {'eval_loss': eval_loss }           
        
        # If it's for items input, qt of "mentioned users" doesn't exits, always only one.
        if self.items:
            MetricByMentions.max_mentions = 1
        # Initialize the MetricsByMentions objects in the dict results
        for m in self.metrics:
            results[m] = MetricByMentions(m)
            

        # For every prediction (one at a time), get all metrics
        for logits, labels, mentions in zip(all_logits, all_labels, l_qt_movies_mentionned):

            # Insure their is at least one target movie 
            # (if not, sample not considered)
            if labels.sum() == 0: continue
            
            # Index of targets at 1 (i.e. liked movies) 
            targets_idx = labels.nonzero().flatten().tolist()
            
            # Get metrics for targets (we only care about liked movies)
            avrg_rk, ndcg, re_1, re_10, re_50 = GetMetrics(logits, \
                                                targets_idx, 100)    # 100 is topx value  
                        
            # Add metric to appropriate MetricByMentions obj
            results['avrg_rank'].Add(avrg_rk, mentions)
            results['ndcg'].Add(ndcg, mentions)
            results['recall@1'].Add(re_1, mentions)
            results['recall@10'].Add(re_10, mentions)
            results['recall@50'].Add(re_50, mentions)


        return results
    
    
    
    
    
    
    def save_model(self): 
        
        path = Path(self.output_dir, 'Results', self.exp_id)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.cuda.empty_cache() 
        # Save a trained model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        model_to_save.save_pretrained(path)
        
        # save the tokenizer
        self.data.tokenizer.save_pretrained(path)
        
        # save the metrics, only the average by mentions
        for key, value in self.results_to_save.items():
            if key == 'train_loss' or key == 'eval_loss': continue
            self.results_to_save[key] = value.AvrgByMentions()
            
        with open(Path(path, 'metrics.json'), 'w') as f:
            json.dump(self.results_to_save, f, indent=4)
    
    
    
    ### Return Predictions ###
    def predict_batch(self, texts=None):
        
        if texts:
            dl = self.data.get_dl_from_texts(texts)
        elif self.data.test_dl:
            dl = self.data.test_dl
        else:
            dl = self.data.val_dl
            
        all_logits = None

        self.model.eval()
        for step, batch in enumerate(dl):
            batch = tuple(t.to(self.device) for t in batch)
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         None }
            
            if self.model_type in ['bert', 'xlnet']:
                    inputs['token_type_ids'] = batch[2]
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]
# *** CHANGE ***
                if logits.size(-1) > 1000:
                    logits = logits.softmax(dim=1)
                elif self.multi_label:
                    print('$$$$$$$$$$$$$$$$$$  SHOULD NOT BE HERE FOR RECOMMENDAION')
# *** CHANGE ***
                    logits = logits.sigmoid()
                elif len(self.data.labels) == 2:
                    logits = logits.sigmoid()
                else:
                    logits = logits.softmax(dim=1)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
# *** CHANGE ***
        # Treat the recommender case 
        if self.data.labels == ['ratings']:
            result_df =  pd.DataFrame(all_logits)
        # Other cases
        else:
            result_df =  pd.DataFrame(all_logits, columns=self.data.labels)
        results = result_df.to_dict('record')
# *** CHANGE ***
        
        return [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results]
