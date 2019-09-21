import torch.nn as nn
import torch 
from torch.optim import optim
import time
import date
import logging
from tqdm import tqdm
import tensorboardX

logger = logging.getLogger(__name__)

class BaseTrainner(object):

	def __init__(self,model,dataloader,config):
		self.model = model
		self.dataset = dataset
		self.__dict__.update(config)

	def train(self):
		raise NotImplementedError()
	def load_checkpoint(self,file_dir):
		pass

	def save_checkpoint(self,file_dir):
		pass
	def evaluate(self):
		pass

 class NaiveTrainner(BaseTrainner):
 	def __inti__(self,model,dataloader,config):
 		super(NaiveTrainner,self).__init__()
 		self.model = model
 		self.dataset = dataset
 		self.__dict__.update(config)

 	def train(self):
 		pass

 class BackBoostTrainner(BaseTrainner):
 	def __init__(self,model,dataloader,config):
 		super(NaiveTrainner,self).__init__()
 		self.model = model 
 		self.dataset = dataset
 		self.__dict__.update(config)

 	def train(self):
 		pass

class SelfBoostTrainner(BaseTrainner):

	def __init__(self,model,dataloader,config):
		super(NaiveTrainner,self).__init__()
		self.model = model
		self.dataset = dataset
		self.__dict__.update(config)

	def train(self):
		pass

class DualBoostTrainner(BaseTrainner):

	def __init__(self,model,dataloader,config):
		super(NaiveTrainner,self).__init__()
		self.model = model
		self.dataset = dataset
		self.__dict__.update(config)

	def train(self):
		pass