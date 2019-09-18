import torch.nn as nn
import torch 
from torch.optim import optim

class BaseTrainner(object):
	def __init__(self,model,dataset,config):
		self.model = model
		self.dataset = dataset
		self.__dict__.update(config)

	def train(self):
		raise NotImplementedError()

 class NaiveTrainner(BaseTrainner):
 	def __inti__(self,model,dataset,config):
 		super(NaiveTrainner,self).__init__()
 		self.model = model
 		self.dataset = dataset
 		self.__dict__.update(config)

 	def train(self):
 		pass

 class BackBoostTrainner(BaseTrainner):
 	def __init__(self,model,dataset,config):
 		super(NaiveTrainner,self).__init__()
 		self.model = model 
 		self.dataset = dataset
 		self.__dict__.update(config)

 	def train(self):
 		pass

class SelfBoostTrainner(BaseTrainner):

	def __init__(self,model,dataset,config):
		super(NaiveTrainner,self).__init__()
		self.model = model
		self.dataset = dataset
		self.__dict__.update(config)

	def train(self):
		pass

class DualBoostTrainner(BaseTrainner):

	def __init__(self,model,dataset,config):
		super(NaiveTrainner,self).__init__()
		self.model = model
		self.dataset = dataset
		self.__dict__.update(config)
		
	def train(self):
		pass