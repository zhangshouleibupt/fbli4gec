import torch.nn as nn
import torch 
from torch.optim import optim
import time
import date
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import (BatchSampler,RandomSampler,
									  SequentialSampler,)
from .models import AttnEncoderDecoder
logger = logging.getLogger(__name__)

class BaseTrainner(object):

	def __init__(self,model,dataset,config):

		self.model = model
		self.dataset = dataset
		self.__dict__.update(config)

	def train(self):
		raise NotImplementedError()

	def load_checkpoint(self,file_dir):

	def save_checkpoint(self,file_dir):
		pass
	def evaluate(self):
		pass

	@property
	def model(self):
		return self.model
class NaiveTrainner(BaseTrainner):

 	def __inti__(self,model,dataset,config):
 		super(NaiveTrainner,self).__init__()
 		self.model = model
 		self.dataset = dataset
 		self.writer = SummaryWriter()
 		self.__dict__.update(config)

 	def train(self,criterion):
 		model.train()
 		sampler = RandomSampler(dataset)
 		train_dataloader = DataLoader(dataset,sampler=sampler,batch_size = self.batch_size)
 		loss = 0
 		optimizer = optim.Adam(model.parameters())
 		#dont use the teaching force totally
 		for epoch in tqdm(range(self.epochs)):
	 		for step,batch in train_dataloader:
	 			input_source,input_mask = (batch[0],batch[2])
	 			output_target,output_mask = (batch[1],batch[3])
	 			hidden = model.init_hidden()
	 			optimizer.zero_grad()
	 			output,hidden = model(output_target[0],input_source,hidden,input_mask,first_step=True)
	 			loss = criterion(output,output_target[1])
	 			loss = output_mask[1] * loss
	 			loss = torch.sum(loss)
	 			loss.backward()
	 			optimizer.step()
	 			for i in range(1,input_source.shape[0]-1):
	 				optimizer.zero_grad()
	 				ouput,hidden = model(output_target[i],input_source,hidden,input_mask)
	 				loss = criterion(output,output_target[i+1])
	 				loss = output_mask[i+1] * loss 
	 				loss = torch.sum(loss)
	 				loss.backward()
	 				optimizer.step()
	 		 logger.info('the %d epoch have finished' %(epoch))

class BackBoostTrainner(BaseTrainner):
	"""the implement of backboost algorithm
	"""
	def __init__(self,model,dataset,config):
		super(NaiveTrainner,self).__init__()
		self.model = model 
		self.dataset = dataset
		self.config = config
		self.__dict__.update(config)

	def train(self,reversed_parallel_dataset,criterion):
		#train a model that could generate the basic error 
		#sentence that used for the latter step
		generation_model = AttnEncoderDecoder(self.config)
 		naive_trainner = NaiveTrainner(model, reversed_parallel_dataset, config)
 		naive_trainner.train(criterion)
 		generation_model = naive_trainner.model
 	def generate_back
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