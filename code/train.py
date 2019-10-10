import random
from torch import nn,optim
import torch
import time
import logging
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from models import AttnEncoderDecoder,RNNLMModel
from model_util import beam_search
from model_util import fluent_score
from data_util import RandomSubsetSampler,RandomIndicesSubsetSampler
from data_util import load_data_into_parallel, PaddedTensorLanguageDataset
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from fairseq.data import  Dictionary
from config import Config
from collections import defaultdict
logger = logging.getLogger(__name__)
device = torch.device('gpu') if torch.cuda.is_available() and Config['use_cuda'] else torch.device('cpu')

class BaseTrainner(object):
	
	def __init__(self,config):
		self.__dict__.update(config)
		self.writer = SummaryWriter('../summary')
		
	def train(self):
		raise NotImplementedError()

	def evaluate(self):
		raise NotImplementedError()

	def update_parameters_step(self,model,optimizer,dataset,cur_epoch=0,every_batch_to_save=6):
		#the epochs parameter is tracked for the whole iteration
		#for example the iteration is calculate by the formal:
		#iteration = epochs * batch_num * + i 
		#update parameters on dataset in one epoch
		#defualt optimizer is that in this class 
		sampler = RandomSampler(dataset)
		train_dataloader = DataLoader(dataset,sampler=sampler,batch_size=self.batch_size)
		for batch_num,batch in tqdm(enumerate(train_dataloader)):

			optimizer.zero_grad()
			logger.info('start training on the %d batch' % batch_num)
			hidden = model.init_hidden()
			input_seqs,output_seqs,input_mask,output_mask = batch
			loss = torch.zeros_like(output_seqs[:,0].squeeze(),dtype=torch.float)
			output,hidden = model(output_seqs[:,0],input_seqs,hidden,input_mask,first_step=True)
			prediction,target = output.squeeze(),output_seqs[:,1].squeeze()
			tmp_loss = self.criterion(prediction,target)
			tmp_loss = tmp_loss * output_mask[:,1].squeeze()
			loss += tmp_loss
			for i in range(1,self.max_len-1):
				optimizer.zero_grad()
				output,hidden = model(output_seqs[:,i],input_seqs,hidden,input_mask)
				prediction,target = output.squeeze(),output_seqs[:,i+1].squeeze()
				tmp_loss = self.criterion(prediction, target)
				tmp_loss = tmp_loss * output_mask[:, 1].squeeze()
				loss += tmp_loss
			this_iter_num = cur_epoch * batch_num
			loss = torch.sum(loss)
			loss.backward()
			optimizer.step()
			#save the check point
			if (batch_num + 1) % every_batch_to_save == 0:
				chpt_model_name = time.strftime("%y-%m-%d-%H:%M:%S",time.localtime()) + "-%dbt.model"%batch_num
				self.save_checkpoint(model,file_dir)
			self.writer.add_scalar('loss', loss.item(), this_iter_num)
	def load_checkpoint(self,file_dir):

		if not os.path.exist(file_dir):
			raise FileNotFoundError('the model not found in %s' % file_dir)
		elif file_dir[-6:] != '.model':
			raise ValueError("model name must end with '.model' ")
		return torch.load(file_dir)

	def save_checkpoint(self,model,file_dir):
		torch.save(model,file_dir)
		logger.info('save model into %s '%file_dir)

class BackBoostTrainner(BaseTrainner):
	"""the implement of backboost algorith
	it first train a generation model,then use it to generate
	some error-right pair, which(reversed pair) will be added into the
	training set for training our correction model
	"""
	def __init__(self,correction_model,dataset,config,
		reversed_parallel_dataset=None,generation_model=None):
		super(BaseTrainner,self).__init__()
		self.__dict__.update(config)
		self.correction_model = correction_model
		self.generation_model = generation_model
		self.dataset = dataset
		self.reversed_parallel_dataset = reversed_parallel_dataset
		self.correction_model_optimizer = optim.Adam(self.correction_model.parameters(),lr=self.learning_rate)
		#self.generation_model_optimizer = optim.Adam(self.generation_model.parameters(),lr=self.learning_rate)
		self.config = config
		self.dictionary = Dictionary().load(Config['word_dict'])
		self.bos = torch.tensor(self.dictionary.bos(),dtype=torch.int64,device=device)
		self.writer = SummaryWriter('../summary')
		
	def train(self,criterion):
		#train a model that could generate the basic error
		#sentence that used for the latter step
		self.criterion = criterion
		for epoch in tqdm(range(self.epochs)):
			#self.update_parameters_step(self.generation_model, self.generation_model_optimizer,self.reversed_parallel_dataset,cur_epoch=epoch)
			logger.info('have finised the %d epoch  on training generation model' % epoch)
			#back_boost_disfluent_dataset = self.generate_back_boost_set(self.generation_model)
			#concat_dataset = (self.dataset,back_boost_disfluent_dataset)
			concat_dataset = (self.dataset, self.dataset)
			after_amplification_dateset = ConcatDataset(concat_dataset)
			self.update_parameters_step(self.correction_model, self.correction_model_optimizer,self.dataset,cur_epoch=epoch)
			logger.info('have finised %d epoch on training correction model' % epoch)
			logger.info("have finished training on whole epoch")
		
	def _generate_one_epoch_disfulent_set(self,back_boost_disfluent_dataset,size=0.9):
		if len(back_boost_disfluent_dataset) / len(self.dataset) < size:
			size = len(back_boost_disfluent_dataset) / len(self.dataset)
		elif size > 1.0:
			size = 1.0
		choosed_index = list(RandomIndicesSubsetSampler(back_boost_disfluent_dataset.keys()),subset_size=size)
		choosed_pairs = [(random.choice(back_boost_disfluent_dataset[index][0]),back_boost_disfluent_dataset[index][1])
						for index in choosed_index]
		this_disfulent_dataset = PaddedTensorLanguageDataset(choosed_pairs)
		return this_disfulent_dataset
	
	def generate_back_boost_set(self,model):
		#first use the beam seach get the n best candidate
		#then by the back formual to calculate the
		#return type: Dataset Dictionary
		#detail index: (error sentence list,right)
		back_boost_disfluent_dataset = {}
		for index,src,src_mask,trg,trg_mask in enumerate(reversed_parallel_dataset):
			src = src[src_mask == 1]
			all_candidate = beam_search(model,input_seqs,self.bos,beam_size=self.beam_size)
			back_set = [seq_gen for seq_gen in all_candidate if (fluent_score(input_seqs) + 1e-6) / (fluent_score(seq_gen) + 1e-6) > self.sigma]
			if back_set:
				back_boost_disfluent_dataset[index] = (back_set,src)
		return back_boost_disfluent_dataset


class SelfBoostTrainner(BaseTrainner):
	"""self-boost learing implementation,only training
	single correction model, during traning, dynamically
	expand the data"""
	def __init__(self,model,dataset,config):
		super(NaiveTrainner,self).__init__()
		self.__dict__.update(config)
		self.correction_model = model
		self.original_set = dataset
		self.original_indexs = list(range(len(dataset)))
		self.optimzier = optim.Adam(model.parameters(),lr=self.learning_rate)
		self.selfset_dict = defaultdict(list)
		
	def train(self,criterion):
		self.criterion = criterion
		expansion_dataset = self.original_dataset
		for epoch in tqdm(range(self.epochs)):
			logger.info("start training in epoch %d" % epoch)
			self.update_parameters_step(self.correction_model,self.optimzier,expansion_dataset,cur_epoch=epoch)
			choosed_idxs = RandomIndicesSubsetSampler(self.original_indexs,subset_size=0.8)
			foo = [()]
			for idx in choosed_idxs:
				raw_sentence,mask,correction_sentence,cor_mask = self.original_set[idx]
				raw_sentence = raw_sentence[mask==1]
				correction_sentence = correction_sentence[cor_mask==1]
				candidate_correction = beam_search(self.correction_model,raw_sentence,self.bos)
				tmp_self_set = [fluent_score(correction_sentence) / (1e-6+fluent_score(sen)) >= self.sigma
				                for sen in correction_sentence]
				self.selfset_dict[idx] += tmp_self_set
				if self.selfset_dict[idx]:
					generation_raw_sentence = random.choice(self.selfset_dict[idx])
					foo += [()]
			data = Dataset(foo)
			expansion_dataset = ConcatDataset((data,self.original_set))
		logger.info("finished training on whole epoch %d" % self.epochs)

		
class DualBoostTrainner(BaseTrainner):

	def __init__(self,model,dataset,config):
		super(NaiveTrainner,self).__init__()
		self.model = model
		self.dataset = dataset
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		self.__dict__.update(config)
		
	def train(self):
		pass


class LMTrainner():
	def __init__(self,model,dataset,config):
		self.model = model
		self.dataset = dataset
		self.__dict__.update(config)
		self.optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)
		self.criterion = nn.CrossEntropyLoss(reduction='none')
		#the summary write dir,variety by the time at train model
		self.writer = SummaryWriter("../summary/lm/" + time.strftime("%Y-%m-%d@%H:%M:%S",time.localtime()))
		self.model_save_to_dir = "../output_models/lm/" + time.strftime("%Y-%m-%d@%H:%M:%S",time.localtime())
	def train(self,epochs):
		self.epochs = epochs
		sampler = RandomSampler(self.dataset)
		dataloader = DataLoader(self.dataset,sampler=sampler,batch_size=self.batch_size)
		for epoch in tqdm(range(self.epochs)):
			logger.info('start training model on %d epoch'%epoch)
			self.update_parameters_step(dataloader,epoch)
		logger.info('finished training on whole %d epoch' % self.epochs)
		
	def update_parameters_step(self,dataloader,cur_epcoch,every_batch_to_save=6):
		batch_num = len(dataloader)
		for i,batch in tqdm(enumerate(dataloader)):
			cur_batch = batch_num * cur_epcoch + i
			self.optimizer.zero_grad()
			sentence,_,mask = batch[1:]
			hidden = self.model.init_hidden()
			loss = 0.0
			for i in range(int(sentence.shape[0])):
				output,hidden = self.model(sentence[:,i],hidden)
				output,tmp_mask = output.squeeze(), mask[:,i+1].squeeze()
				target = sentence[:,i+1].squeeze()
				tmp_loss = self.criterion(output,target)
				tmp_loss = tmp_loss * tmp_mask
				loss += torch.sum(tmp_loss)
			loss.backward()
			logger.info('current loss is %.2f'%loss.item())
			self.optimizer.step()
			if (cur_batch + 1) % every_batch_to_save == 0:
				self.writer.add_scalar('loss',loss.item(),cur_batch)
				self.save_checkpoint(self.model,"-epoch%d-batch%d.model"%(cur_epcoch,i))
			logger.info("finished %d batch in %d epoch" % (i,cur_batch))
		logger.info('finished training on epoch %d'%cur_batch)
		
	def save_checkpoint(self,model,file_dir):
		
		if not os.path.exists(self.model_save_to_dir):
			os.makedirs(self.model_save_to_dir)
		file_dir = os.path.join(self.model_save_to_dir,file_dir)
		torch.save(model,file_dir)
		logger.info('save model into %s '%file_dir)
		
def main():
	lm_model = RNNLMModel(Config)
	src_file = '../data/nucle/nucle-train.tok.src'
	trg_file = '../data/nucle/nucle-train.tok.trg'
	src_trg_pair_langs = load_data_into_parallel(src_file, trg_file)
	train_dataset = PaddedTensorLanguageDataset(src_trg_pair_langs)
	lm_trainner = LMTrainner(lm_model,train_dataset,Config)
	lm_trainner.train(5)
if __name__ == "__main__":
	main()