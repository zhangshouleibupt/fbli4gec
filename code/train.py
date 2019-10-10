import random
from torch import nn,optim
import torch
import time
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from models import AttnEncoderDecoder
from model_util import beam_search
from model_util import fluent_score
from data_util import RandomSubsetSampler,RandomIndicesSubsetSampler
from data_util import load_data_into_parallel, PaddedTensorLanguageDataset
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from fairseq.data import  Dictionary
from config import Config
logger = logging.getLogger(__name__)
device = torch.device('gpu') if torch.cuda.is_available() and Config['use_cuda'] else torch.device('cpu')

class BaseTrainner(object):
	"""
	"""
	def __init__(self,config):
		self.__dict__.update(config)
		self.writer = SummaryWriter('../summary')
	"""common method in all trainner, all trainner that
	inherit from the basetrainner must implement it
	"""
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
			print("-----passed----- ")
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


def main():
	correction_model = AttnEncoderDecoder(Config)
	generation_model = AttnEncoderDecoder(Config)
	src_file = '../data/nucle/nucle-train.tok.src'
	trg_file = '../data/nucle/nucle-train.tok.trg'
	src_trg_pair_langs = load_data_into_parallel(src_file, trg_file)
	train_dataset = PaddedTensorLanguageDataset(src_trg_pair_langs)
	sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=16)
	reversed_train_dataset = PaddedTensorLanguageDataset(src_trg_pair_langs, reversed=True)
	reversed_train_dataloader = DataLoader(reversed_train_dataset, sampler=sampler, batch_size=16)
	correction_model.init_weights()
	criterion = nn.CrossEntropyLoss(reduction="none")
	trainner = BackBoostTrainner(correction_model,train_dataset,Config)
	trainner.train(criterion)
if __name__ == "__main__":
	main()