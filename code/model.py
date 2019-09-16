import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import loggiing 

logger = loggiing.getLogger(__name__)

config = {
	'input_voc_size' : 100*1000,
	'output_voc_size' : 50 * 1000,
	'init_way' : 'uniform_init',
	'attn_mothod' : 'general',
	'embedding_dim' : 500,
	'encoder_rnn_layer_size' : 2,
	'decoder_rnn_layer_size' : 2,
	'learing_rate' : 0.1 * 0.001,
	'epochs' : 40, 
	'batch_size' : 128, 
	'dropput_prop' : 0.15, 
	'use_batch_norm' : False, 
	'use_seq_norm' : False, 
	'optim_engine' : 'Adam',
	'disfluente_candidate_size' : 10, 
	'use_cuda' : True,
}
device = torch.device('gpu') if torch.cuda.is_available() and config['use_cuda'] else torch.device('cpu')

class Encoder(nn.Module):
	def __init__(self,config):

		self.__dict__.update(config)
		self.embedding = nn.Embedding(self.input_voc_size, self.embedding_dim)

		self.encoder_rnn_layer = nn.ModuleList(
			[nn.GRU(self.embedding_dim,self.embedding_dim,bidirectional=True)
			for i in range(self.encoder_rnn_layer_size)])
	def forward(self,seq_input):
		seq_output = self.embedding(seq_input)
		for encoder_layer in self.encoder_rnn_layer:
			seq_output = self.encoder_layer(seq_input)
		return seq_input
	def init_hidden(self):
		pass

class Attention(nn.Module):

	def __init__(self,config):
		self.__dict__.update(config)
	def general_attn(self,Q,K,V):
		pass
	def concat_attn(self,Q,K,V):
		pass
	def dot_attn(self,Q,K,V):
		pass
	def forward(self,Q,K,V):
		if self.attn_method not in ['general','concat','dot']:
			raise NameError('input the appropriate attention way')
		if self.attn_method == 'general_attn':
			return self.general_attn(Q,K,V)
		elif self.attn_method == 'dot':
			return self.dot_attn(Q, K, V)
		else:
			return self.concat_attn(Q, K, V)

class AttnDecoder(nn.Module):

	def __init__(self,config):
		self.__dict__update(config)
		self.embedding = nn.Embedding(self.output_voc_size, embedding_dim)
		self.encoder = Encoder(config)
		self.rnn_type_map = {
			'rnn': nn.RNN(),
			'gru': nn.GRU(),
			'lstm': nn.LSTM(),
		}
		if self.rnn_type not in ['rnn','lstm','gru']:
			raise NameError('input the appropriate rnn type')
		self.decoder_rnn_layer = nn.ModuleList(
			[nn.GRU(self.embedding_dim,self.embedding_dim,bidirectional=True)
			for i in range(self.decoder_rnn_layer_size)])
		
	def forward(self,input_seq,input_hiddens):
		seq_
class 