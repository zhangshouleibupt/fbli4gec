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
	'uniform_num' : 0.1,
	'attn_mothod' : 'general',
	'embedding_dim' : 500,
	'encoder_hidden_dim' : 500,
	'decoder_hidden_dim' : 500,
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
		self.dropout = nn.Dropout(self.dropout_prob)
		self.encoder_rnn_layer = nn.GRU(self.embedding_dim,self.encoder_hidden_dim,
								num_layers=self.encoder_rnn_layer_size,bidirectional=True)
	def forward(self,seq_input):
		hidden_0 = self.init_hidden()
		#transpose the input shape into rnn need format,
		#which should be : (seq_len,batch,embedding_dim)
		seq_embeded = self.dropout(self.embedding(seq_input)).transpose(0,1)
		seq_out,h_last = self.encoder_rnn_layer(seq_embeded,hidden_0)
		return seq_input[:,:,self.encoder_hidden_dim:].transpose(0,1),h_last.transpose(0,1)
	def init_hidden(self):
		b = self.batch_size
		num = 2 if self.user_birectional else 1 
		return torch.zeros(num*self.encoder_rnn_layer_size,b,self.encoder_hidden_dim,device=device)

class Attention(nn.Module):

		"""
		the implementation  of three way attention in papers:
		Effective Approaches to Attention-based Neural Machine Translation,
		defaultly it is the global way cause the main paper (after mentioned) use it,
		use example:
		input include the output hidden of time t(decoder hidden), also called h_t,
		(if we use the concat way,we also need the last time (t-1) decoder's input),
		and all of the hidden of the encoder output,
		for simplity,we uniformaly call the seperately query,key(s),value(s),which in here
		the key(s) is identical the value(s)
		the initial input query shape : (batch_size,hidden_size),
						  keys shape : (batch_szie,seq_len,hidden_size),
						  values shape : (batch_size,seq_len,hidden_size)
		if needed we reshaped the query shape into (batch_size,1,hidden_size)
		"""

	def __init__(self,attn_mothod,input_hidden_dim,output_hidden_dim,config):
		super(Attention,self).__init__()
		self.attn_method = attn_method
		if self.attn_method not in ['general','concat','dot']:
			raise NameError('input the appropriate attention way')
		self.query_dim = input_hidden_dim
		self.key_dim = output_hidden_dim
		self.use_mask = use_mask
		self.attention_weight = nn.ParameterDict({'concat':ParameterList(
														[nn.Parameter(torch.randn(self.query_dim)),
														nn.Parameter(torch.randn(self.key_dim*2,self.query_dim))]),
												  'general':nn.Parameter(torch.randn(self.key_dim,self.query_dim))})
		self.atten_fn_map = {
			'dot' : self.general_attn,
			'general' : self.dot_attn,
			'concat' : self.concat_attn,
		}
		self.softmax = nn.Softmax(dim=-1)

	def general_attn(self,q,K,V,mask=None):
		#format the query into (batch,1,query_dim) for simplifide 
		#the caclcalation 
		q.unsqueeze(1)
		mask.unsqueeze(2)
		projection = self.attention_weight['general'].mm(K)
		K,V = mask * K, mask * V
		attn_socre = torch.bmm(q,projection.transpose(1,2))
		attn_softmax_score = self.softmax(attn_socre)
		#maybe this way is not efficent!
		combine = torch.bmm(attn_softmax_score.transpose(1,2),V)
		return combine.unsqueeze(1)

	def concat_attn(self,q,K,V,mask=None):
		pass

	def dot_attn(self,q,K,V,mask=None):
		q.unsqueeze(1)
		mask.unsqueeze(2)
		K,V = mask * K, mask * V
		attn_score = torch.bmm(q,K.transpose(1,2))
		attn_softmax_score = self.softmax(attn_score)
		combine = torch.bmm(V.transpose(1,2),attn_softmax_score.transpose(1,2))
		return combime.unsqueeze(1)
	def forward(self,q,K,V,mask=None):
		return self.atten_fn_map[self.attn_method](q, K, V,mask=mask)

class AttnEncoderDecoder(nn.Module):

	def __init__(self,config):
		super(AttnEncoderDecoder,self).__init__()
		self.__dict__update(config)
		self.embedding = nn.Embedding(self.output_voc_size, embedding_dim)
		self.dropout = nn.Dropout(self.dropout_prob)
		self.encoder = Encoder(config)
		self.rnn_type_map = {
			'rnn': nn.RNN(),
			'gru': nn.GRU(),
			'lstm': nn.LSTM(),
		}
		self.attn = Attention(self.attn_mothod, self.input_hidden_dim, self.output_hidden_dim)
		if self.rnn_type not in ['rnn','lstm','gru']:
			raise NameError('input the appropriate rnn type')
		self.decoder_rnn_layer = nn.GRU(self.decoder_embeded_dim,self.decoder_hidden_dim,
										self.decoder_rnn_layer_size,bidirectional=True)
		self.concat_weight = nn.Linear(self.decoder_hidden_dim*2,self.decoder_hidden_dim)
		self.output_project = nn.Linear(self.decoder_hidden_dim,self.output_voc_size)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self,input_token,input_seqs,hidden,mask):
		#input_token shape : (batch,) after embedding shape : (batch,embed_dim)
		input_token_embeded = self.drpout(self.embedding(input_token))
		#add one dimension into input_token and transpose in (0,1) dim 
		#for the format into the rnn input data format
		input_token_embeded = input_token_embeded.unsqueeze(1).transpose(0,1)
		#use the t time step hidden as query
		encoder_seq_hiddens,h = self.encoder(input_seqs)
		decoder_rnn_out,hidden = self.decoder_rnn_layer(input_token_embeded,hidden)
		#make the shape into (bathc_size,hidden_dim)
		decoder_rnn_out = decoder_rnn_out[:,:,self.decoder_hidden_dim:].unsqueeze(1)
		attn_out = self.attn(decoder_rnn_out,encoder_seq_hiddens,encoder_seq_hiddens,mask)
		concat_fea = torch.concat(attn_out,decoder_rnn_out)
		out = self.concat_weight(concat_fea)
		out = self.output_project(out)
		out = self.softmax(out)
		return out,hidden

	def init_hidden(self):
		num = 2 if self.use_bidirectional else 1
		b = self.batch_size
		return torch.zeros(num*self.decoder_rnn_layer_size,b,
							self.decoder_hidden_dim,device=device)

	def init_weights(self):
		pass
	def load_model(self,path):
		pass
	
