import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import logging
from functools import partial
logger = logging.getLogger(__name__)
from config import Config
device = torch.device('gpu') if torch.cuda.is_available() and Config['use_cuda'] else torch.device('cpu')

class Encoder(nn.Module):

	def __init__(self,Config):
		super(Encoder,self).__init__()
		self.__dict__.update(Config)
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
		num = 2 if self.use_bidirectional else 1
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
    				  keys shape : (batch_size,seq_len,hidden_size),
    				  values shape : (batch_size,seq_len,hidden_size)
    if needed we reshaped the query shape into (batch_size,1,hidden_size)
    """
    def __init__(self,input_hidden_dim,output_hidden_dim):
        super(Attention,self).__init__()
        self.query_dim = input_hidden_dim
        self.key_dim = output_hidden_dim
        self.attn_weight = nn.Linear(input_hidden_dim,output_hidden_dim,bias=False)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,q,K,V,mask=None):
        q = q.unsqueeze(1)
        mask = mask.unsqueeze(2)
        K,V = mask * K, mask * V
        projection = self.attn_weight(K)
        attn_socre = torch.bmm(q,projection.transpose(1,2))
        attn_softmax_score = self.softmax(attn_socre)
        combine = torch.bmm(attn_softmax_score,V)
        return combine.unsqueeze(1)

class AttnEncoderDecoder(nn.Module):

	def __init__(self,Config):
		super(AttnEncoderDecoder,self).__init__()
		self.__dict__.update(Config)
		self.embedding = nn.Embedding(self.output_voc_size, self.embedding_dim)
		self.dropout = nn.Dropout(self.dropout_prob)
		self.encoder = Encoder(Config)
		self.rnn_type_map = {
			'rnn': nn.RNN,
			'gru': nn.GRU,
			'lstm': nn.LSTM,
		}
		self.attn = Attention(self.encoder_hidden_dim, self.decoder_hidden_dim)
		if self.rnn_type not in ['rnn','lstm','gru']:
			raise NameError('input the appropriate rnn type')
		self.decoder_rnn_layer = nn.GRU(self.embedding_dim,self.decoder_hidden_dim,
										self.decoder_rnn_layer_size,bidirectional=True)
		self.concat_weight = nn.Linear(self.decoder_hidden_dim*2,self.decoder_hidden_dim)
		self.output_project = nn.Linear(self.decoder_hidden_dim,self.output_voc_size)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self,input_token,input_seqs,hidden,mask,first_step=False):
		#input_token shape : (batch,) after embedding shape : (batch,embed_dim)
		input_token_embeded = self.dropout(self.embedding(input_token))
		#add one dimension into input_token and transpose in (0,1) dim
		#for the format into the rnn input data format
		input_token_embeded = input_token_embeded.unsqueeze(1).transpose(0,1)
		#use the t time step hidden as query
		encoder_seq_hiddens,h = self.encoder(input_seqs)
		if first_step:
			decoder_rnn_out,hidden = self.decoder_rnn_layer(input_token_embeded,h)
		else:
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
		b = self.batch_size
		num = 2 if self.use_bidirectional else 1
		return torch.zeros(num*self.encoder_rnn_layer_size,b,self.encoder_hidden_dim,device=device)

	def init_weights(self):
		#simple init method just init all parameters
		#by the uniform way
		init_method_map = {
		'uniform' : partial(init.uniform_,a = self.uniform_lower_bound, b = self.uniform_upper_bound),
		'normal' : partial(init.normal_, mean = self.normal_mean, std = self.normal_std),
		'xavier_normal' : partial(init.xavier_normal_, gain = self.xavier_normal_gain),
		'xavier_uniform' : partial(init.uniform_, gain = self.xavier_uniform_gain),
		}
		init_fn = init_method_map[self.init_method]
		for p in self.parameters():
			if p.data.dim() == 1 and 'normal' in self.init_method:
				init.uniform_(p.data)
			else:
				init_fn(p.data)

class RNNLMModel(nn.Module):
	'''
	rnn language model implement
	<<Recurrent neural network based language model>>

	'''
	def __init__(self,Config):
		super(RNNLMModel,self).__init__()
		self.embedding_dim = Config['lang_model_embed_dim']
		self.hidden_dim = Config['lang_model_hidden_dim']
		self.voc_size = Config['lang_model_voc_size']
		self.embed_layer = nn.Embedding(self.voc_size,self.embedding_dim)
		self.gru = nn.GRU(self.embedding_dim,self.hidden_dim,layer=2)
		self.projection = nn.Linear(self.embedding_dim,self.voc_size)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self,x,hidden):
		x = self.embed_layer(x)
		x,hidden = self.gru(x,hidden)
		x = self.projection(x)
		x = self.softmax(x)
		return x,hidden
	def get_score(self,x):
		pass
def main():
	model = AttnEncoderDecoder(Config)
	from data_util import load_data_into_parallel,PaddedTensorLanguageDataset
	src_file = '../data/nucle/nucle-train.tok.src'
	trg_file = '../data/nucle/nucle-train.tok.trg'
	src_trg_pair_langs = load_data_into_parallel(src_file, trg_file)
	train_dataset = PaddedTensorLanguageDataset(src_trg_pair_langs)
	from torch.utils.data import RandomSampler
	from torch.utils.data import DataLoader
	sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset,sampler=sampler,batch_size=16)
	reversed_train_dataset = PaddedTensorLanguageDataset(src_trg_pair_langs,reversed=True)
	reversed_train_dataloader = DataLoader(reversed_train_dataset,sampler=sampler,batch_size=16)
	model.init_weights()
	for batch in train_dataloader:
		input_seq,output_seq,input_mask,output_mask = batch
		hidden = model.init_hidden()
		print(hidden.shape)
		print(output_seq[0].dtype)
		ouput,hidden = model(output_seq[0],input_seq,hidden,input_mask,first_step=True)
		for i in range(1,Config['max_len']-1):
			output,hidden = model(output_seq[i],input_seq,hidden,input_mask)
			print(output.shape)
if __name__ == "__main__":
    main()
