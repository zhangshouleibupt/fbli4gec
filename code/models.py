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
		return seq_out[:,:,self.encoder_hidden_dim:].transpose(0,1),h_last
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
        "notice the query shape is (batch,1,dim)"
        mask = mask.unsqueeze(2)
        K,V = mask * K, mask * V
        projection = self.attn_weight(K)
        attn_socre = torch.bmm(q,projection.transpose(1,2))
        attn_softmax_score = self.softmax(attn_socre)
        combine = torch.bmm(attn_softmax_score,V)
        return combine

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
        #print("input embed shape" ,input_token_embeded.shape)
        #use the t time step hidden as query
        encoder_seq_hiddens,h = self.encoder(input_seqs)
        #print(encoder_seq_hiddens.shape,h.shape)
        #print('-----------passed-------------')
        if first_step:
            decoder_rnn_out,hidden = self.decoder_rnn_layer(input_token_embeded,h)
        else:
            decoder_rnn_out,hidden = self.decoder_rnn_layer(input_token_embeded,hidden)
        #make the shape into (bathc_size,1,hidden_dim)
        decoder_rnn_out = decoder_rnn_out[:,:,self.decoder_hidden_dim:].transpose(0,1)
        #print("decoder rnn out",decoder_rnn_out.shape)
        attn_out = self.attn(decoder_rnn_out,encoder_seq_hiddens,encoder_seq_hiddens,mask)
        #print(attn_out.shape,decoder_rnn_out.shape)
        concat_fea = torch.cat((attn_out,decoder_rnn_out),dim=-1).squeeze()
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
    def __init__(self,Config):
        super(RNNLMModel,self).__init__()
        self.embedding_dim = Config['lang_model_embed_dim']
        self.hidden_dim = Config['lang_model_hidden_dim']
        self.voc_size = Config['lang_model_voc_size']
        self.use_bidirectional = Config['use_bidirectional']
        self.layers = Config['rnn_lang_model_layers']
        self.batch_size = Config['batch_size']
        self.embed_layer = nn.Embedding(self.voc_size,self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim,self.hidden_dim,num_layers=self.layers,bidirectional=self.use_bidirectional)
        self.projection = nn.Linear(self.embedding_dim,self.voc_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,hidden):
        x = self.embed_layer(x)
        x = x.unsqueeze(0)
        x,hidden = self.gru(x,hidden)
        x = x[:, :, self.hidden_dim:].transpose(0,1)
        x = self.projection(x)
        x = self.softmax(x)
        return x,hidden
    def init_hidden(self):
        b = 2 if self.use_bidirectional else 1
        return torch.zeros(b*self.layers,self.batch_size,self.hidden_dim)

    def get_score(self,x,use_batch=False,mask=None):
        if use_batch and mask is None:
            raise ValueError("in use batch mode mask needed")

        batch = x.shape[0] if use_batch else 1
        l = x.shape[1]
        b = 2 if self.use_bidirectional else 1
        hidden = torch.zeros(b*self.layers,batch,self.hidden_dim)
        score = torch.zeros(batch)
        with torch.no_grad():
            for i in range(l-1):
                x_step = x[:,i]
                out,hidden = self.forward(x_step,hidden)
                out = out.squeeze()
                idx = x[:,i+1].squeeze()
                tmp_score = torch.log(out[torch.arange(batch),idx])
                score += tmp_score
        return score

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
    rnn_lm_model = RNNLMModel(Config)
    for batch in train_dataloader:
        input_seq,output_seq,input_mask,output_mask = batch
        # hidden = model.init_hidden()
        # ouput,hidden = model(output_seq[:,0],input_seq,hidden,input_mask,first_step=True)
        # for i in range(1,Config['max_len']-1):
        #     output,hidden = model(output_seq[:,i],input_seq,hidden,input_mask)
        #     print(output.shape)
        score = rnn_lm_model.get_score(input_seq,use_batch=True)
        print(score)
if __name__ == "__main__":
    main()
