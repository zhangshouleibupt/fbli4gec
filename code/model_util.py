import torch
import numpy as np
import fairseq as fs
#ToDo lack of the shallow fusion
from models import RNNLMModel
from config import config

#return the conditonal score that calculate
#P(x_i | x_(< i))

def get_score(sentence,rnn_lm_model):
	return rnn_lm_model.get_score(sentence)

def fluent_score(sentence,rnn_model):
 
	sentence = sentence.split(' ')
	sen_len = len(sentence)
	if sen_len  <= 0:
		raise ValueError('sentence must not be empty')
	scores = [get_score(sentence[:i],rnn_lm_model) for i in range(1,sen_len+1)]
	H = sum(scores) / sen_len
	return 1 / (1 + H)

def beam_search(model,input_seqs,sos,beam_size=10):
	hidden = model.init_hidden()
	model.eval()
	max_len = config['max_len']
	#the first input would always <sos> token
	ouput = model(input_seqs,sos,hidden,mask=None,first_step=True)
	_,all_index = output.topk(ouput,beam_size)
	candidate = all_index
	all_condidate_seqs =[[i] for i in index]
	while eos not in index and current_len <= max_len:
		#recalculate the k candidate in next step
		index = [output.topk(token,1)[1] for token in candidate]
		candidate = all_index
		for i,index in enumerate(all_index):
			all_condidate_seqs[i].append(index)
	all_seqs = [torch.tensor(each_seq) for each_seq in all_condidate_seqs]
	return all_seqs