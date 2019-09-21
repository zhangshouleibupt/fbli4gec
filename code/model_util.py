import torch
import numpy as np
import fairseq as fs

def get_score(sentence):

def fluent_score(sentence):
	sentence = sentence.split(' ')
	sen_len = len(sentence)
	if sen_len  <= 0:
		raise ValueError('sentence must not be empty')
