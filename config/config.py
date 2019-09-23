config = {
	'input_voc_size' : 100*1000,
	'output_voc_size' : 50 * 1000,
	'init_method' : 'uniform',
	'uniform_lower_bound' : -0.1,
	'uniform_upper_bound' : 0.1,
	'attn_method' : 'general',
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
	'dropout_prob' : 0.1,
	'rnn_type' : 'gru',
	'normal_mean' : 0,
	'normal_std' : 1,
	'xavier_normal_gain' : 1,
	'xavier_uniform_gain' : 1,
	
}
