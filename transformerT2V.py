import json
import time
import os
import random
import joblib
import math

import numpy as np
import pandas as pd

from datetime import date
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataProcess


class TransformerDiscreteT2V(nn.Module):
	def __init__(
		self,
		num_featuresin,
		num_featuresout,
		num_frequency,
		max_seq_len,
		dim_model,
		num_heads,
		num_encoder_layers,
		num_decoder_layers,
		dim_feedforward,
		dropout_p,
	):
		super().__init__()

		# INFO
		self.model_type = "Transformer"
		self.dim_model = dim_model
		# LAYERS
		self.t2v_src = Time2Vec(num_frequency=num_frequency, 
								num_vars=num_featuresin, 
								max_seq_len=max_seq_len)
		self.t2v_tgt = Time2Vec(num_frequency=num_frequency, 
								num_vars=num_featuresout, 
								max_seq_len=max_seq_len)
		self.src_embedding = nn.Linear(num_featuresin+num_frequency, dim_model)
		self.tgt_embedding = nn.Linear(num_featuresout+num_frequency, dim_model)
		self.transformer = nn.Transformer(d_model=dim_model,
											nhead=num_heads,
											num_encoder_layers=num_encoder_layers,
											num_decoder_layers=num_decoder_layers,
											dim_feedforward=dim_feedforward,
											dropout=dropout_p)
		self.out = nn.Linear(dim_model, num_featuresout-1)

		self.positionalEncoding = True
		if self.positionalEncoding:
			self.positional_encoder = PositionalEncoding(dim_model=dim_model, 
														dropout_p=dropout_p, 
														max_len=5000)

	def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
		# Src size must be (batch_size, sequence_length, src)
		# Tgt size must be (batch_size, sequence_length, tgt)
		src = self.t2v_src(src)
		tgt = self.t2v_tgt(tgt)
		# Src size must be (batch_size, src, sequence_length)
		# Tgt size must be (batch_size, tgt, sequence_length)
		src = self.src_embedding(src) * math.sqrt(self.dim_model)
		tgt = self.tgt_embedding(tgt) * math.sqrt(self.dim_model)
		# Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
		if self.positionalEncoding:
			src = self.positional_encoder(src)
			tgt = self.positional_encoder(tgt)
		# to obtain size (sequence_length, batch_size, dim_model),
		src = src.permute(1,0,2)
		tgt = tgt.permute(1,0,2)
		# Transformer blocks - Out size = (sequence_length, batch_size, num_tokens)
		tgt_mask=None
		transformer_out = self.transformer(src, 
											tgt, 
											tgt_mask=tgt_mask, 
											src_key_padding_mask=src_pad_mask, 
											tgt_key_padding_mask=tgt_pad_mask)
		transformer_out = transformer_out[0,:,:].reshape(1, transformer_out.shape[1], transformer_out.shape[2])
		out = self.out(transformer_out)
		return out.permute(1,0,2)

	def get_tgt_mask(self, size) -> torch.tensor:
		mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
		mask = mask.float()
		mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
		mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
		return mask

	def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
		return (matrix == pad_token)



class Time2Vec(nn.Module):
	def __init__(self, num_frequency, num_vars, max_seq_len):
		super(Time2Vec, self).__init__()
		self.num_frequency = num_frequency
		self.num_vars = num_vars
		self.max_seq_len = max_seq_len
		self.trend_weight = nn.Parameter(torch.Tensor(1))
		self.trend_bias = nn.Parameter(torch.Tensor(1))
		# Periodic weights and bias; will be initialized in forward
		self.periodic_weight = nn.Parameter(torch.Tensor(1, self.num_frequency))
		self.periodic_bias = nn.Parameter(torch.Tensor(self.max_seq_len, self.num_frequency))
		# initialize weights
		nn.init.uniform_(self.trend_weight)
		nn.init.uniform_(self.trend_bias)
		nn.init.uniform_(self.periodic_weight)
		nn.init.uniform_(self.periodic_bias)
	def forward(self, inputs):
		# Split inputs into x and t
		x = inputs[:, :, :self.num_vars-1]
		t = inputs[:, :, self.num_vars-1:]
		# Trend component
		trend_component = self.trend_weight * t + self.trend_bias
		# Periodic component
		periodic_component = torch.sin(torch.matmul(t, self.periodic_weight) + self.periodic_bias) # this periodic bias might be affected by padding
		# alternatively, time encode and then pad...
		# Concatenate trend and periodic components
		t_encoded = torch.cat([trend_component, periodic_component], dim=-1)
		# Concatenate x and t_encoded
		output = torch.cat([x, t_encoded], dim=-1)
		return output



class PositionalEncoding(nn.Module):
	def __init__(self, dim_model, dropout_p, max_len):
		super().__init__()        
		# Info
		self.dropout = nn.Dropout(dropout_p)
		# Encoding - From formula
		pos_encoding = torch.zeros(max_len, dim_model)
		positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
		division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
		# PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
		pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
		# PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
		pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
		# Saving buffer (same as parameter without gradients needed)
		pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
		self.register_buffer("pos_encoding",pos_encoding)
	def forward(self, token_embedding: torch.tensor) -> torch.tensor:
		# Residual connection + pos encoding
		return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


