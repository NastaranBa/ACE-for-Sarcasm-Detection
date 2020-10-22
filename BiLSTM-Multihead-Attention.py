# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import torch
import numpy as np
import torch.nn as nn
from sklearn.utils import shuffle
from torch.autograd import Variable
from models.BaseModel import BaseModel

class LSTM_attention(nn.Module):
    ''' Compose with two layers '''
    def __init__(self,lstm_hidden,bilstm_flag,data):
        super(LSTM_attention, self).__init__()

        self.lstm = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=1, batch_first=True, bidirectional=bilstm_flag)
        #self.slf_attn = multihead_attention(data.HP_hidden_dim,num_heads = data.num_attention_head, dropout_rate=data.HP_dropout)
        self.label_attn = multihead_attention(data.HP_hidden_dim, num_heads=data.num_attention_head,dropout_rate=data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.gpu = data.HP_gpu
        if self.gpu:
            self.lstm =self.lstm.cuda()
            self.label_attn = self.label_attn.cuda()


    def forward(self,lstm_out,label_embs,word_seq_lengths,hidden):
        lstm_out = pack_padded_sequence(input=lstm_out, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        lstm_out, hidden = self.lstm(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))
        # lstm_out (seq_length * batch_size * hidden)
        label_attention_output = self.label_attn(lstm_out, label_embs, label_embs)
        # label_attention_output (batch_size, seq_len, embed_size)
        lstm_out = torch.cat([lstm_out, label_attention_output], -1)
        return lstm_out

class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0, gpu=True, causality=False):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        if self.gpu:
            self.Q_proj = self.Q_proj.cuda()
            self.K_proj = self.K_proj.cuda()
            self.V_proj = self.V_proj.cuda()


        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]
        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        # Dropouts
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        if last_layer == True:
            return outputs
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries

        return outputs
class LSTMAttention(torch.nn.Module):
    def __init__(self,opt):

        super(LSTMAttention, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
  
        self.num_layers = opt.lstm_layers
        #self.bidirectional = True
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, batch_first=True,num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.mean = opt.__dict__.get("lstm_mean",True) 
        self.attn_fc = torch.nn.Linear(opt.embedding_dim, 1)
    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size= self.batch_size
        
        if self.use_gpu:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2))
        return (h0, c0)


    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state],1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)
    # end method attention
    

    def forward(self, X):
        embedded = self.word_embeddings(X)
        hidden= self.init_hidden(X.size()[0]) #
        rnn_out, hidden = self.bilstm(embedded, hidden)
        h_n, c_n = hidden
        attn_out = self.attention(rnn_out, h_n)
        logits = self.hidden2label(attn_out)
        return logits