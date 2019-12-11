import torch.nn as nn

class LSTMEncoderDecoder(nn.Module):

	def __init__(self, encode_size, decode_size):
		super(LSTMEncoderDecoder, self).__init__()
		self.hidden_size = 256
		self.encoder = nn.LSTM(encode_size, self.hidden_size, bias=True, batch_first=True)
		self.decoder = nn.LSTM(decode_size, self.hidden_size, bias=True, batch_first=True)
		self.final_linear = nn.Linear(self.hidden_size, 2, bias=True)

	def forward(self, X_encode, X_decode):
		_, (h_n, c_n) = self.encoder(X_encode)
		hidden_states, _ = self.decoder(X_decode, (h_n, c_n))
		output = self.final_linear(hidden_states)
		return output
	

######################################################################
# Define the model
# ----------------
#


######################################################################
# In this tutorial, we train ``nn.TransformerEncoder`` model on a
# language modeling task. The language modeling task is to assign a
# probability for the likelihood of a given word (or a sequence of words)
# to follow a sequence of words. A sequence of tokens are passed to the embedding
# layer first, followed by a positional encoding layer to account for the order
# of the word (see the next paragraph for more details). The
# ``nn.TransformerEncoder`` consists of multiple layers of
# `nn.TransformerEncoderLayer <https://pytorch.org/docs/master/nn.html?highlight=transformerencoderlayer#torch.nn.TransformerEncoderLayer>`__. Along with the input sequence, a square
# attention mask is required because the self-attention layers in
# ``nn.TransformerEncoder`` are only allowed to attend the earlier positions in
# the sequence. For the language modeling task, any tokens on the future
# positions should be masked. To have the actual words, the output
# of ``nn.TransformerEncoder`` model is sent to the final Linear
# layer, which is followed by a log-Softmax function.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, nlabels, n_input1, n_input2, nhead, nhid, nlayers, dropout=0.5):
        # nlabels number of labels (2)
        # ninp number of features in input
        # nhead number of heads in attention model
        # nhid dimension of hidden state
        # nlayers number of layers in encoder and decoder
        
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer 
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model=n_input1, dropout=dropout)
        
        encoder_layers = TransformerEncoderLayer(n_input1, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        decoder_layer = nn.TransformerDecoderLayer(n_input2, nhead, nhid, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers)
#         memory = torch.rand(10, 32, 512)
#         tgt = torch.rand(20, 32, 512)
#         out = transformer_decoder(tgt, memory)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        #self.decoder = nn.Linear(ninp, ntoken)
        self.final_linear = nn.Linear(self.n_input2, nlabels, bias=True)

        #self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        #src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        enc_hidden_output = self.transformer_encoder(src, self.src_mask)
        print("enc output", enc_hidden_output)
        output = self.transformer_decoder(trg, enc_hidden_output)
        print("dec output", output)
        output = self.final_linear(output)
        print("output", output)
        
#         _, (h_n, c_n) = self.encoder(X_encode)
# 		hidden_states, _ = self.decoder(X_decode, (h_n, c_n))
# 		output = self.final_linear(hidden_states)
        
        return output

######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

