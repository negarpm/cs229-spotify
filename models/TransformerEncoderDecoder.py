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
import torch
import copy
#from .. import functional as F

class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", kdim=30, vdim=68):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=30, vdim=30)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=68, vdim=68)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        #print("tgt, mem", tgt.shape, memory.shape)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)
        
# In this tutorial, we train ``nn.TransformerEncoder`` model on a
# language modeling task. The language modeling task is to assign a
# probability for the likelihood of a given word (or a sequence of words)
# to follow a sequence of words. A sequence of tokens are passed to the embedding
# layer first, followed by a positional encoding layer to account for the order
# of the word (

# class MultiheadAttention(nn.Module):
#     r"""Allows the model to jointly attend to information
#     from different representation subspaces.
#     See reference: Attention Is All You Need
#     .. math::
#         \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
#         \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
#     Args:
#         embed_dim: total dimension of the model.
#         num_heads: parallel attention heads.
#         dropout: a Dropout layer on attn_output_weights. Default: 0.0.
#         bias: add bias as module parameter. Default: True.
#         add_bias_kv: add bias to the key and value sequences at dim=0.
#         add_zero_attn: add a new batch of zeros to the key and
#                        value sequences at dim=1.
#         kdim: total number of features in key. Default: None.
#         vdim: total number of features in key. Default: None.
#         Note: if kdim and vdim are None, they will be set to embed_dim such that
#         query, key, and value have the same number of features.
#     Examples::
#         >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
#         >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
#     """
#     __annotations__ = {
#         'bias_k': torch._jit_internal.Optional[torch.Tensor],
#         'bias_v': torch._jit_internal.Optional[torch.Tensor],
#     }
#     __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
#         super(MultiheadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         if self._qkv_same_embed_dim is False:
#             self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
#             self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
#             self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
#             self.register_parameter('in_proj_weight', None)
#         else:
#             self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)

#         if bias:
#             self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

#         if add_bias_kv:
#             self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
#             self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         self._reset_parameters()

#     def _reset_parameters(self):
#         if self._qkv_same_embed_dim:
#             xavier_uniform_(self.in_proj_weight)
#         else:
#             xavier_uniform_(self.q_proj_weight)
#             xavier_uniform_(self.k_proj_weight)
#             xavier_uniform_(self.v_proj_weight)

#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             xavier_normal_(self.bias_v)

#     def __setstate__(self, state):
#         super(MultiheadAttention, self).__setstate__(state)

#         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#         if 'self._qkv_same_embed_dim' not in self.__dict__:
#             self._qkv_same_embed_dim = True

#     def forward(self, query, key, value, key_padding_mask=None,
#                 need_weights=True, attn_mask=None):
#         # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
#         r"""
#     Args:
#         query, key, value: map a query and a set of key-value pairs to an output.
#             See "Attention Is All You Need" for more details.
#         key_padding_mask: if provided, specified padding elements in the key will
#             be ignored by the attention. This is an binary mask. When the value is True,
#             the corresponding value on the attention layer will be filled with -inf.
#         need_weights: output attn_output_weights.
#         attn_mask: mask that prevents attention to certain positions. This is an additive mask
#             (i.e. the values will be added to the attention layer).
#     Shape:
#         - Inputs:
#         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
#           the embedding dimension.
#         - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
#         - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
#         - Outputs:
#         - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
#           E is the embedding dimension.
#         - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
#           L is the target sequence length, S is the source sequence length.
#         """
#         if not self._qkv_same_embed_dim:
#             return nn.F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, use_separate_proj_weight=True,
#                 q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
#                 v_proj_weight=self.v_proj_weight)
#         else:
#             return nn.F.multi_head_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask)

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

    def __init__(self, nlabels, n_input1, n_input2, nhead, nhid, nlayers, dropout=0.5, kdim=68, vdim=68):
        # nlabels number of labels (2)
        # ninp number of features in input
        # nhead number of heads in attention model
        # nhid dimension of hidden state
        # nlayers number of layers in encoder and decoder
        
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder 
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model=n_input1, dropout=dropout)
        
        encoder_layers = TransformerEncoderLayer(n_input1, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        decoder_layer = TransformerDecoderLayer(n_input2, nhead, nhid, dropout, "relu", 30, 68)
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
        #print("enc output", enc_hidden_output.shape)
        output = self.transformer_decoder(trg, enc_hidden_output)
        #print("dec output", output.shape)
        output = self.final_linear(output)
        #print("output", output.shape)
        
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

