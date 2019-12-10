import torch.nn as nn

class BiLSTMEncoderDecoder(nn.Module):

	def __init__(self, encode_size, decode_size):
		super(BiLSTMEncoderDecoder, self).__init__()
		self.hidden_size = 256
		self.encoder = nn.LSTM(encode_size, self.hidden_size, bias=True, batch_first=True, bidirectional=True)
		self.decoder = nn.LSTM(decode_size, self.hidden_size, bias=True, batch_first=True, bidirectional=True)
		self.final_linear = nn.Linear(self.hidden_size*2, 2, bias=True)

	def forward(self, X_encode, X_decode):
		_, (h_n, c_n) = self.encoder(X_encode)
		hidden_states, _ = self.decoder(X_decode, (h_n, c_n))
		output = self.final_linear(hidden_states)
		return output
