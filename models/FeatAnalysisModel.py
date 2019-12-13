import torch.nn as nn

class FeatAnalysisModel(nn.Module):

	def __init__(self, input_size):
		super(FeatAnalysisModel, self).__init__()
		self.input_size = input_size
		self.linear = nn.Linear(self.input_size, 2, bias=True)

	def forward(self, X):
		output = self.linear(X)
		return output
	