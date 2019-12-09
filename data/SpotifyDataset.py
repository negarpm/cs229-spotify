from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import math


class SpotifyDataset(Dataset):

	def __init__(self, session_csv, track_feat_csv):
		self.sessions = pd.read_csv(session_csv)
		self.session_ids = self.sessions["session_id"].unique()
		self.track_feats = pd.read_csv(track_feat_csv)

	def __len__(self):
		return len(self.session_ids)

	'''
	Gets the data for the given index.
	Returns:
		- encode_state, shape: (num_first_half, 67), desc: DataFrame representing the first half of the tracks in our session
		- decode_state, shape: (num_second_half, 29), desc: DataFrame representing the last half of the tracks in our session
		- labels, shape: (num_second_half, 1), desc: DataFrame representing the skip_2 labels for the last half of the tracks in our session
	'''
	def __getitem__(self, idx):
		session_id = self.session_ids[idx]
		example = self.sessions[ self.sessions["session_id"] == session_id ].drop(["session_id"], axis=1)
		encode_state = self.get_encode(example)
		decode_state, labels = self.get_decode_and_label(example)
		return encode_state, decode_state, labels

	'''
	Extracts and pads features being passed into the encoder LSTM
	'''
	def get_encode(self, example):
		num_encode = math.ceil(len(example) / 2)
		encode_data = example.iloc[ 0:num_encode, : ]
		encode_state = encode_data.merge(self.track_feats, left_on='track_id_clean', right_on='track_id', how='left').drop(["track_id_clean", "track_id"], axis=1)
		return torch.tensor(encode_state.values, dtype=torch.float32)

	'''
	Extracts and pads features being passed into the decoder LSTM and their skip_2 labels
	'''
	def get_decode_and_label(self, example):
		num_encode = math.ceil(len(example) / 2)
		decode_data = example.iloc[ num_encode: , : ]
		decode_ids = pd.DataFrame(decode_data["track_id_clean"])
		decode_state = decode_ids.merge(self.track_feats, left_on='track_id_clean', right_on='track_id', how='left').drop(["track_id_clean", "track_id"], axis=1)
		labels = decode_data["skip_2"].to_frame()
		return torch.tensor(decode_state.values, dtype=torch.float32), torch.tensor(labels.values, dtype=torch.long)
