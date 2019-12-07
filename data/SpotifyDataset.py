from torch.utils.data import Dataset
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
		- encode_state, shape: (10, 67), desc: pd DataFrame representing the ten states to put into encoder
		- decode_state, shape: (10, 29), desc: pd DataFrame representing the ten states to put into decoder
		- labels, shape: (10, 1), desc: pd DataFrame representing the skip_2 labels for the decoder states
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
		for _ in range(10 - len(encode_state)):
			zero_pad = pd.DataFrame(np.zeros((1, encode_state.shape[1])), columns=encode_state.columns.values)
			encode_state = pd.concat([zero_pad, encode_state]).reset_index(drop=True)
		return encode_state

	'''
	Extracts and pads features being passed into the decoder LSTM and their skip_2 labels
	'''
	def get_decode_and_label(self, example):
		num_encode = math.ceil(len(example) / 2)
		decode_data = example.iloc[ num_encode: , : ]
		decode_ids = pd.DataFrame(decode_data["track_id_clean"])
		decode_state = decode_ids.merge(self.track_feats, left_on='track_id_clean', right_on='track_id', how='left').drop(["track_id_clean", "track_id"], axis=1)
		labels = decode_data["skip_2"].to_frame()
		for _ in range(10 - len(decode_state)):
			zero_pad = pd.DataFrame(np.zeros((1, decode_state.shape[1])), columns=decode_state.columns.values)
			decode_state = decode_state.append(zero_pad)
			labels = labels.append(pd.DataFrame(np.zeros((1, 1)), columns=labels.columns.values))
		return decode_state, labels
