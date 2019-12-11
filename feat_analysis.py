if __name__ == "__main__":

	def evaluate_batch_mean_average_accuracy(y_truth, y_pred):
		matches = [x == y for (x,y) in zip(y_truth, y_pred)]
		maas = []
		for batch in range(0, len(matches), 10):
			num_correct = 0
			summ = 0
			for i in range(0, 10):
				if matches[batch+i] == 1:
					num_correct += 1
					summ += (num_correct / (i+1)) / 10
			maas.append(summ)
		return sum(maas) / len(maas)

	import torch
	import torch.nn as nn
	from torch.utils.data import DataLoader
	from models.LSTMEncoderDecoder import LSTMEncoderDecoder

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	WEIGHT_PATH = "./models/weight_decay_X.pt"

	from data.SpotifyDataset import SpotifyDataset

	train_set = SpotifyDataset("./data/train_data_20.csv", "./data/track_feats.csv")
	test_set = SpotifyDataset("./data/test_data_20.csv", "./data/track_feats.csv")

	datasets = {"train": train_set,
				"test": test_set}

	model = LSTMEncoderDecoder(encode_size=67, decode_size=29).to(device)

	loss_fn = nn.CrossEntropyLoss()
	batch_size = 64
	learning_rate = 1e-2

	weight_maas = []
	weight_losses = []
	weight_decay_schedule = [0, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
	for w, weight_decay in enumerate(weight_decay_schedule):
		### TRAINING BLOCK
		model.load_state_dict(torch.load("./models/lstm_weights.pt"))
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
		
		print("Training on L1 penalty of {}...".format(weight_decay))
		total_loss = []
		dataloader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=2)
		for i, (X_encode, X_decode, y) in enumerate(dataloader):
			if i == 100:
				break
			if i % 10 == 0:
				print("Calculating batch {} / {}".format(i, 100))
			X_encode, X_decode, y = X_encode.to(device), X_decode.to(device), y.to(device)
			model.train()
			scores = model(X_encode, X_decode).flatten(start_dim=0, end_dim=1)
			labels = y.flatten(start_dim=0, end_dim=1).squeeze()
			cross_ent_loss = loss_fn(scores, labels)
			l1_reg = torch.tensor(0.0).to(device)
			for param in model.parameters():
				l1_reg += torch.norm(param, p=1)
			loss = cross_ent_loss + weight_decay * l1_reg
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss.append(loss)

		epoch_loss = sum(total_loss) / len(total_loss)
		weight_losses.append(epoch_loss.item())
		print("L1 penalty {} avg. batch loss: {}".format(weight_decay, epoch_loss))

		torch.save(model.state_dict(), WEIGHT_PATH.replace("X", str(w)))

		### TESTING BLOCK
		with torch.no_grad():
			dataloader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=2)
			total_maa = []
			for i, (X_encode, X_decode, labels) in enumerate(dataloader):
				if i == 25:
					break
				X_encode, X_decode, y = X_encode.to(device), X_decode.to(device), y.to(device)
				labels = y.flatten(start_dim=0, end_dim=1).squeeze()
				scores = model(X_encode, X_decode).flatten(start_dim=0, end_dim=1)
				y_pred = torch.argmax(scores, dim=1)
				maa = evaluate_batch_mean_average_accuracy(labels, y_pred)
				total_maa.append(maa)
			test_maa = sum(total_maa) / len(total_maa)
			weight_maas.append(test_maa)
			print("Average batch MAA over test set: {}".format(test_maa))
		print("Loss across L1 penalties: ")
		print(weight_losses)
		print("MAA across L1 penalties: ")
		print(weight_maas)
		print()
