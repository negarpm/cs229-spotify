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
	MODEL_PATH = "./models/lstm_encoder_decoder_1.pt"

	from data.SpotifyDataset import SpotifyDataset

	train_set = SpotifyDataset("./data/train_data_20.csv", "./data/track_feats.csv")
	val_set = SpotifyDataset("./data/val_data_20.csv", "./data/track_feats.csv")
	test_set = SpotifyDataset("./data/test_data_20.csv", "./data/track_feats.csv")

	datasets = {"train": train_set,
	            "val": val_set,
	            "test": test_set}

	### TRAINING BLOC:
	model = LSTMEncoderDecoder(encode_size=67, decode_size=29).to(device)
	loss_fn = nn.CrossEntropyLoss()

	batch_size = 64
	learning_rate = 1e-3
	weight_decay = 2.5e-3
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

	num_epochs = 3
	best_maa = 0
	batch_losses = []
	epoch_losses = []
	for epoch in range(num_epochs):
	    print('Epoch {}/{}'.format(epoch+1, num_epochs))
	    for phase in ['train', 'val']:
	        print("Running {} phase...".format(phase))
	        total_maa = []
	        total_loss = []
	        dataloader = DataLoader(datasets[phase], batch_size=batch_size, shuffle=True, num_workers=2)
	        for i, (X_encode, X_decode, y) in enumerate(dataloader):
	            if i % 10 == 0:
	                print("Calculating batch {} / {}".format(i, len(dataloader)))
	            X_encode, X_decode, y = X_encode.to(device), X_decode.to(device), y.to(device)
	            if phase == 'train':
	                model.train()
	                scores = model(X_encode, X_decode).flatten(start_dim=0, end_dim=1)
	                labels = y.flatten(start_dim=0, end_dim=1).squeeze()
	                loss = loss_fn(scores, labels)
	                loss.backward()
	                optimizer.step()
	                optimizer.zero_grad()
	                total_loss.append(loss)
	                batch_losses.append(loss.item())
	            else:
	                model.eval()
	                scores = model(X_encode, X_decode).flatten(start_dim=0, end_dim=1)
	                labels = y.flatten(start_dim=0, end_dim=1).squeeze()
	                y_pred = torch.argmax(scores, dim=1)
	                maa = evaluate_batch_mean_average_accuracy(labels, y_pred)
	                total_maa.append(maa)
	        if phase == 'train':
	            epoch_loss = sum(total_loss) / len(total_loss)
	            print("Epoch {} Avg. Loss: {}".format(epoch, epoch_loss))
	            epoch_losses.append(epoch_loss.item())
	        else:
	            epoch_maa = sum(total_maa) / len(total_maa)
	            print("Epoch {} Avg. MAA: {}, Best MAA: {}".format(epoch, epoch_maa, best_maa))
	            if epoch_maa > best_maa:
	                torch.save(model.state_dict(), MODEL_PATH)
	                best_maa = epoch_maa
	    print()
	print()
	print("List of avg. loss across epochs: ")
	print(epoch_losses)
	print("List of losses across batches: ")
	print(batch_losses)

	### TESTING BLOCK
	test_model = LSTMEncoderDecoder(encode_size=67, decode_size=29).to(device)
	test_model.load_state_dict(torch.load(MODEL_PATH))

	with torch.no_grad():
	    dataloader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=2)
	    total_maa = []
	    for i, (X_encode, X_decode, labels) in enumerate(dataloader):
	        X_encode, X_decode, y = X_encode.to(device), X_decode.to(device), y.to(device)
	        labels = y.flatten(start_dim=0, end_dim=1).squeeze()
	        scores = test_model(X_encode, X_decode).flatten(start_dim=0, end_dim=1)
	        y_pred = torch.argmax(scores, dim=1)
	        maa = evaluate_batch_mean_average_accuracy(y, y_pred)
	        total_maa.append(maa)
	    print("Average batch MAA over test set: {}".format(sum(total_maa) / len(total_maa)))
