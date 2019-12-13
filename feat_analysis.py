if __name__ == "__main__":

	import torch
	import torch.nn as nn
	from torch.utils.data import DataLoader
	from models.FeatAnalysisModel import FeatAnalysisModel

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	WEIGHT_PATH = "./models/feat_analysis_decay_X.pt"

	from data.FeatAnalysisDataset import FeatAnalysisDataset

	train_set = FeatAnalysisDataset("./data/train_data_20.csv", "./data/track_feats.csv")
	test_set = FeatAnalysisDataset("./data/test_data_20.csv", "./data/track_feats.csv")


	datasets = {"train": train_set,
				"test": test_set}

	model = FeatAnalysisModel(input_size=699).to(device)

	loss_fn = nn.CrossEntropyLoss()
	batch_size = 64
	learning_rate = 1e-3

	weight_maas = []
	weight_losses = []
	weight_decay_schedule = [0, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
	for w, weight_decay in enumerate(weight_decay_schedule):
		### TRAINING BLOCK
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
		
		print("Training on L1 penalty of {}...".format(weight_decay))
		total_loss = []
		dataloader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=4)
		for i, (X, y) in enumerate(dataloader):
			if i % 10 == 0:
				print("Calculating batch {} / {}".format(i, len(dataloader)))
			X, y = X.to(device), y.to(device)
			model.train()
			scores = model(X)
			cross_ent_loss = loss_fn(scores, y)
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