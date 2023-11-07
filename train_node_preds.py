from dataset import OpencellPPI
from models import gcn
import torch
import ipdb 

def train(model, data, train_idx, optimizer, loss_fn):
	""" train one step """
	model.train()
	loss = 0
	optimizer.zero_grad()
	pred_logit = model(data.x, data.edge_index)
	pred_logit_train, label_train = pred_logit[train_idx], data.y[train_idx]
	loss = loss_fn(pred_logit_train, label_train.float())
	loss.backward()
	optimizer.step()
	return loss.item()

def eval(model, data, test_idx):
	pred_logit = model(data.x, data.edge_index)
	pred_logit_test, label_test = pred_logit[test_idx], data.y[test_idx]
	pred_test = torch.sigmoid(pred_logit_test)
	pred_test = (pred_test>=0.5).int()

	acc = (pred_test==label_test).sum() / len(pred_test)
	return acc.item()

if __name__=="__main__":
	dataset = OpencellPPI(root="data", features_type="dummy", 
		test_split_method="cite_order", test_split_frac=0.2)
	data = dataset[0]
	data.y = data.y_loc_nuclear
	data = data.cuda()

	model = gcn.GCN(input_dim=data.x.shape[1], hidden_dim=64, output_dim=1, 
		num_layers=3, dropout=0.2)
	model.reset_parameters()
	model.cuda()


	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	loss_fn = torch.nn.BCEWithLogitsLoss()
	train_idx = torch.where(data.train_mask)[0]
	test_idx = torch.where(data.test_mask)[0]

	acc_all, loss_all = [], []
	for i in range(10000):
		loss_scal = train(model, data, train_idx, optimizer, loss_fn)
		if i%100 == 0:
			acc = eval(model, data, test_idx)
			acc_all.append(acc)
			loss_all.append(loss_scal)
			print(acc)

	ipdb.set_trace()
	pass