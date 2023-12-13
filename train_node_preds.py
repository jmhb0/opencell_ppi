from dataset import OpencellPPI
from models.gcn import GCN
from models.mlp import MLP 

import torch
import ipdb
from sklearn.metrics import f1_score
from utils import FocalLoss
from typing import Literal
import itertools
import pandas as pd
from pathlib import Path

def train_step(model,
			   data,
			   train_idx,
			   optimizer,
			   loss_fn,
			   is_graph_model=True):
	""" train one step of a PYG graph model """
	model.train()
	loss = 0
	optimizer.zero_grad()
	if is_graph_model:
		pred_logit = model(data.x, data.edge_index)
	else:
		pred_logit = model(data.x)
	pred_logit_train, label_train = pred_logit[train_idx], data.y[train_idx]
	loss = loss_fn(pred_logit_train, label_train.float())
	loss.backward()
	optimizer.step()
	return loss.item()

def eval(model, data, test_idx, is_graph_model=True):
	"""
	Differennt implementation for binary vs multilabel classification
	"""
	if is_graph_model:
		pred_logit = model(data.x, data.edge_index)
	else:
		pred_logit = model(data.x)
	pred_logit_test, label_test = pred_logit[test_idx], data.y[test_idx]
	pred_test = torch.sigmoid(pred_logit_test)
	pred_test = (pred_test >= 0.5).int()
	label_test, pred_test = label_test.detach().cpu().numpy(
	), pred_test.detach().cpu().numpy()

	# case 1: binary classification - single class
	if pred_logit.shape[1] == 1:
		acc = (pred_test == label_test).sum() / len(pred_test)  # float
		acc_mean = acc
		f1 = f1_score(label_test, pred_test)
		f1_mean = f1

	else:
		acc = (pred_test == label_test).sum(0) / len(pred_test)  # is an array
		acc_mean = acc.mean()  # balanced accuracy
		f1 = f1_score(label_test, pred_test, average=None)
		f1_mean = f1.mean()

	return acc, acc_mean, f1, f1_mean

def get_model_from_name(model_name: Literal["gcn", "linear"],
						data,
						hyperparam_key=0):
	input_dim = data.x.shape[1]
	output_dim = data.y.shape[1]
	if hyperparam_key == 0:
		kwargs = dict(hidden_dim=32, num_layers=2, dropout=0.2)
	elif hyperparam_key == 1:
		kwargs = dict(hidden_dim=128, num_layers=2, dropout=0.2)
	elif hyperparam_key == 2:
		kwargs = dict(hidden_dim=32, num_layers=3, dropout=0.2)
	elif hyperparam_key == 3:
		kwargs = dict(hidden_dim=128, num_layers=3, dropout=0.2)
	elif hyperparam_key == 4:
		kwargs = dict(hidden_dim=32, num_layers=5, dropout=0.2)
	elif hyperparam_key == 5:
		kwargs = dict(hidden_dim=128, num_layers=5, dropout=0.2)
	else: 
		raise ValueError(f"No hyperparam_key [{hyperparam_key}]")

	if 'gcn' in model_name:
		model = GCN(layer_type_name=model_name,
						input_dim=input_dim,
						output_dim=output_dim,
						**kwargs)

	elif model_name == "linear":
		model = torch.nn.Linear(input_dim, output_dim)

	elif model_name == "mlp":
		model = MLP(input_dim=input_dim, output_dim=output_dim, **kwargs)
	else:
		raise NotImplementedError()
	return model


def do_training(task: Literal["organelle_nuclear", "organelle_multiclass"],
				model_name: Literal['gcn', 'linear', 'mlp'],
				test_split_frac=0.2,
				test_split_method="cite_order",
				features_type='dummy',
				n_steps=10000,
				hyperparam_key=0,
				verbose=1,
				debug_multiclass_dataset=False):
	"""
	First experiment is the most basic. 
	The task is a simple binary classification for whether therte 
	This label is in the dataset attribute `y_loc_nuclear`. 
	"""
	dataset = OpencellPPI(root="data",
						  features_type=features_type,
						  test_split_method=test_split_method,
						  test_split_frac=test_split_frac)
	if verbose:
		print(
			"Training problem 0: binary classification of whether there is nuclear organelles"
		)
		print(f"Dataset attributes, test_split_method={test_split_method}, " \
		 f"test_split_frac={test_split_frac}, features_type={features_type}")

	data = dataset[0]
	if task == "organelle_nuclear":
		data.y = data.y_loc_nuclear
		is_binary = True
		is_multiclass = False
	elif task == "organelle_multiclass":
		data.y = data.y_loc
		is_binary = False
		is_multiclass = True
	else:
		raise ValueError()
	output_dim = data.y.shape[1]
	data = data.cuda()

	if debug_multiclass_dataset:
		data = debug_multiclass_dataset(model_name=model_name,
										test_split_frac=test_split_frac,
										test_split_method=test_split_method,
										features_type=features_type,
										n_steps=n_steps)

	model = get_model_from_name(model_name, data, hyperparam_key)
	is_graph_model = False if model_name in ('linear', 'mlp') else True
	model.reset_parameters()
	model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	if is_binary:
		loss_fn = torch.nn.BCEWithLogitsLoss()
	elif is_multiclass:
		loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
		# loss_fn = torch.nn.BCEWithLogitsLoss()
	else:
		raise

	train_idx = torch.where(data.train_mask)[0]
	test_idx = torch.where(data.test_mask)[0]

	# get the baseline accuracy and print it
	labels_test = data.y[test_idx]
	if is_binary:
		labels_majority_classs = (labels_test.sum() / len(labels_test)).item()
		labels_majority_classs = max(labels_majority_classs,
									 1 - labels_majority_classs)
		if verbose:
			print(
				f"Accuracy of choosing the majority class {100*labels_majority_classs:.1f}%"
			)
	elif is_multiclass:
		class_prevalence = labels_test.sum(0) / len(labels_test)
		# if verbose:
		# 	print("Class prevalences (train+test):")
		# 	for i in range(len(class_prevalence)):
		# 		print(
		# 			f" {dataset.id_to_loc[i]} {100*class_prevalence[i]:.1f}%")

	acc_all, loss_all = [], []
	best_acc, best_f1 = 0, 0
	for i in range(n_steps):
		loss_scal = train_step(model,
							   data,
							   train_idx,
							   optimizer,
							   loss_fn,
							   is_graph_model=is_graph_model)
		if i % 100 == 0:
			acc, acc_mean, f1, f1_mean = eval(model,
											  data,
											  test_idx,
											  is_graph_model=is_graph_model)
			if is_binary:
				best_acc = max(acc, best_acc)
				best_f1 = max(f1, best_f1)
				acc_all.append(acc)
				loss_all.append(loss_scal)
				if verbose:
					print(f"\rIt: {i}/{n_steps} Current accuracy={100*acc:.1f}%, best_accuracy={100*best_acc:.1f}%, " \
					 f"f1={f1:.2f}, best f1={best_f1:.2f}", end='', flush=True) # print with return carriage

			elif is_multiclass:
				acc_classwise, f1_classwise = acc, f1
				acc, f1 = acc_mean, f1_mean
				if acc_mean > best_acc:
					best_acc_classwise = acc_classwise
					best_f1_classwise = f1_classwise
				best_acc = max(acc_mean, best_acc)
				best_f1 = max(f1_mean, best_f1)
				acc_all.append(acc)
				loss_all.append(loss_scal)
				if verbose:
					print(f"\rIt: {i}/{n_steps} Current accuracy={100*acc:.1f}%, best_accuracy={100*best_acc:.1f}%, " \
					 f"f1={f1:.2f}, best f1={best_f1:.2f}", end='', flush=True) # print with return carriage

	if is_binary:
		if verbose:
			print(f"Best accuracy {100*best_acc:.1f}%")
		result = dict(best_acc=best_acc, baseline_acc=1-labels_majority_classs)

	elif is_multiclass:
		if verbose:
			print()
			print(f"Best accuracy {100*best_acc:.1f}%")
			print()

			print("Baseline accuracy")
			for i in range(len(class_prevalence)):
				print(f"{(1-class_prevalence[i]):.2f} ", end='')
			print()
			print("Accuracy best")
			for i in range(len(class_prevalence)):
				print(f"{(best_acc_classwise[i]):.2f} ", end='')
			print()
			print(f"mean(baseline_acc)={(1-class_prevalence).mean():.2f}")
			print(f"mean(best_acc)={best_acc_classwise.mean():.2f}")

			print('f1 best')
			for i in range(len(class_prevalence)):
				print(f"{(best_f1_classwise[i]):.2f} ", end='')
			print(f"mean(best_f1)={(1-class_prevalence[i]).mean():.2f}")
			print(f"mean(best_f1)={best_f1_classwise.mean()}")

		result = dict(best_acc=best_acc,
					  best_f1=best_f1,
					  class_prevalence=class_prevalence,
					  best_acc_classwise=best_acc_classwise,
					  best_f1_classwise=best_f1_classwise,
					  baseline_acc=(1-class_prevalence).mean())
	else:
		raise

	return result


def debug_multiclass_dataset():
	""" 
	Wanted to check that the multiclass classification task was properly 
	implemented. So here it is.
	"""
	dataset = OpencellPPI(root="data",
						  features_type=features_type,
						  test_split_method=test_split_method,
						  test_split_frac=test_split_frac)
	print("Training problem 1: multiclass organelle classification")
	print(f"Dataset attributes, test_split_method={test_split_method}, " \
	 f"test_split_frac={test_split_frac}, features_type={features_type}")
	data = dataset[0]
	data.y = data.y_loc
	data = data.cuda()

	# now overwrite the data object to this synthetic one
	print("*" * 80)
	print("Debugging with synthetic multiclass dataset")
	print("*" * 80)
	from sklearn.datasets import make_multilabel_classification
	n_classes = data.y.shape[1]
	n_samples = len(data.x)
	X, y = make_multilabel_classification(n_samples=n_samples,
										  n_features=10,
										  n_classes=n_classes,
										  n_labels=3,
										  allow_unlabeled=False,
										  random_state=42)
	data.x = torch.from_numpy(X).float()
	data.y = torch.from_numpy(y).float()
	data = data.cuda()


def do_experiment():
	"""
	"""
	# experimental conditions
	task_ = ["organelle_nuclear", "organelle_multiclass"]
	test_split_method_ = ['random', 'cite_order']
	test_split_frac_ = [0.2, 0.1]

	# modeling parameters
	features_type_ = ['dummy', 'image', 'sequencelanguage']
	n_steps = 1000
	model_name_ = ['linear', 'mlp'] + list(GCN.layer_type_lookup.keys())
	model_name_ = ['gcn-conv']
	hyperparam_key_ = list(range(6))
	hyperparam_key_ = list(range(6))

	# `all_results` is a dict for all results. `df_lookup` will map keys in `all_results` to params
	idx = 0
	all_results = {}
	results_dir = Path("./results")
	results_dir.mkdir(exist_ok=True)

	df_lookup = pd.DataFrame(columns=[
		'idx',
		'task',
		'test_split_method',
		'test_split_frac',
		'features_type',
		'model_name',
		'hyperparam_key',
	])

	# loop over experiments
	for i, (task, test_split_method, test_split_frac,
			features_type) in enumerate(
				itertools.product(task_, test_split_method_, test_split_frac_,
								  features_type_)):

		# loop over modeling parameters, recording which model has best accuracy
		this_exp_best_accs = []
		this_exp_idxs = []
		for j, (model_name, hyperparam_key) in enumerate(
				itertools.product(model_name_, hyperparam_key_)):
			print(idx, end=" ", flush=True)
			kwargs = dict(
				task=task,
				test_split_method=test_split_method,
				test_split_frac=test_split_frac,
				features_type=features_type,
				model_name=model_name,
				hyperparam_key=hyperparam_key,
			)
			# log all experiments
			result = do_training(**kwargs, n_steps=n_steps, verbose=0)
			all_results[idx] = result
			kwargs['idx'] = idx
			kwargs['best_acc'] = result['best_acc']
			for k, v in kwargs.items(): kwargs[k] = [v]
			df_lookup = pd.concat((df_lookup, pd.DataFrame(kwargs)), ignore_index=True)

			# log the accuracy scores for just this experimental setting 
			this_exp_best_accs.append(result['best_acc'])
			this_exp_idxs.append(idx)
			
			idx += 1
		torch.save((all_results, df_lookup), results_dir / "experiment_results.pt")



if __name__ == "__main__":
	# do_experiment()
	model_name = 'gcn-conv'  # ('linear','mlp','gcn-conv','gcn-sage','gcn-gat','gcn-gin','gcn-transformer')
	model_name = 'linear'  # ('linear','mlp','gcn-conv','gcn-sage','gcn-gat','gcn-gin','gcn-transformer')
	task = "organelle_nuclear"  # ("organelle_nuclear", "organelle_multiclass")
	task = "organelle_multiclass"  # ("organelle_nuclear", "organelle_multiclass")
	n_steps = 1000  # 10000
	test_split_method = "cite_order"  # ("random","cite_order")
	# test_split_method = "random"  # ("random","cite_order")
	test_split_frac = 0.2
	hyperparam_key = 5

	features_type = 'image'
	# features_type = 'sequencelanguage'
	# for features_type in ['dummy', 'image', 'sequencelanguage']:
	for model_name in ['linear', 'mlp'] + list(GCN.layer_type_lookup.keys()):
		# for features_type in ['image', 'sequencelanguage']:
		print(f'task: ', task)
		print(f'model type: ', model_name)
		print(f"features_type ", features_type)
		do_training(task=task,
					model_name=model_name,
					features_type=features_type,
					test_split_method=test_split_method,
					test_split_frac=test_split_frac,
					n_steps=n_steps, 
					hyperparam_key=hyperparam_key,)
		print()
		print()

	ipdb.set_trace()
	pass