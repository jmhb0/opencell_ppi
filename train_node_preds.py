from dataset import OpencellPPI
from models import gcn
import torch
import ipdb
from sklearn.metrics import f1_score
from utils import FocalLoss
from typing import Literal


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
                        hyperparam_key=None):
    if model_name == "gcn":
        model = gcn.GCN(input_dim=data.x.shape[1],
                        hidden_dim=128,
                        output_dim=data.y.shape[1],
                        num_layers=3,
                        dropout=0.2)

    elif model_name == "linear":
        model = torch.nn.Linear(data.x.shape[1], data.y.shape[1])

    else:
        raise NotImplementedError()
    return model


def do_training_0(model_name='gcn',
                  test_split_frac=0.2,
                  test_split_method="cite_order",
                  features_type='dummy',
                  n_steps=10000):
    """
	First experiment is the most basic. 
	The task is a simple binary classification for whether therte 
	This label is in the dataset attribute `y_loc_nuclear`. 
	"""
    dataset = OpencellPPI(root="data",
                          features_type=features_type,
                          test_split_method=test_split_method,
                          test_split_frac=test_split_frac)
    print(
        "Training problem 0: binary classification of whether there is nuclear organelles"
    )
    print(f"Dataset attributes, test_split_method={test_split_method}, " \
     f"test_split_frac={test_split_frac}, features_type={features_type}")
    data = dataset[0]
    data.y = data.y_loc_nuclear
    output_dim = data.y.shape[1]
    data = data.cuda()
    model = get_model_from_name(model_name, data)
    is_graph_model = False if model_name == 'linear' else True
    model.reset_parameters()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_idx = torch.where(data.train_mask)[0]
    test_idx = torch.where(data.test_mask)[0]

    # get the baseline accuracy and print it
    labels_test = data.y[test_idx]
    labels_majority_classs = (labels_test.sum() / len(labels_test)).item()
    labels_majority_classs = max(labels_majority_classs,
                                 1 - labels_majority_classs)
    print(
        f"Accuracy of choosing the majority class {100*labels_majority_classs:.1f}%"
    )

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
            best_acc = max(acc, best_acc)
            best_f1 = max(f1, best_f1)
            acc_all.append(acc)
            loss_all.append(loss_scal)
            print(f"\rCurrent accuracy={100*acc:.1f}%, best_accuracy={100*best_acc:.1f}%, " \
             f"f1={f1:.2f}, best f1={best_f1:.2f}", end='', flush=True) # print with return carriage
    print()
    print(f"Best accuracy {100*best_acc:.1f}%")


def do_training_1(model='gcn',
                  test_split_frac=0.2,
                  test_split_method="cite_order",
                  features_type='dummy',
                  n_steps=10000):
    """
	Multi-class classification for the 
	This label is in the dataset attribute `y_loc_nuclear`. 
	"""
    dataset = OpencellPPI(root="data",
                          features_type=features_type,
                          test_split_method=test_split_method,
                          test_split_frac=test_split_frac)
    print(
        "Training problem 0: binary classification of whether there is nuclear organelles"
    )
    print(f"Dataset attributes, test_split_method={test_split_method}, " \
     f"test_split_frac={test_split_frac}, features_type={features_type}")
    data = dataset[0]
    data.y = data.y_loc
    data = data.cuda()

    output_dim = data.y.shape[1]
    model = gcn.GCN(input_dim=data.x.shape[1],
                    hidden_dim=64,
                    output_dim=output_dim,
                    num_layers=3,
                    dropout=0.2)
    # model = gcn.GCN(input_dim=data.x.shape[1], hidden_dim=128, output_dim=output_dim,
    #   num_layers=5, dropout=0.2)
    model.reset_parameters()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
    train_idx = torch.where(data.train_mask)[0]
    test_idx = torch.where(data.test_mask)[0]

    # get the class prevalences
    labels_test = data.y[test_idx]
    class_prevalence = labels_test.sum(0) / len(labels_test)
    # class_prevalence_test = labels_test[test_idx.cpu()].sum(0) / len(labels_test[test_idx.cpu()])

    print("Class prevalences (train+test):")
    for i in range(len(class_prevalence)):
        print(f" {dataset.id_to_loc[i]} {100*class_prevalence[i]:.1f}%")

    acc_all, loss_all = [], []
    best_acc, best_f1 = 0, 0
    for i in range(n_steps):
        loss_scal = train_step(model, data, train_idx, optimizer, loss_fn)
        if i % 100 == 0:
            acc_classwise, acc_mean, f1_classwise, f1_mean = eval(
                model, data, test_idx)
            acc, f1 = acc_mean, f1_mean
            best_acc = max(acc, best_acc)
            best_f1 = max(f1, best_f1)
            acc_all.append(acc)
            loss_all.append(loss_scal)
            print(f"\rCurrent accuracy={100*acc:.1f}%, best_accuracy={100*best_acc:.1f}%, " \
             f"f1={f1:.2f}, best f1={best_f1:.2f}", end='', flush=True) # print with return carriage
    print()
    print(f"Best accuracy {100*best_acc:.1f}%")
    print()

    print("Baseline accuracy")
    for i in range(len(class_prevalence)):
        print(f"{(1-class_prevalence[i]):.2f} ", end='')
    print()
    print("actual")
    for i in range(len(class_prevalence)):
        print(f"{(acc_classwise[i]):.2f} ", end='')
    print()
    print('f1')
    for i in range(len(class_prevalence)):
        print(f"{(f1_classwise[i]):.2f} ", end='')
    ipdb.set_trace()


if __name__ == "__main__":
    model_name = 'linear'
    
    for features_type in ['dummy', 'sequencelanguage', 'image']:
        print(f'model type: ', model_name)
        print(f"features_type ", features_type)
        do_training_0(model_name=model_name,
                      features_type=features_type,
                      test_split_frac=0.2,
                      n_steps=10000)
        print()
        print()

    # do_training_1(test_split_frac=0.2, n_steps=10000)

    ipdb.set_trace()
    pass