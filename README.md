Code for [this blog post](https://medium.com/@jamesmhburgess1/protein-interaction-network-prediction-in-opencell-with-graph-neural-nets-20bf739c2971) as part of the cs224w class on machine learning on graphs. There is also a [Colab version](https://colab.research.google.com/drive/1gmU0_-__qHqJo-dEcz_v1GVl7nLEjpT0) of the code.

## Setup:
```
pip install fair-esm biopython torch sklearn pandas torch torch_geometric scikit-learn
```

## Data preprocessing 
The Colab downloads preprocessed data files from [this Drive link](https://drive.google.com/file/d/1xnwJ9jt-GhE61Gqv3035eOmPXr9EtNB6/view). The data is generated using the preprocessing functions in `dataset_preprocessing.py`. 

## GNN experiments 
```
python train_node_preds.py
```
In that file, the function `do_training` trains and evaluates a model. At the bottom of the file, you can specify the parameters for a single experiment: the task, ML model, node features, dataset splitting method. The function `do_experiment` loops over every experiment and model choice and saves the results.



