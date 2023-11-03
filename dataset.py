from torch_geometric.data import InMemoryDataset
from typing import Callable, List, Optional
import os
import requests
import ipdb
import pandas as pd
import torch
import numpy as np
from pathlib import Path


class OpencellPPI(InMemoryDataset):
	raw_files = [
			f"opencell-protein-interactions.csv",
			f"opencell-protein-interactions-readme.csv",
			f"opencell-localization-annotations.csv",
			f"opencell-localization-annotations-readme.csv"
		]
	url_base = "https://opencell.czbiohub.org/data/datasets/"

	def __init__(self,
				 root: str,
				 transform: Optional[Callable] = None,
				 pre_transform: Optional[Callable] = None,
				 pre_filter: Optional[Callable] = None):
		super().__init__(root, transform, pre_transform, pre_filter)

	def download(self):
		# data dir
		data_dir = Path(self.root)
		data_dir.mkdir(exist_ok=True)
		headers = {'Referer': self.url_base}

		# do HTTP request
		for file in self.raw_files:
			response = requests.get(f"{self.url_base}/{file}", headers=headers)
			if response.status_code == 200:
				file_path = data_dir / "raw" / file
				with open(file_path, 'wb') as f:
					f.write(response.content)
				print(f"Downloaded {file_path}")
			else:
				raise (
					f"Failed to download {file_path} with status code {response.status_code}"
				)

	@property
	def raw_file_names(self):
		return self.raw_files 

	@property
	def processed_file_names(self):
		return []

	def process(self):
		df = pd.read_csv(f"{self.root}/raw/opencell-protein-interactions.csv")

		ipdb.set_trace()
		# unique genes and assign an index to each one 
		self.genes = np.unique(
			np.concatenate(
				(df.target_gene_name.values, df.interactor_gene_name.values)))
		self.gene_to_idx = {k:v for (k,v) in zip(self.genes, np.arange(len(self.genes)))}
		self.idx_to_gene = {v:k for (k,v) in self.gene_to_idx.items()} 

		# get the edges
		idx_targets = [self.gene_to_idx[gene] for gene in df.target_gene_name.values]
		idx_interactors = [self.gene_to_idx[gene] for gene in df.interactor_gene_name.values]
		self.edge_index = torch.from_numpy(np.stack((idx_targets,idx_interactors)))

		# get the features, depending on the passed options

		# create a list of data objects 
		data_list = []

		# boilerplate filtering 
		if self.pre_filter is not None:
			data_list = [data for data in data_list if self.pre_filter(data)]

		if self.pre_transform is not None:
			data_list = [self.pre_transform(data) for data in data_list]

		## this was also in  https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html 
		# data, slices = self.collate(data_list)
		# torch.save((data, slices), self.processed_paths[0])
		# return data_list 




if __name__ == "__main__":
	dataset = OpencellPPI(root="data")
	ipdb.set_trace()
	# dataset.download()
	pass
