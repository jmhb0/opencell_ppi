from torch_geometric.data import Data, InMemoryDataset
from typing import Callable, List, Optional
import os
import requests
import ipdb
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import tqdm

class OpencellPPI(InMemoryDataset):
	file_metadata = "opencell-library-metadata.csv"
	file_interactions = "opencell-protein-interactions.csv"
	file_interactions_readme = "opencell-protein-interactions-readme.csv"
	file_localization_annotation = "opencell-localization-annotations.csv"
	file_localization_annotation_readme = "opencell-localization-annotations-readme.csv"
	raw_files = [
		file_metadata,
		file_interactions, file_interactions_readme, 
		file_localization_annotation,  file_localization_annotation_readme
		]
	url_base = "https://opencell.czbiohub.org/data/datasets/"

	processed_files = ["processed.pt"]

	def __init__(self,
				 root: str,
				 transform: Optional[Callable] = None,
				 pre_transform: Optional[Callable] = None,
				 pre_filter: Optional[Callable] = None,
				 features_type="dummy",
				 test_split_method="cite_order",
				 test_split_frac=0.2,
				 ):
		"""
		test_split_method: ("cite_order","random")
		features: ("dummy","sequencelanguage","image","sequencelanguage_image")
		"""
		self.test_split_method = test_split_method
		self.test_split_frac = test_split_frac
		self.features_type = features_type
		super().__init__(root, transform, pre_transform, pre_filter)
		self.load(self.processed_paths[0])

	def download(self):
		# data dir
		data_dir = Path(self.root)
		data_dir.mkdir(exist_ok=True)
		headers = {'Referer': self.url_base}

		# do HTTP request for the OpenCell data
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
		return self.processed_files

	def process(self):

		# start with annotations for protein localization - defines protein indexing
		fname_localization = f"{self.root}/raw/{self.file_localization_annotation}"
		df_loc = pd.read_csv(fname_localization, na_filter=False)
		self.prots = df_loc['target_name'].values
		self.prot_to_idx = dict(zip(self.prots, range(len(self.prots))))
		self.idx_to_prot = {v:k for k,v in self.prot_to_idx.items()}

		# labels for subcellular localization
		self.y_loc_nuclear, self.y_loc = self.get_node_labels_localization(df_loc)

		# get the features
		if self.features_type=="dummy":
			self.features = torch.ones((len(self.prots),5))

		else:
			raise NotImplementedError()
			# other options are ("dummy","sequencelanguage","image","sequencelanguage_image") 
			self.features_sequence = self.get_features_sequence()
			self.features_image = self.get_features_image()

		# get edges: the interaction data 
		df_interact = pd.read_csv(f"{self.root}/raw/{self.file_interactions}")
		edge_lst = []
		for i, row in df_interact.iterrows():
			prot0, prot1 = row['target_gene_name'], row['interactor_gene_name']
			if ((prot0 in self.prots) and (prot1 in self.prots)):
				edge_lst.append([self.prot_to_idx[prot0], self.prot_to_idx[prot1]])
		self.edge_index = torch.from_numpy(np.stack(edge_lst)).swapaxes(1,0)

		# generate train mask. Random or by ordering the proteins by pubmed cites
		if self.test_split_method == "random":
			idxs = torch.randperm(len(self.prots))
			idxs_test = idxs[:int(self.test_split_frac*len(self.prots))]

		elif self.test_split_method == "cite_order":
			self.pubmed_cite_order_norm = self.get_protein_ordering()
			idxs_test = torch.where(self.pubmed_cites_order_norm<=self.test_split_frac)[0]

		else:
			raise ValueError("test_split_method must be one of ()'cite_order','random")

		self.train_mask = torch.ones(len(self.prots))
		self.train_mask[idxs_test] = 0
		self.test_mask = 1-self.train_mask

		# create a list of data objects 
		data_list = [Data(x=self.features, edge_index=self.edge_index, 
			train_mask=self.train_mask, test_mask=self.test_mask, 
			y_loc_nuclear=self.y_loc_nuclear, y_loc=self.y_loc,
			)]

		# # boilerplate stuff  
		# if self.pre_filter is not None:
		# 	data_list = [data for data in data_list if self.pre_filter(data)]

		# if self.pre_transform is not None:
		# 	data_list = [self.pre_transform(data) for data in data_list]
		
		self.save(data_list, self.processed_paths[0])

	def get_node_labels_localization(self, df_loc):
		""" 
		Read the subcellular localization labels. Generate two types of labels:
			
			y_loc_nuclear: 1 if the protein localizes to the nucleus, and 0 
			otherwise. If it localizes both in and outside the nulceus, then 1.
			The shape is (n_proteins,1)
			
			y_loc: multiclass classification. Protein can localize to multiple 
			compartments. If there are `n_classes` number of classes, then the 
			output shape is (n_proteins,n_classes), where each entry is 0 or 1.
		"""
		loc_grade3 = df_loc['annotations_grade_3']
		# get labels for the binary classification task
		y_loc_nuclear = torch.tensor(['nucle' in s for s in loc_grade3])
		y_loc_nuclear = y_loc_nuclear.long().unsqueeze(1)

		# get the unique localization labels
		all_locs = [s.split(";") for s in loc_grade3]
		all_locs_flat = [s for s_ in all_locs for s in s_] 	# flatten the list
		uniq_locs = np.sort(np.unique(all_locs_flat))		# get unique stuff
		self.uniq_locs = [s for s in uniq_locs if s!='']# remove empty entries
		self.loc_to_id = dict(zip(self.uniq_locs, range(len(self.uniq_locs))))
		self.id_to_loc = {v:k for k,v in self.loc_to_id.items()}

		# get labels for the multiclass classification task
		n_proteins = len(loc_grade3)
		n_classes = len(self.uniq_locs)
		y_loc = torch.zeros((n_proteins, n_classes))
		for i, locs in enumerate(all_locs):
			for _, loc in enumerate(locs):
				if loc == "": 
					continue
				y_loc[i, self.loc_to_id[loc]] = 1

		return y_loc_nuclear, y_loc
	
	def get_features_sequence(self):
		""" 
		From file, generate a list of sequences and an array of representations 
		derived from a protein language model. 
		"""
		
		file_protein_sequences = f"{self.root}/raw/protein_sequences.pt"
		lookup_protein_sequences = torch.load(file_protein_sequences)
		self.sequences = [lookup_protein_sequences.get(p,"") for p in self.prots]
		
		file_sequence_representations = f"{self.root}/raw/sequence_representations.pt"
		self.sequence_representations = torch.load(file_sequence_representations)

	def get_features_image(self):
		""" """
		pass 

	def get_protein_ordering(self):
		""" 
		Assign a rank ordering, and a normalized rank orderings to each of protein.
		This is based on data that we import that was previously calculated in 
		the `dataset_preprocessing.py` func. 
		"""
		file_pubmed_cites = f"{self.root}/raw/prot_pubmed_counts.pt"
		lookup_pubmed_cites = torch.load(file_pubmed_cites)
		self.pubmed_cites = torch.tensor([lookup_pubmed_cites[p] for p in self.prots])
		# get the rank ordering 
		sorted_indices = sorted(range(len(self.pubmed_cites)), 
			key=lambda x: self.pubmed_cites[x])
		self.pubmed_cites_order = torch.zeros(len(sorted_indices))
		for rank, index in enumerate(sorted_indices):
			self.pubmed_cites_order[index] = rank

		self.pubmed_cites_order_norm = self.pubmed_cites_order / self.pubmed_cites_order.max()

		return self.pubmed_cites_order_norm


if __name__ == "__main__":
	dataset = OpencellPPI(root="data")
	ipdb.set_trace()
	pass
