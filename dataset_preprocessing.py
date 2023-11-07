"""
Data preprocessing that is too expensive (in time or GPUs), or rely on unreliable
internet connections. This makes them not suitable to run in the download() func 
of the standard torch_geometric.data.InMemoryDataset definition

Instead, the code is run here once, saved to a file, and then put on Google Drive. 
Then the Dataset loader in `dataset.py` will just wget that data.
"""

import requests
from pathlib import Path
import os 
import ipdb 
import torch
import esm
import numpy as np
import pandas as pd 
import tqdm
import urllib

from Bio import Entrez
import time
Entrez.email = "jmhb@stanford.edu"

DATA_DIR = Path("data_preprocessed")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# files to download
file_localization_annotation = "opencell-localization-annotations.csv"
file_metadata = "opencell-library-metadata.csv"
files = [file_localization_annotation, file_metadata]
url_base = "https://opencell.czbiohub.org/data/datasets/"

# filenames for outputs
file_pubmed_counts = "prot_pubmed_counts.pt"
file_sequences = "protein_sequences.pt" # raw protein sequences 
file_sequence_representations = "sequence_representations.pt" # ESM model repesentations

def get_files():
	""" Download the files from the `files` list above that we need """
	for file in files:
		file_path = DATA_DIR / file
		if not os.path.exists(file_path):
			headers = {'Referer': url_base}
			response = requests.get(f"{url_base}/{file}", 
				headers=headers)
			if response.status_code == 200:
				with open(file_path, 'wb') as f:
					f.write(response.content)
				print(f"Downloaded {file_path}")
			else:
				raise (
					f"Failed to download {file_path} with status code {response.status_code}"
				)

def count_protein_pubmed_cites():
	"""
	Use the Biopython Entrez to query NCBI database. Here, we take a protein 
	in its human-readable ID, e.g. "TOMM20", searching the number of articles 
	mentioning that protein, and counting them. 

	The point of this is to provide a ranking of "well-known" proteins. Less 
	well-known proteins will then be put in the test set later on, because this 
	better simulates the discovery process. 
	"""
	file_path = DATA_DIR / file_localization_annotation
	df_loc = pd.read_csv(file_path)
	prots = df_loc['target_name'].values

	def search_protein_citations(protein_name):
		# Search for the protein in PubMed
		query = protein_name + "[Title/Abstract]"
		handle = Entrez.esearch(db="pubmed", term=query, retmax="10000")
		record = Entrez.read(handle)
		handle.close()
		# Get the list of PubMed IDs (PMIDs)
		pmid_list = record["IdList"]

		# Return the count of articles
		return len(pmid_list), pmid_list

	counts = {}
	print(f"Counting protein PubMed cites")
	for protein_name in tqdm.tqdm(prots):
		# time.sleep(0.4)
		try: 
			citation_count, pmids = search_protein_citations(protein_name)
		except:  
			print("Error")
			time.sleep(60)
			# if it fails a second time after the pause, it won't continue
			citation_count, pmids = search_protein_citations(protein_name)

		counts[protein_name] = citation_count

	torch.save(counts, DATA_DIR / file_pubmed_counts)
	return counts

def get_protein_sequence_representation():
	""" 
	Get the protein sequence by reading the metadata file, getting the uniprot 
	id and looking up the sequence using the uniprot API.
	"""
	df_meta = pd.read_csv(DATA_DIR / file_metadata)
	
	def get_uniprot_sequence(uniprot_id):
		""" Returns in FASTA format """
		# UniProt REST API URL for fetching sequence data
		url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
		
		# Send a GET request to the UniProt API
		response = requests.get(url)
		
		# Check if the request was successful
		if response.ok:
			# Return the FASTA formatted sequence data
			return response.text
		else:
			# If the request failed, print the error
			print(f"Error retrieving data for UniProt ID {uniprot_id}: {response.status_code}")
			return None

	def extract_sequence_from_fasta(fasta_str):
		""" Extract the raw sequence from the fasta format """
		# Split the string into lines
		lines = fasta_str.strip().split('\n')
		# Join the lines that do not start with '>' (ignoring the header line)
		sequence_only = ''.join(line for line in lines if not line.startswith('>'))
		return sequence_only

	prot_to_seq = {}
	for i, row in tqdm.tqdm(df_meta.iterrows(), total=len(df_meta)):
		try: 
			seq_fasta = get_uniprot_sequence(row['uniprot_id'])
			seq = extract_sequence_from_fasta(seq_fasta)

		except:
			time.sleep(60)
			# if it fails again, the program will exit 
			seq_fasta = get_uniprot_sequence(row['uniprot_id'])
			seq = extract_sequence_from_fasta(seq_fasta)

		prot_to_seq[row['gene_name']] = seq

	torch.save(prot_to_seq, DATA_DIR / file_sequences)

def get_sequence_representation_language():
	""" 
	read the 
	sources for extracting embeddings: https://pypi.org/project/fair-esm/
	"""
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
	batch_converter = alphabet.get_batch_converter(truncation_seq_length=750)
	model.eval()  # disables dropout for deterministic results

	prot_to_seq = torch.load(DATA_DIR / file_sequences)
	
	data = []
	for i, k in enumerate(prot_to_seq.keys()):
		# if i >3:break
		data.append([k, prot_to_seq[k]])

	batch_labels, batch_strs, batch_tokens = batch_converter(data)

	# get the representations in batches
	from torch.utils.data import TensorDataset, DataLoader
	loader = DataLoader(TensorDataset(batch_tokens), batch_size=4)

	token_representations_all = []
	model.to(device)
	with torch.no_grad():
		for x, in tqdm.tqdm(loader):
			print('1 ', end=" ")
			results = model(x.to(device), repr_layers=[33], return_contacts=True)
			token_representations = results["representations"][33]
			token_representations_all.append(token_representations.cpu())
	token_representations_all = torch.cat(token_representations_all)
	
	ipdb.set_trace()
	sequence_representations = []
	for i, (_, seq) in enumerate(data):
	    sequence_representations.append(token_representations_all[i, 1 : len(seq) + 1].mean(0))

	ipdb.set_trace()
	out = torch.stack(sequence_representations)
	torch.save(out, DATA_DIR / file_sequence_representations)

if __name__=="__main__":
	get_files()
	# count_protein_pubmed_cites()
	# get_protein_sequence_representation()
	# get_sequence_representation_language()
	ipdb.set_trace()
	pass 
