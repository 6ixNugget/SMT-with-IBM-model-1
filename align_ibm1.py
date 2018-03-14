from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import pickle
import os
import copy
import json

placeholders = ["SENTSTART", "SENTEND"]

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
	"""
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model

	OUTPUT:
	AM :			(dictionary) alignment model structure

	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.

			LM['house']['maison'] = 0.5
	"""
	# Read training data
	eng, fre = read_hansard(train_dir, num_sentences)

	# Initialize AM uniformly
	AM = initialize(eng, fre)
	
	# Iterate between E and M steps
	for i in range(max_iter):
		em_step(AM, eng, fre)

	# Save AM as file
	if fn_AM:
		with open(fn_AM + '.pickle', 'wb') as file:
			pickle.dump(AM, file, protocol=pickle.HIGHEST_PROTOCOL)

	return AM

# ------------ Support functions --------------
def read_from_dir(train_dir, language, num_sentences):
	"""
	Read the specified language data from train_dir, preprocess it and return the processed lines
	"""
	filenames = set('.'.join(f.split('.')[:-1]) for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f)))
	line_count = 0

	for filename in filenames:
		processed_lines = [] 
		with open(os.path.join(train_dir, filename + '.' + language), 'r') as file:
			for line in file.read().splitlines():
				processed_lines.append(preprocess(line, language).split())
				line_count += 1
				if line_count >= num_sentences:
					return processed_lines

	return processed_lines

def read_hansard(train_dir, num_sentences):
	"""
	Read up to num_sentences from train_dir.

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider


	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

	Make sure to read the files in an aligned manner.
	"""
	eng = read_from_dir(train_dir, 'e', num_sentences)
	fre = read_from_dir(train_dir, 'f', num_sentences)
	return eng, fre

def initialize(eng, fre):
	"""
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	AM= {}
	assert (len(eng) == len(fre)), "The numbers of sentences do not match."
	for i in range(len(eng)):
		for e in eng[i][1:-1]:
			if e not in AM.keys():
				AM[e] = {}
			for f in fre[i][1:-1]:
				AM[e][f] = 1
	
	for e_k in AM.keys():
		for f_k in AM[e_k].keys():
			AM[e_k][f_k] = 1 / len(AM[e_k].keys())

	AM['SENTSTART'] = {'SENTSTART':1}
	AM['SENTEND'] = {'SENTEND':1}
	return AM

def populate_tcount(AM):
	tcount = copy.deepcopy(AM)
	for e in tcount.keys():
		for f in tcount[e].keys():
			tcount[e][f] = 0
	return tcount

def populate_total(AM):
	total = {}
	for e in AM.keys():
		total[e] = 0
	return total

def em_step(AM, eng, fre):
	"""
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	tcount = populate_tcount(AM)
	total = populate_total(AM)
	
	for i in range(len(eng)):
		E = eng[i]
		F = fre[i]
		for f in F[1:-1]:
			denom_c = 0
			for e in set(E[1:-1]):
				denom_c += AM[e][f] * F.count(f)
			for e in set(E[1:-1]):
				tcount[e][f] += AM[e][f] * F.count(f) * E.count(e) / denom_c
				total[e] += AM[e][f] * F.count(f) * E.count(e) / denom_c
	for e in total.keys():
		for f in tcount[e].keys():
			if e not in placeholders and f not in placeholders:
				AM[e][f] = tcount[e][f] / total[e]

# if __name__ == "__main__":
# 	align_ibm1("./data/Hansard/Training", 30000, 50, "./AM_30K")