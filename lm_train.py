from preprocess import *
import pickle
from os import listdir
from os.path import isfile
import os

data_path = "./data/processed/Training/" # /u/cs401/A2_SMT/data/Hansard/Training/

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
    filenames = set('.'.join(f.split('.')[:-1]) for f in listdir(data_dir) if isfile(os.path.join(data_dir, f)))

    LM = {}
    LM["uni"], LM["bi"] = {}, {}
    for filename_stem in filenames:
        build_uni(os.path.join(data_dir, filename_stem + '.' + language), LM["uni"])
        build_bi(os.path.join(data_dir, filename_stem + '.' + language), LM["bi"])

    # Save Model
    if fn_LM:
        with open(fn_LM+'.pickle', 'wb') as file:
            pickle.dump(LM, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return LM

def build_uni(filename, uni_dict):
    with open(filename, 'r') as file:
        lines = file.read().splitlines()

    for line in lines:
        for word in line.split():
            if word in uni_dict:
                uni_dict[word] += 1
            else:
                uni_dict[word] = 1

def build_bi(filename, bi_dict):
    with open(filename, 'r') as file:
        lines = file.read().splitlines()

    for line in lines:
        words = line.split()
        for i in range(len(words)-1):
            if not words[i] in bi_dict:
                bi_dict[words[i]] = {}
            
            if words[i+1] in bi_dict[words[i]]:
                bi_dict[words[i]][words[i+1]] += 1
            else:
                bi_dict[words[i]][words[i+1]] = 1

if __name__ == "__main__":
    LM = lm_train(data_path, 'e', None)
    print (LM["bi"]["SENTSTART"])