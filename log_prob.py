from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
    Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
    word_list = sentence.split()
    log_prob = 0
    for i in range(len(word_list)-1):
        print(word_list[i], word_list[i+1])
        bi_count = LM['bi'][word_list[i]][word_list[i+1]]
        uni_count = LM['uni'][word_list[i]]
        if uni_count == 0 and smoothing:
            return float('-inf')
        log_prob += log(((bi_count + delta)/(uni_count + delta * vocabSize)))
    return log_prob