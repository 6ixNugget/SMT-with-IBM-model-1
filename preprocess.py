import re
import string
import os

dword_list = ["d'abord","d'accord", "d'ailleurs", "d'habitude"]
punc_regex = r'([' + re.escape(string.punctuation.replace("'",'')) +']+)'

def singular_def_article(word_list):
    result = []
    for i in range(len(word_list)):
        result += re.split(r'(^l\')', word_list[i])
    result = list(filter(None, result))
    return result

def single_consonant(word_list):
    result = []
    for i in range(len(word_list)):
        if word_list[i] in dword_list:
            result += [word_list[i]]
        else:
            result += re.split(r'(^[b-df-hj-np-tv-z]\')', word_list[i])
    result = list(filter(None, result))
    return result

def que_words(word_list):
    result = []
    for i in range(len(word_list)):
        result += re.split(r'(^qu\')', word_list[i])
    result = list(filter(None, result))
    return result

def puisque_or_lorsque(word_list):
    result = []
    for i in range(len(word_list)):
        result += re.split(r'(^puisqu\'|^lorsqu\')', word_list[i])
    result = list(filter(None, result))
    return result

def split_contraction(word_list):
    word_list = singular_def_article(word_list)
    word_list = single_consonant(word_list)
    word_list = que_words(word_list)
    word_list = puisque_or_lorsque(word_list)
    return word_list

def split_punctuation(word_list):
    result = []
    for i in range(len(word_list)):
        result += re.split(punc_regex, word_list[i])
    result = list(filter(None, result))
    return result

def split_quotations(word_list):
    result = []
    for i in range(len(word_list)):
        result += re.split(r'(^\'+|\'+$)', word_list[i])
    result = list(filter(None, result))
    return result

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    word_list = list(filter(None, [x.strip() for x in in_sentence.lower().split()]))
    word_list = split_quotations(word_list)
    word_list = split_punctuation(word_list)

    if language == 'f':
        word_list = split_contraction(word_list)

    word_list = ["SENTSTART"] + word_list + ["SENTEND"]
    return " ".join(word_list)

if __name__ == "__main__":
    data_path = "./data/Hansard/Training"
    filenames = set('.'.join(f.split('.')[:-1]) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)))
    save_to = "./data/processed/Training"
    language = 'e'

    for filename in filenames:
        processed_lines = [] 
        with open(os.path.join(data_path, filename + '.' + language), 'r') as file:
            for line in file.read().splitlines():
                processed_lines.append(preprocess(line, language))
        with open(os.path.join(save_to, filename + '.' + language), 'w') as file:
            for line in processed_lines:
                file.write(line)


    
