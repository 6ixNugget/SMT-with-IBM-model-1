from lm_train import *
from log_prob import *
from preprocess import *
from decode import *
from BLEU_score import *
import pickle
import numpy

def read_in_references(test_dir, filename):
    """
    Preprocess the reference files.
    """
    ref_e = []
    ref_google_e = []
    with open(os.path.join(test_dir, filename + '.e')) as file:
        for line in file.read().splitlines():
            ref_e.append(preprocess(line, 'e'))
    with open(os.path.join(test_dir, filename + '.google.e')) as file:
        for line in file.read().splitlines():
            ref_google_e.append(preprocess(line, 'e'))

    refs = []
    for i in range(len(ref_e)):
        refs.append([ref_e[i], ref_google_e[i]])

    return refs

def read_in_candidates(test_dir, filename):
    """
    Preprocess the candidate file.
    """
    candidates = []
    with open(os.path.join(test_dir, filename + '.f')) as file:
        for line in file.read().splitlines():
            candidates.append(preprocess(line, 'f'))
    return candidates

def eval_candidates(test_dir, filename, LM_filename, AM_filename, n):
    LM = pickle.load(open(LM_filename, "rb"))
    AM = pickle.load(open(AM_filename, "rb"))
    
    cans = read_in_candidates(test_dir, filename)
    refs = read_in_references(test_dir, filename)

    scores = []
    for i in range(len(cans)):
        scores.append(BLEU_score(decode(cans[i], LM, AM), refs[i], n))
    return scores

def evalAligh(test_dir, filename, LM_filename, AM_filename):
    AM_models = ["1K", "10K", "15K", "30K"]
    eval_results = []
    for n in range(1, 3+1):
        model_result = []
        for model in AM_models:
            model_result.append(eval_candidates(test_dir, filename, LM_filename + '.pickle',
                                                AM_filename + '_' + model + '.pickle', n))
        eval_results.append(model_result)
    return eval_results

# if __name__ == "__main__":
#     result = evalAligh("./data/Hansard/Testing", "Task5", "./LM_e", "./AM")

#     for i in range(len(result)):
#         a = numpy.asarray(result[i])
#         numpy.savetxt("analysis_"+str(i+1)+".csv", a, delimiter=",")
    