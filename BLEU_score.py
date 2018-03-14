import math

def BLEU_score(candidate, references, n):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
    bleu_score = 1
    word_list = candidate.split()
    for pi in range(1, n+1):
        hit = 0
        for i in range(len(word_list)-pi+1):
            keyword = " ".join(word_list[i:i+pi])
            in_ref = False
            for ref in references:
                in_ref = (keyword in ref) | in_ref
            if in_ref:
                hit += 1
        bleu_score *= hit / (len(word_list)-pi+1)
    
    brevity = 0
    for ref in references:
        brev_cur = len(ref.split()) / len(word_list)
        if abs(brev_cur - 1) < abs(brevity - 1):
            brevity = brev_cur
    
    BP = 1 if brevity < 1 else math.exp(1-brevity)
    bleu_score = BP * (bleu_score) ** (1/n)

    return bleu_score


# if __name__ == "__main__":
#     c = "It is a guide to action which ensures that the military always obeys the commands of the party".lower()
#     #c = "It is to insure the troops forever hearing the activity guidebook that party direct".lower()
#     ref = [
#         "It is a guide to action that ensures that the military will forever heed Party commands".lower(),
#         "It is the guiding principle which guarantees the military forces always being under command of the Party".lower(),
#         "It is the practical guide for the army always to heed the directions of the party".lower()
#     ]
#     print(BLEU_score(c, ref, 2))

