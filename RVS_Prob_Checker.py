import numpy as np
from RVSGen import GenVerbSpace

reduced_verb_space = GenVerbSpace()
totalSamples = reduced_verb_space.getTotalSamples(mode="train")
P_Noun_Verb = reduced_verb_space.calProbCombinations(totalSamples)
print("Sum of probabilities:",np.sum(np.array(list(P_Noun_Verb.values()))))