"""
import numpy as np
from RVSGen import GenVerbSpace

reduced_verb_space = GenVerbSpace()
totalSamples = reduced_verb_space.getTotalSamples(mode="train")
P_Noun_Verb = reduced_verb_space.calProbCombinations(totalSamples)
print("Sum of probabilities:",np.sum(np.array(list(P_Noun_Verb.values()))))
"""

import numpy as np
import pandas as pd
Nouns = np.load("data/results/test_reports/Nouns.npz",allow_pickle=True)
df = pd.read_csv("data/Splits/test_split1.csv")
print(np.sum(Nouns['a']==df["Noun"]))

sum_top5 = 0
for i in range(Nouns['b'].shape[0]):
    if df["Noun"][i] in Nouns['b'][i]:
        sum_top5+=1
print(sum_top5)
