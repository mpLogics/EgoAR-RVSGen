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

Class_wise={}
Total_elements={}
for i in range(53):
    Class_wise[i+1] = 0
    Total_elements[i+1] = 0


for i in range(Nouns['a'].shape[0]):
    Total_elements[df["Noun"][i]]+=1
    if df["Noun"][i] == Nouns['a'][i]:
        Class_wise[df["Noun"][i]]+=1

Tot=[]
Acc_cw = []
Crct_val = []
for i in range(len(Class_wise)):
    Tot.append(Total_elements[i+1])
    Crct_val.append(Class_wise[i+1])
    if Total_elements[i+1]==0:
        Acc_cw.append(0)    
    else:
        Acc_cw.append(Class_wise[i+1]/Total_elements[i+1])
df = pd.DataFrame()

print(Total_elements)
print(Class_wise)
print(Acc_cw)

df["Total Examples"] = Tot
df["Correct Examples"] = Crct_val
df["Accuracy"] = Acc_cw
df.to_csv("Class_Wise_Noun.csv")

