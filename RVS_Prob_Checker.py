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


def get_top_1(tot_classes,ground_truth,predicted):
    return np.sum(ground_truth==predicted),np.mean(ground_truth==predicted)

def get_top_5(tot_classes,ground_truth,predicted):
    top5=0
    
    for i in range(ground_truth.shape[0]):
        if ground_truth[i] in predicted[i]:
            top5+=1    
    return top5,top5/ground_truth.shape[0]

def get_class_wise(tot_classes,predicted,ground_truth,zero_indexing=True,start_index=0):
    Class_wise={}
    Total_elements={}

    for i in range(start_index,start_index + tot_classes):
        Class_wise[i] = 0
        Total_elements[i] = 0

    for i in range(ground_truth.shape[0]):
        Total_elements[ground_truth[i]]+=1
        if ground_truth[i] == predicted[i]:
            Class_wise[ground_truth[i]]+=1

    Tot=[]
    Acc_cw = []
    Crct_val = []
    Non_existent = []
    for i in range(start_index,start_index+len(Class_wise)):
        Tot.append(Total_elements[i])
        Crct_val.append(Class_wise[i])
        if Total_elements[i]==0:
            Acc_cw.append(0)
            Non_existent.append(i-start_index)
        else:
            Acc_cw.append(Class_wise[i]/Total_elements[i])
    
    print(Total_elements)
    print(Class_wise)
    mean_class_acc = np.sum(np.array(Acc_cw))/(len(Acc_cw)-len(Non_existent))
    return mean_class_acc

def get_test_metrics():
    Nouns = np.load("data/results/test_reports/Nouns.npz",allow_pickle=True)
    Verbs = np.load("data/results/test_reports/Verbs.npz",allow_pickle=True)

    df = pd.read_csv("data/Splits/test_split1.csv")
    #noun_top5 = 0
    #verb_top5 = 0
    #for i in range(Nouns['b'].shape[0]):
    #   if df["Noun"][i] in Nouns['b'][i]:
    #        noun_top5+=1
    #
    #    if df["Verb"][i] in Verbs['b'][i]:
    #        verb_top5+=1
    #
    #print("Top 1 (Noun): ",np.sum(Nouns['a']==df["Noun"]))
    #print("Top 5 (Noun): ",noun_top5)
    #
    #print("Top 1 (Verb): ",np.sum(Verbs['a']==df["Verb"]))
    #print("Top 5 (Verb): ",verb_top5)


    print("\n")
    print("Nouns:",get_top_1(tot_classes=53,predicted=Nouns['a'],ground_truth=df["Noun"]),
                get_top_5(tot_classes=53,predicted=Nouns['b'],ground_truth=df["Noun"]))
    print("\n")
    print("Verbs",get_top_1(tot_classes=19,predicted=Verbs['a'],ground_truth=df["Verb"]),
                get_top_5(tot_classes=19,predicted=Verbs['b'],ground_truth=df["Verb"]))
    print("\n")
    print("\nMean Class Accuracy (Noun): ",get_class_wise(tot_classes=53, ground_truth=df["Noun"],predicted=Nouns['a'],zero_indexing=False,start_index=1))
    print("\n")
    print("\nMean Class Accuracy (Verb): ",get_class_wise(tot_classes=19, ground_truth=df["Verb"],predicted=Verbs['a'],zero_indexing=False,start_index=1))
        






"""
Class_wise_Noun={}
Total_elements_Noun={}

Class_wise_Verb={}
Total_elements_Verb={}

for i in range(tot_noun):
    Class_wise_Noun[i+1] = 0
    Total_elements_Noun[i+1] = 0

for i in range(tot_verb):
    Class_wise_Verb[i+1] = 0

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

"""