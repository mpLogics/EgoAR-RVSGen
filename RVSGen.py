import numpy as np
import pandas as pd
import regex as re

class GenVerbSpace():
    def __init__(self):
        self.Noun_Path = "data/action_annotation/noun_idx.txt"
        self.Verb_Path = "data/action_annotation/verb_idx.txt"
        self.Noun_Space = pd.read_csv(self.Noun_Path,header=None)
        self.Verb_Space = pd.read_csv(self.Verb_Path,header=None)
        self.Splits = 3

    def getVerbSet(self):
        Verb = []
        for i in range(len(self.Verb_Space)):
            Verb.append((int)(re.split(" ",self.Verb_Space[0][i])[-1]))
        return Verb
    
    def getNounSet(self):
        Noun=[]
        for i in range(len(self.Noun_Space)):
            Noun.append((int)(re.split(" ",self.Noun_Space[0][i])[-1]))
        return Noun

    def calProbNouns(self,totalSamples):
        Noun = self.getNounSet()
        N = np.count_nonzero(np.unique(Noun))
    
        P_Noun = {}
        for i in range(N):
            frequency=0
            for j in range(3):
                train = pd.read_csv("data/Splits/train_split" + (str)(j+1)+".csv")
                frequency+=np.sum(train["Noun"]==Noun[i])
            P_Noun[Noun[i]] = frequency/totalSamples
        return P_Noun

    def calProbVerbs(self,totalSamples):
        Verb = self.getVerbSet()
        V = np.count_nonzero(np.unique(Verb))
        P_Verb = {}
        for i in range(V):
            frequency=0
            for j in range(3):
                train = pd.read_csv("data/Splits/train_split" + (str)(j+1)+".csv")
                frequency+=np.sum(train["Verb"]==Verb[i])
            P_Verb[Verb[i]] = frequency/totalSamples
        return P_Verb
    
    def calProbCombinations(self,totalSamples):
        Noun = self.getNounSet()
        Verb = self.getVerbSet()
        N = len(Noun)
        V = len(Verb)
        P_Noun_Verb = {}
            
        for i in range(N):
            for j in range(V):
                frequency = 0
                for k in range(3):
                    train = pd.read_csv("data/Splits/train_split" + (str)(k+1)+".csv")
                    Combined = np.array(list(zip(train["Noun"],train["Verb"])))
                    frequency += np.sum(np.sum(Combined==(Noun[i],Verb[j]),axis=1)==2)
                if frequency == 0:
                    P_Noun_Verb[(Noun[i],Verb[j])] = 0.001
                else:
                    P_Noun_Verb[(Noun[i],Verb[j])] = frequency/totalSamples
        
        return P_Noun_Verb

    def getTotalSamples(self,mode):
        totalSamples=0
        for i in range(self.Splits):
            if mode=="train":
                data = pd.read_csv("data/Splits/train_split" + (str)(i+1) + ".csv")
            else:
                data = pd.read_csv("data/Splits/train_split" + (str)(i+1) + ".csv")
            totalSamples+=len(data)
        return totalSamples
    
    def RVSGen(self,Noun_Pred,K_Value):
            P_YVerb={}

            totalSamples = self.getTotalSamples(mode="train")
            P_Noun_Verb = self.calProbCombinations(totalSamples=totalSamples)
            P_Noun = self.calProbVerbs(totalSamples=totalSamples)
            P_Verb = self.calProbNouns(totalSamples=totalSamples)
            
            Verb_Set = self.getVerbSet()
            V = len(Verb_Set)

            for i in range(V):
                print(i,":",Verb_Set[i+1],":",P_Verb[Verb_Set[i]])
                P_YVerb[Verb_Set[i]] = (P_Noun_Verb[(Noun_Pred,Verb_Set[i])]/P_Noun[Noun_Pred])/P_Verb[Verb_Set[i]]
                
            Final_Probabilities = dict(sorted(P_YVerb.items(), key = lambda kv: kv[1]))
            Verb_Probable = list(Final_Probabilities.keys())[-K_Value:]
            return Verb_Probable