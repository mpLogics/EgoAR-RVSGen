import numpy as np
import pandas as pd
import regex as re

class GenVerbSpace():
    def __init__(self):
        self.Noun_Path = "data/action_annotation/noun_idx.txt"
        self.Verb_Path = "data/action_annotation/verb_idx.txt"
        #self.Action_Path = "data/action_annotation/action_idx.txt"
        self.Noun_Space = pd.read_csv(self.Noun_Path,header=None)
        self.Verb_Space = pd.read_csv(self.Verb_Path,header=None)
        self.Non_Occuring_Actions = 901
        #self.Action_Space = pd.read_csv(self.Action_Path,header=None)
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
    
    def getActionSet(self):
        Action=[]
        for i in range(len(self.Action_Space)):
            Action.append((int)(re.split(" ",self.Action_Space[0][i])[-1]))
        return Action

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
        #Action = self.getActionSet()

        N = len(Noun)
        V = len(Verb)
        #A = len(Action)
        P_Noun_Verb = {}
        sum_zeros=0

        for i in range(N):
            for j in range(V):
                frequency = 0
                for k in range(self.Splits):
                    train = pd.read_csv("data/Splits/train_split" + (str)(k+1)+".csv")
                    Combined = np.array(list(zip(train["Noun"],train["Verb"])))
                    frequency += np.sum(np.sum(Combined==(Noun[i],Verb[j]),axis=1)==2)
                if frequency == 0:
                    sum_zeros+=1
                    #Without loss of generality
                    P_Noun_Verb[(Noun[i],Verb[j])] = 0
                else:
                    P_Noun_Verb[(Noun[i],Verb[j])] = frequency/(totalSamples+self.Non_Occuring_Actions)

        it_what_remains = (1 - np.sum(np.array(list(P_Noun_Verb.values()))))/self.Non_Occuring_Actions
        
        
        for i in range(N):
            for j in range(V):
                if P_Noun_Verb[(Noun[i],Verb[j])]==0:
                    P_Noun_Verb[(Noun[i],Verb[j])] = it_what_remains

        print("Total non-occuring Actions:",sum_zeros)
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
    
    def RVSGen(self,Noun_Pred,K_Value,P_Noun_Verb,P_Verb):
            P_YVerb={}
            #P_Verb = self.calProbVerbs(totalSamples=totalSamples)
            #P_Noun = self.calProbNouns(totalSamples=totalSamples)
            #P_Noun_Verb = self.calProbCombinations(totalSamples=totalSamples)
            Verb_Set = self.getVerbSet()
            V = len(Verb_Set)

            for i in range(V):
                #P_YVerb[Verb_Set[i]] = (P_Noun_Verb[(Noun_Pred,Verb_Set[i])])/(P_Noun[Noun_Pred]*P_Verb[Verb_Set[i]])
                P_YVerb[Verb_Set[i]] = P_Noun_Verb[(Noun_Pred,Verb_Set[i])]
            
            Final_Probabilities = dict(sorted(P_YVerb.items(), key = lambda kv: kv[1]))
            Verb_Probable = list(Final_Probabilities.keys())[-K_Value:]
            return Verb_Probable