import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras import backend as bk

import scipy.io as sio
import pandas as pd
import os
from Data import LoadData

#from OpticalFlow import learn_optical_flow
import math
from RVSGen import GenVerbSpace
import numpy as np
import math

class RVS_Implement():
    def __init__(self):
        self.VerbSet = []
        self.rvs_generator = GenVerbSpace()

    def custom_activation(self,x,P_Verb):        
        sum=0
        activation_values=[]
        
        for j in range(len(x)):
            if j in self.VerbSet:
                sum+=math.exp(x[j]*P_Verb[j+1])
        
        for j in range(len(x)):
            if j in self.VerbSet:
                activation_values.append((math.exp(x[j]*P_Verb[j+1]))/sum)
            else:
                activation_values.append(0)
        
        activation_values = np.array(activation_values)
        return activation_values
    
    
    def get_models(self,return_all):
        if not return_all:
            return keras.models.load_model("Verb_Predictor")
        else:
            return keras.models.load_model("Noun_Predictor"),keras.models.load_model("Verb_Predictor")
        
    def predict_with_RVS(self,Verb_Probable,pred):
        activation=[]
        sum_features=0
        
        for i in range(len(Verb_Probable)):
            sum_features+=math.exp(pred[Verb_Probable[i]])

        for i in range(len(Verb_Probable)):
            soft_value = math.exp(Verb_Probable[i])
            activation.append(soft_value/sum_features)

        final_features = np.array(activation)
        print(final_features)
        predicted_verb = np.argmax(final_features)
        print(predicted_verb)

    def return_true_annotation(self,value,component):
        if component=="Verb":
            return value + 1
        else:
            if value<=15:
                value+=1
            elif value>15 and value<=43:
                value+=2
            else:
                value-=3


    def set_verb_rules(self):
        print("No existing rules found, creating new rules!")
        reduced_verb_space = GenVerbSpace()
        Nouns = reduced_verb_space.getNounSet()
        Verb_Set = reduced_verb_space.getVerbSet()
        totalSamples = reduced_verb_space.getTotalSamples(mode="train")

        P_Noun = reduced_verb_space.calProbNouns(totalSamples)
        P_Verb = reduced_verb_space.calProbVerbs(totalSamples)
        P_Noun_Verb = reduced_verb_space.calProbCombinations(totalSamples)

        return P_Noun,P_Verb,P_Noun_Verb



data_loader = LoadData()
K = 10
rvs_checker = RVS_Implement()
P_Noun,P_Verb,P_Noun_Verb = rvs_checker.set_verb_rules()

total_samples = rvs_checker.rvs_generator.getTotalSamples(mode="train")
verb_predictor = rvs_checker.get_models(return_all=False)
verb_predictor.summary()

try:
    Metrics = np.load("data/results/K_"+(str)(K)+"_Metrics.npz",allow_pickle=True)
    ground_truth = Metrics['a']
    RVS_Predicted = Metrics['b']
except FileNotFoundError:
    print("No existing file found!")
    ground_truth = []
    RVS_Predicted = []
except:
    print("Directory Not Found")
Nouns = pd.read_csv("data/Splits/train_split1.csv")["Noun"]

for i in range(total_samples):
    mag,angle,encoding = data_loader.load_file(i,modality="OF")    
    rvs_checker.VerbSet = np.array(rvs_checker.rvs_generator.RVSGen(Noun_Pred=Nouns[i],K_Value=K))-1
    init_matrix,init_Annot = data_loader.get_any_matrix(
        mag,
        angle,
        encoding)
    
    ground_truth.append(init_Annot[0]-1)    
    final_matrix = np.reshape(init_matrix,(
        1,
        init_matrix.shape[0],
        init_matrix.shape[1],
        init_matrix.shape[2],
        1))
        
    base_model = verb_predictor.get_layer('dense_3').output 
    feature_extractor = keras.Model(
        inputs = verb_predictor.input,
        outputs = base_model)

    pred1 = verb_predictor.predict(final_matrix)
    pred2 = feature_extractor.predict(final_matrix)
    activated_values = rvs_checker.custom_activation(x=pred2[0],P_Verb=P_Verb)
    
    print("\nVerb Set: ",rvs_checker.VerbSet)
    print("Ground Truth: ",init_Annot)
    print("From activated Values: ",np.argmax(activated_values))
    print("From feature vector values: ",np.argmax(pred2[0]))
    print("From fully predicted values: ",np.argmax(pred1[0]))
    RVS_Predicted.append(np.argmax(activated_values))

    if i+1%100:
        print("\n\nFiles read:",i)
        print("Current Accuracy:",np.mean(np.array(ground_truth)==np.array(RVS_Predicted)))

np.savez(
    "data/results/K_"+(str)(K)+"_Metrics.npz",
    a = np.array(ground_truth),
    b = np.array(RVS_Predicted))


#Verb_Probable = reduced_verb_space.RVSGen(Noun_Pred=Nouns[0],K_Value=10)

