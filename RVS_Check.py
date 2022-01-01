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
        self.VerbSet = np.array([0,13,1,4,12,5,6,7])

    def custom_activation(self,x):        
        print("Feature Values: ",x)
        
        sum=0
        activation_values=[]
        
        for j in range(len(x)):
            if j in self.VerbSet:
                sum+=math.exp(j)
        for j in range(len(x)):
            if j in self.VerbSet:
                activation_values.append(math.exp(j)/sum)
            else:
                activation_values.append(0)
        
        activation_values = np.array(activation_values)
        print("Activation Values: ",activation_values)
        print("Sum of actiation values: ",np.sum(activation_values))
        return activation_values
    
    
    def vectorize_ca(self):
        np_vectorize = np.vectorize(self.custom_activation)

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
        Verbs = reduced_verb_space.getNounSet()
        totalSamples = reduced_verb_space.getTotalSamples(mode="train")

        P_Noun = reduced_verb_space.calProbNouns(totalSamples)
        P_Verb = reduced_verb_space.calProbVerbs(totalSamples)
        P_Noun_Verb = reduced_verb_space.calProbCombinations(totalSamples)

        return P_Noun,P_Verb,P_Noun_Verb

#print("Predicted Noun =",Nouns[4])

#print(Verb_Probable)
#P_Noun,P_Verb,P_Noun_Verb = set_verb_rules(root="data/")

data_loader = LoadData()
rvs_checker = RVS_Implement()
verb_predictor = rvs_checker.get_models(return_all=False)
verb_predictor.summary()

for i in range(1):
    mag,angle,encoding = data_loader.load_file(i,modality="OF")
    
    init_matrix,init_Annot = data_loader.get_any_matrix(
        mag,
        angle,
        encoding)
    
    final_matrix = np.reshape(init_matrix,(
        1,
        init_matrix.shape[0],
        init_matrix.shape[1],
        init_matrix.shape[2],
        1))
    
    
    base_model = verb_predictor.get_layer('flatten').output
    final_model = keras.layers.Dense(units=19,name="Predictions",activation=None)(base_model)
    #final_model = keras.layers.Dense
    
    #feature_extractor = keras.Model(
    #    inputs=verb_predictor.input,
    #    outputs=verb_predictor.get_layer('flatten').output)
    feature_extractor = keras.Model(
        inputs = verb_predictor.input,
        outputs = final_model)

    pred1 = verb_predictor.predict(final_matrix)
    pred2 = feature_extractor.predict(final_matrix)

    activated_values = rvs_checker.custom_activation(x=pred2)
    
    print("Pred 1 shape: ",pred1.shape)
    print("Pred 2 shape: ",pred2.shape)
    print("Activated Values: ",activated_values)
    print()
    print("Pred 1: ",pred1)
    
    print("From Pred 1: ",np.argmax(pred1))
    print("From Pred 2: ",np.argmax(activated_values))

#Verb_Probable = reduced_verb_space.RVSGen(Noun_Pred=Nouns[0],K_Value=10)

