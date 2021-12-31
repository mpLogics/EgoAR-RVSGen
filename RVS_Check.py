"""
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
"""
import scipy.io as sio
import pandas as pd
import os
from Data import LoadData

#from OpticalFlow import learn_optical_flow
import math
from RVSGen import GenVerbSpace
import numpy as np

def get_models(return_all):
    if not return_all:
       return keras.models.load_model("Verb_Predictor")
    else:
        return keras.models.load_model("Noun_Predictor"),keras.models.load_model("Verb_Predictor")
    
def predict_with_RVS(Verb_Probable,pred):
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

def return_true_annotation(value,component):
    if component=="Verb":
        return value + 1
    else:
        if value<=15:
            value+=1
        elif value>15 and value<=43:
            value+=2
        else:
            value-=3


def set_verb_rules():
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
verb_predictor = get_models(return_all=False)

for i in range(10):
    mag,angle,encoding = data_loader.load_file(i,modality="Verb")
    
    init_matrix,init_Annot = data_loader.get_any_matrix(
        mag,
        angle,
        encoding)
    
    final_matrix = np.reshape(init_matrix,((
        1,init_matrix.shape[0],
        init_matrix.shape[1],
        init_matrix.shape[2])))
    
    feature_extractor = keras.Model(
        inputs=verb_predictor.input,
        outputs=verb_predictor.output)
    
    pred1 = verb_predictor.predict(final_matrix)
    pred2 = feature_extractor.predict(final_matrix)

print(pred1)
print()
print(pred2)
#Verb_Probable = reduced_verb_space.RVSGen(Noun_Pred=Nouns[0],K_Value=10)

