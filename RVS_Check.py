import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D
from train import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras import backend as bk


import pandas as pd
from Data import LoadData,Data_Access

from OpticalFlow import learn_optical_flow
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
                activation_values.append((math.exp(x[j])*P_Verb[j+1])/sum)
            else:
                activation_values.append(0)
        
        activation_values = np.array(activation_values)
        return activation_values
    
    
    def get_noun_model(self,model_weights):
        m1 = Model()
        #m1.RGB_input_shape = (config_values["train"]["input_shape_x"],
        #                config_values["train"]["input_shape_y"],
        #                config_values["train"]["input_shape_z"]
        #                )
        model = m1.Time_Distributed_Model()
        model.summary()
        model.load_weights(model_weights)
        return model

        #m1.base_trainable = config_values["train"]["base_trainable"]
        #return keras.models.load_model("model_name")
    
    def get_verb_model(self):

        m1 = learn_optical_flow()
        m1.convLSTM_model()
        #print("Loading Weights")
        m1.temporal_extractor.load_weights('verb_weights.h5')
        #print("Weights loaded successfully")
        #model.load_weights("verb_weights.h5")
        #return keras.models.load_model("Verb_Predictors/Verb_Predictor_diff_521_1399_ver_19")
        return m1.temporal_extractor

        
    def set_verb_rules(self):
        print("No existing rules found, creating new rules!")
        reduced_verb_space = GenVerbSpace()
        totalSamples = reduced_verb_space.getTotalSamples(mode="train")

        P_Noun = reduced_verb_space.calProbNouns(totalSamples)
        P_Verb = reduced_verb_space.calProbVerbs(totalSamples)
        P_Noun_Verb = reduced_verb_space.calProbCombinations(totalSamples)

        return P_Noun,P_Verb,P_Noun_Verb

"""

rvs_rules = RVS_Implement()
rvs_gen = GenVerbSpace()
P_Noun,P_Verb,P_Noun_Verb = rvs_rules.set_verb_rules()

total_samples = rvs_gen.getTotalSamples(mode="test")
noun_predictor,verb_predictor = rvs_rules.get_models(return_all=True)
noun_predictor.summar()
verb_predictor.summary()

num_classes_verbs = 100
num_classes_nouns = 100
scale_factor = 1
fix_frames = 5
frame_rows = 120
frame_cols = 320
channels = 1
test_set = pd.read_csv("data/Splits/test_split1.csv")
total_samples = test_set["FileName"].shape[0]
ground_truth_Verb = np.array(test_set["Verb"])
ground_truth_Action = np.array(test_set["Action"])
ground_truth_Noun = np.array(test_set["Noun"])

data_loader = LoadData()
data_loader.mode = "test"
K = [i for i in range(1,15)]
access_order = [i for i in range(total_samples)]


model_pred=False
for z in range(len(K)):
    if z>=1:
        model_pred=True

    ground_truth_Verb = []
    Verb_Predicted = []

    RVS_Predicted = []
    
    i=0
    
    num_batches=0
    
    base_model = verb_predictor.get_layer('dense_3').output 
    feature_extractor = keras.Model(
        inputs = verb_predictor.input,
        outputs = base_model)

    while i < total_samples:
        if i>=2000:
            num_classes_verbs=1
            scale_factor = 1
        
        try:
            if not model_pred:
                Frame,Y_Noun,Val_Frame,Val_Noun = data_loader.read_frames(
                    i,
                    access_order,
                    num_classes_nouns)

                X_Value,Y_Value,Val_Frame,Val_Verb = data_loader.read_val_flow(
                    i,
                    access_order,
                    num_classes=num_classes_verbs,
                    multiply_factor=scale_factor)
        except:
            print("Error reading file index:",i)
            print("Num:",num_classes_verbs)
            break

        X_OF = np.reshape(X_Value,(
                    num_classes_verbs*scale_factor,
                    fix_frames,
                    frame_rows,
                    frame_cols,
                    channels))
        
        #Predicting for Training Set
        if not model_pred:
            pred_OF = verb_predictor.predict(X_OF)
            
        feature_pred = feature_extractor.predict(X_OF)

        for k in range(len(feature_pred)):
            rvs_rules.VerbSet = np.array(
                rvs_rules.rvs_generator.RVSGen(
                    Noun_Pred=Noun,
                    K_Value=K[z],
                    P_Noun_Verb=P_Noun_Verb,
                    P_Verb=P_Verb))-1
                    
            activated_values = rvs_rules.custom_activation(x=feature_pred[k],P_Verb=P_Verb)

            RVS_Predicted.append(np.argmax(activated_values))
            

            if not model_pred:
                Noun_Predicted.append(Noun)
                Verb_Predicted.append(np.argmax(pred_OF[k])+1)

        if num_batches%5==0:
            print("\nCurrent K-value:",K[z],", Batch(es) read:",num_batches)
        num_batches+=1

        i+=((num_classes_verbs*scale_factor) + num_classes_verbs)      

    if not model_pred:
        np.savez(
            "data/results/K_test_"+(str)(K[z])+"_Metrics.npz",
            a = np.array(ground_truth),
            b = np.array(RVS_Predicted),
            c = np.array(Predicted))
    else:
        np.savez(
            "data/results/K_test_"+(str)(K[z])+"_Metrics.npz",
            a = np.array(ground_truth),
            b = np.array(RVS_Predicted))
"""
"""
while i < total_samples:
    try:
        mag,angle,encoding = data_loader.load_file(i,modality="OF")

        rvs_checker.VerbSet = np.array(
            rvs_checker.rvs_generator.RVSGen(
                Noun_Pred=Nouns[i],
                K_Value=K,
                P_Noun_Verb=P_Noun_Verb))-1
    
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
        
        if (i+1)%95==0:
            i+=19
        else:
            i+=1
        
        #print("\nVerb Set: ",rvs_checker.VerbSet)
        #print("Ground Truth: ",init_Annot)
        #print("From activated Values: ",np.argmax(activated_values))
        #print("From feature vector values: ",np.argmax(pred2[0]))
        #print("From fully predicted values: ",np.argmax(pred1[0]))

        RVS_Predicted.append(np.argmax(activated_values))
        Predicted.append(np.argmax(pred1[0]))

        if (j+1)%100==0:
            print("\n\nFiles read:",i)
            print("Current Accuracy (with RVSGen):",np.mean(np.array(ground_truth)==np.array(RVS_Predicted)))
            print("Current Accuracy (without RVSGen):",np.mean(np.array(ground_truth)==np.array(Predicted)))
        
        j+=1
    
    except:
        print("Error while processing file index",i)
"""


#Verb_Probable = reduced_verb_space.RVSGen(Noun_Pred=Nouns[0],K_Value=10)

