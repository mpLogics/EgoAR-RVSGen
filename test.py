import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras import backend as bk

import numpy as np
from Data import LoadData
from RVS_Check import RVS_Implement
import pandas as pd

class Test_Experiments():
    def __init__(self):
        data_loader = LoadData()
        data_loader.mode = "test"
                
    def reverse_annot(self,Y):
        if Y<=15:
            Y+=1
        elif Y>15 and Y<=43:
            Y+=2
        else:
            Y+=3
        return Y

    def predict_noun(self,noun_predictor,total_samples):
        data_loader = LoadData()
        data_loader.mode = "test"

        access_order = [i for i in range(total_samples)]
        print("Beginning Noun Prediction")
        batch_size = 100
        Noun_Predicted = []
        i=0
        while i < total_samples:
            try:
                Frame = data_loader.read_frames(
                                i,
                                access_order,
                                batch_size)
                X_RGB = np.array(Frame)

                pred_RGB = noun_predictor.predict(X_RGB)
                
                for k in range(len(pred_RGB)):
                    Noun = self.reverse_annot(np.argmax(pred_RGB[k]))
                    Noun_Predicted.append(Noun)
            except:
                print("Error encountered at index:",i)
            
            if (i+batch_size)>=total_samples:
                batch_size=1
            i+=batch_size
        return np.array(Noun_Predicted)
    
    def predict_verb(self,rvs_rules,verb_predictor,use_RVS,K_range,total_samples,nouns_with_path):
        data_loader = LoadData()
        data_loader.mode = "test"

        try:
            Noun = np.load(nouns_with_path,allow_pickle=True)
        except:
            print("No nouns found. Terminating!!!")
            exit()
        
        access_order = [i for i in range(total_samples)]
        batch_size = 100
        scale_factor = 1
        fix_frames = 5
        input_shape = (120,320,1)
        
        if use_RVS:
            rvs_rules = RVS_Implement()
            P_Noun,P_Verb,P_Noun_Verb = rvs_rules.set_verb_rules()
            base_model = verb_predictor.get_layer('dense_3').output 
            feature_extractor = keras.Model(
                inputs = verb_predictor.input,
                outputs = base_model)

        i = 0
        results = pd.DataFrame()
        samples = pd.read_csv("data/Splits/test_split1.csv")
        err_ctr=0
        while i < total_samples:
            try:
                X_Value = data_loader.read_val_flow(
                    i,
                    access_order,
                    num_classes=batch_size,
                    multiply_factor=scale_factor)
                
                X_OF = np.reshape(X_Value,(
                    batch_size*scale_factor,
                    fix_frames,
                    input_shape[0],
                    input_shape[1],
                    input_shape[2]))
                
                pred_OF = verb_predictor.predict(X_OF)
                
                if use_RVS:
                    K = [i for i in range(K_range[0],K_range[1]+1)]
                    feature_pred = feature_extractor.predict(X_OF)
                    
                    for k in range(len(feature_pred)):
                        
                        df = pd.DataFrame()
                        df["GT_Verb"]=[samples["Verb"][i+k]]
                        df["GT_Noun"]=[samples["Noun"][i+k]]
                        df["Verb_Predicted"] = [np.argmax(pred_OF[k])+1]
                        
                        for z in range(len(K)):
                            rvs_rules.VerbSet = np.array(
                                rvs_rules.rvs_generator.RVSGen(
                                    Noun_Pred=Noun[i+k],
                                    K_Value=K[z],
                                    P_Noun_Verb=P_Noun_Verb,
                                    P_Verb=P_Verb))-1
                            
                            activated_values = rvs_rules.custom_activation(
                                x=feature_pred[k],
                                P_Verb=self.P_Verb)
                            
                            field_name = "RVS Verb(K="+(str)(K[z])
                            df[field_name] = [np.argmax(activated_values)+1]
                        
                        results.append(df)        
                if (i+batch_size)>=total_samples:
                    batch_size = 1
                
                i+=batch_size
            except:
                if err_ctr>=5:
                    break
                print("Error encountered at index:",i)
                err_ctr+=1
        return results

total_samples = pd.read_csv("data/Splits/test_split1.csv")["FileName"].shape[0]


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

t1 = Test_Experiments()
rvs_rules=RVS_Implement()
noun_predictor = rvs_rules.get_noun_model()

Nouns = t1.predict_noun(noun_predictor=noun_predictor,total_samples=total_samples)
np.savez("data/results/test_reports/Nouns.npz",a = Nouns)

session.close()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

verb_predictor = rvs_rules.get_verb_model()
Results = t1.predict_verb(
    rvs_rules,
    verb_predictor,
    use_RVS=True,
    Noun=Nouns,
    K_range=[1,14],
    total_samples=total_samples,
    nouns_with_path="data/results/test_reports/Nouns.npz")

Results.to_csv("data/results/Results.csv")

session.close()

