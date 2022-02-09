import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D
from train import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras import backend as bk

import numpy as np
from Data import LoadData
from RVS_Check import RVS_Implement
import pandas as pd
from scipy import stats

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
        data_loader.mode="test"
        data_loader.fix_frames = 10
        noun_predictor.load_weights("model_weights.h5")
        print("Weights loaded")
        noun_predictor.summary()
        access_order = [i for i in range(total_samples)]
        print("Beginning Noun Prediction")
        batch_size = 100
        Noun_Predicted_top1 = []
        Noun_Predicted_top5 = []
        i=0
        num_batches=0
        err_ctr=0
        print("Total Samples:",total_samples)
        
        for i in range(0,total_samples-batch_size,batch_size):
            if num_batches%5 or num_batches%10==0:
                print("Files read:",i,", Ongoing batch size:",batch_size,", Batches completed:",num_batches)
            
            try:
                Frames,Y_Noun = data_loader.read_any_rgb(access_order,start_index=i,end_index=i+batch_size)
                X = np.array(Frames)
                #noun_predictor.evaluate(X,(np.array(Y_Noun)-1))
                pred = noun_predictor.predict(X)
            except:
                print("Error at file index",i)
                err_ctr+=1
            
            if err_ctr>=5:
                break
            
            print(len(pred))
            
            for j in range(len(pred)):                
                Noun_Predicted_top1.append(np.argmax(pred[j])+1)
                Noun_Predicted_top5.append(pred[j].argsort()[-5:][::-1]+1)
            
            print("Current Length of predictions (top 1):",len(Noun_Predicted_top1))
            print("Current Length of predictions (top 5):",len(Noun_Predicted_top5))
            
            num_batches+=1
        
        batch_size = 1
        
        for i in range(2000,total_samples,1):
            try:
                Frames,Y_Noun = data_loader.read_any_rgb(access_order,start_index=i,end_index=i+batch_size)
                X = np.array(Frames)
                #noun_predictor.evaluate(X,(np.array(Y_Noun)-1))
                pred = noun_predictor.predict(X)
            except:
                print("Error at file index",i)
                err_ctr+=1
            
            if err_ctr>=5:
                break
            
            print(len(pred))
            
            for j in range(len(pred)):                
                Noun_Predicted_top1.append(np.argmax(pred[j])+1)
                Noun_Predicted_top5.append(pred[j].argsort()[-5:][::-1]+1)
            
            print("Current Length of predictions (top 1):",len(Noun_Predicted_top1))
            print("Current Length of predictions (top 5):",len(Noun_Predicted_top5))
            print("Files read:",i,", Ongoing batch size:",batch_size,", Batches completed:",num_batches)
            
            num_batches+=1

        return np.array(Noun_Predicted_top1),np.array(Noun_Predicted_top5)
    
    def predict_verb(self,rvs_rules,verb_predictor,use_RVS,K_range,total_samples,nouns_with_path):
        access_order = [i for i in range(2022)]
        data_loader = LoadData()
        data_loader.mode = "test"
        data_loader.fix_frames = 5
        Verb_Predicted_top1 = []
        Verb_Predicted_top5 = []
        i=0
        num_batches=0
        err_ctr=0
        print("Total Samples:",total_samples)
        
        #try:
        #    Noun = np.load(nouns_with_path,allow_pickle=True)
        #except:
        #    print("No nouns found. Terminating!!!")
        #    exit()
        
        access_order = [i for i in range(total_samples)]
        batch_size = 100
        scale_factor = 1
        fix_frames = 5
        input_shape = (120,320,1)
        
        #base_model = verb_predictor.get_layer('dense_3').output 
        #feature_extractor = keras.Model(
        #    inputs = verb_predictor.input,
        #    outputs = base_model)

        #rvs_rules = RVS_Implement()
        #P_Noun,P_Verb,P_Noun_Verb = rvs_rules.set_verb_rules()

        i = 0
        #results = pd.DataFrame()
        #samples = pd.read_csv("data/Splits/test_split1.csv")
        #err_ctr = 0
        #Predicted_Verbs=[]
        #base_model = verb_predictor.get_layer('dense_1').output 
        #feature_extractor = keras.Model(
        #    inputs = verb_predictor.input,
        #    outputs = base_model)

        for i in range(0,total_samples-batch_size,batch_size):
            if num_batches%5 or num_batches%10==0:
                print("Files read:",i,", Ongoing batch size:",batch_size,", Batches completed:",num_batches)
            
            
            try:
                X_Value,Verb = data_loader.read_any_flow(
                    access_order,
                    start_index = i,
                    end_index = i + batch_size)
                
            except:
                print("Error at file index",i)
                err_ctr+=1
            
            if err_ctr>=5:
                break
            
            
            for j in range(len(pred)):                
                Verb_Predicted_top1.append(np.argmax(pred[j])+1)
                Verb_Predicted_top5.append(pred[j].argsort()[-5:][::-1]+1)
            
            print("Current Length of predictions (top 1):",len(Verb_Predicted_top1))
            print("Current Length of predictions (top 5):",len(Verb_Predicted_top5))
            
            num_batches+=1
        
        batch_size = 1
        
        for i in range(2000,total_samples,1):
            
            try:
                X_Value,Verb = data_loader.read_any_flow(
                    access_order,
                    start_index = i,
                    end_index = i + batch_size)
                pred = verb_predictor.predict(X_Value)
            except:
                print("Error at file index",i)
                err_ctr+=1
            
            if err_ctr>=5:
                break
            
            print(len(pred))
            
            for j in range(len(pred)):                
                Verb_Predicted_top1.append(np.argmax(pred[j])+1)
                Verb_Predicted_top5.append(pred[j].argsort()[-5:][::-1]+1)
            
            print("Current Length of predictions (top 1):",len(Verb_Predicted_top1))
            print("Current Length of predictions (top 5):",len(Verb_Predicted_top5))
            print("Files read:",i,", Ongoing batch size:",batch_size,", Batches completed:",num_batches)
            
            num_batches+=1

        return np.array(Verb_Predicted_top1),np.array(Verb_Predicted_top5)

        """
        while i <total_samples:
            #try:
                X_Value,Verb = data_loader.read_any_flow(
                    access_order,
                    start_index = i,
                    end_index = i + batch_size)
                
                
                verb_predictor.evaluate(X_Value,np.array(Verb))
                #pred_OF = verb_predictor.predict(X_OF)
                #feature_pred = feature_extractor.predict(X_OF)
                #for i in range(len(pred_OF)):
                #    Predicted_Verbs.append(np.argmax(pred_OF[i])+1)

            #except:
                print("Error at reading file",i)
                if (i+batch_size)>=total_samples:
                    batch_size = 1
                
                i+=batch_size
        
        
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
                feature_pred = feature_extractor.predict(X_OF)
                if use_RVS:
                    K = [i for i in range(K_range[0],K_range[1]+1)]
                    
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
        
        """
def test_my_model():
    total_samples = pd.read_csv("data/Splits/test_split1.csv")["FileName"].shape[0]

    t1 = Test_Experiments()
    rvs_rules=RVS_Implement()
    #m1 = Model()
    #noun_predictor = m1.Time_Distributed_Model()
    #noun_predictor = rvs_rules.get_noun_model()

    #Nouns,top_5 = t1.predict_noun(noun_predictor=noun_predictor,total_samples=total_samples)
    #np.savez("data/results/test_reports/Nouns.npz",a = Nouns,b=top_5)

    #session.close()
    #"""
    verb_predictor = rvs_rules.get_verb_model()
    verb_predictor.summary()
    Verbs,top_5 = t1.predict_verb(
        rvs_rules,
        verb_predictor,
        use_RVS=True,
        K_range=[1,14],
        total_samples=total_samples,
        nouns_with_path="data/results/test_reports/Nouns.npz")
    np.savez("data/results/test_reports/Verbs.npz",a=Verbs,b=top_5)

#Results.to_csv("data/results/Results.csv")
#"""
