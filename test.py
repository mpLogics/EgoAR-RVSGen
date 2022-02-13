import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D
from train import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras import backend as bk

from OpticalFlow import learn_optical_flow

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
                pred = noun_predictor.predict(X)
            except:
                print("Error at file index",i)
                err_ctr+=1
            
            if err_ctr>=5:
                break
            
            for j in range(len(pred)):                
                Noun_Predicted_top1.append(np.argmax(pred[j])+1)
                Noun_Predicted_top5.append(pred[j].argsort()[-5:][::-1]+1)
            
            #print("Current Length of predictions (top 1):",len(Noun_Predicted_top1))
            #print("Current Length of predictions (top 5):",len(Noun_Predicted_top5))
            
            num_batches+=1
        
        batch_size = 1
        
        for i in range(2000,total_samples,1):
            try:
                Frames,Y_Noun = data_loader.read_any_rgb(access_order,start_index=i,end_index=i+batch_size)
                X = np.array(Frames)
                pred = noun_predictor.predict(X)
            except:
                print("Error at file index",i)
                err_ctr+=1
            
            if err_ctr>=5:
                break
            
            for j in range(len(pred)):                
                Noun_Predicted_top1.append(np.argmax(pred[j])+1)
                Noun_Predicted_top5.append(pred[j].argsort()[-5:][::-1]+1)
            
            #print("Current Length of predictions (top 1):",len(Noun_Predicted_top1))
            #print("Current Length of predictions (top 5):",len(Noun_Predicted_top5))
            print("Files read:",i,", Ongoing batch size:",batch_size,", Batches completed:",num_batches)
            
            num_batches+=1

        return np.array(Noun_Predicted_top1),np.array(Noun_Predicted_top5)
    
    def get_action(self,P_Noun_Verb,noun_features,n_args,verb_features,v_args):
        
        max_rows = 5
        max_cols = 5
        
        #Obtaining ffv - final feature vector
        ffv = np.zeros((max_rows,max_cols))
        
        for i in range(max_rows):
            for j in range(max_cols):
                #print(P_Noun_Verb[(n_args[i]+1,v_args[j]+1)]*noun_features[i]*verb_features[j])
                ffv[i][j] = noun_features[n_args[i]]*verb_features[v_args[j]]
        
        predicted_value = np.argmax(ffv)
        i_pred = (int)(predicted_value/max_rows)
        j_pred = predicted_value % max_rows
        
        Noun = n_args[i_pred] + 1
        Verb = v_args[j_pred] + 1

        return Noun, Verb     
    
    def predict_action(self,noun_predictor,verb_predictor,total_samples):

        #noun_predictor.load_weights("model_weights.h5")
        #verb_predictor.load_weights("")
        Final_Nouns=[]
        Final_Verbs=[]

        rvs_rules = RVS_Implement()
        P_Noun,P_Verb,P_Noun_Verb = rvs_rules.set_verb_rules()

        noun_base_model = noun_predictor.get_layer('dense_1').output
        verb_base_model = verb_predictor.get_layer('dense_3').output 
        
        noun_feature_extractor = keras.Model(
            inputs = noun_predictor.input,
            outputs = noun_base_model)

        verb_feature_extractor = keras.Model(
            inputs = verb_predictor.input,
            outputs = verb_base_model)
        
        noun_loader = LoadData()
        noun_loader.mode = "train"
        noun_loader.fix_frames = 10
        
        verb_loader = LoadData()
        verb_loader.mode = "train"
        verb_loader.fix_frames = 5
        
        noun_predictor.summary()
        access_order = [i for i in range(total_samples)]
        batch_size = 100
        num_batches = 0
        
        err_ctr=0
        Noun_predicted_top5 = []
        for i in range(0,total_samples-batch_size,batch_size):
            
            if num_batches % 5 or num_batches % 10 == 0:
                print("Files read:",i,", Ongoing batch size:",batch_size,", Batches completed:",num_batches)

            try:
                X_RGB,Y_Noun = noun_loader.read_any_rgb(
                                                access_order,
                                                start_index=i,end_index=i+batch_size)
                X_RGB = np.array(X_RGB)

                X_Flow,Verb = verb_loader.read_any_flow(
                                                    access_order,
                                                    start_index = i,end_index = i + batch_size)
                    
                noun_pred = noun_feature_extractor.predict(X_RGB)
                verb_pred = verb_feature_extractor.predict(X_Flow)

                for i in range(len(noun_pred)):
                    # Top 5 noun feature vectors
                    n_args = noun_pred[i].argsort()[-5:][::-1]
                    # Top 5 verb feature vectors
                    v_args = verb_pred[i].argsort()[-5:][::-1]

                    Noun,Verb = self.get_action(
                        P_Noun_Verb=P_Noun_Verb,
                        noun_features=noun_pred[i],n_args=n_args,
                        verb_features=verb_pred[i],v_args=v_args)
                    
                    Final_Nouns.append(Noun+1)
                    Final_Verbs.append(Verb+1)                

            except:
                print("Error at index",i)
                err_ctr+=1

            if err_ctr>=5:
                break
        
        batch_size = 1
        for i in range(2000,total_samples,1):
            if num_batches % 5 or num_batches % 10 == 0:
                print("Files read:",i,", Ongoing batch size:",batch_size,", Batches completed:",num_batches)

            try:
                X_RGB,Y_Noun = noun_loader.read_any_rgb(
                                                access_order,
                                                start_index=i,end_index=i+batch_size)
                X_RGB = np.array(X_RGB)

                X_Flow,Verb = verb_loader.read_any_flow(
                                                    access_order,
                                                    start_index = i,end_index = i + batch_size)
                    
                noun_pred = noun_feature_extractor.predict(X_RGB)
                verb_pred = verb_feature_extractor.predict(X_Flow)

                for i in range(len(noun_pred)):
                    # Top 5 noun feature vectors
                    n_args = noun_pred[i].argsort()[-5:][::-1]
                    # Top 5 verb feature vectors
                    v_args = verb_pred[i].argsort()[-5:][::-1]

                    Noun,Verb = self.get_action(
                        P_Noun_Verb=P_Noun_Verb,
                        noun_features=noun_pred[i],n_args=n_args,
                        verb_features=verb_pred[i],v_args=v_args)
                    
                    Final_Nouns.append(Noun+1)
                    Final_Verbs.append(Verb+1)                

            except:
                print("Error at index",i)
                err_ctr+=1
        
        return Final_Nouns,Final_Verbs
    
    def set_data_output(self,total_samples,K_range):
        df = pd.DataFrame()
        K = [i for i in range(K_range[0],K_range[1]+1)]
        for z in K:
            field_name = "K_"+(str)(z)
            df[field_name] = [0 for i in range(total_samples)]
        return df    

    def verb_from_top_5(self,feature_extractor,Nouns,index,K_range,P_Noun_Verb,P_Verb,X,data_output,rvs_rules,P_Noun):
        feature_pred = feature_extractor.predict(X)
        K = [i for i in range(K_range[0],K_range[1]+1)]
        
        Verb_Set = rvs_rules.rvs_generator.getVerbSet()
        V = len(Verb_Set)
        
        for i in range(len(feature_pred)):
            for K_Value in K:
                P_YVerb = {i:0 for i in Verb_Set}
                
                for j in Nouns[index+i]:
                    Noun_Pred = j
                    #print("Current Noun",Noun_Pred)
                    for k in range(V):
                        P_YVerb[Verb_Set[k]] += (P_Noun_Verb[(Noun_Pred,Verb_Set[k])]*P_Noun[j])
                Final_Probabilities = dict(sorted(P_YVerb.items(), key = lambda kv: kv[1]))
                Verb_Probable = list(Final_Probabilities.keys())[-K_Value:]
                
                rvs_rules.VerbSet = np.array(Verb_Probable)
                
                activated_values = rvs_rules.custom_activation(
                                    x=feature_pred[i],
                                    P_Verb=P_Verb)
                
                field_name = "K_"+(str)(K_Value)
                data_output[field_name][index+i] = np.argmax(activated_values)+1

        return data_output

    def get_RVS(self,feature_extractor,Nouns,index,K_range,P_Noun_Verb,P_Verb,X,data_output,rvs_rules):
        feature_pred = feature_extractor.predict(X)
        K = [i for i in range(K_range[0],K_range[1]+1)]
        
        for i in range(len(feature_pred)):        
            for K_Value in K:
                final_verb_set=[]
                for j in Nouns[index+i]:
                    VerbSet = np.array(
                        rvs_rules.rvs_generator.RVSGen(
                            Noun_Pred=j,
                            K_Value=K_Value,
                            P_Noun_Verb=P_Noun_Verb,
                            P_Verb=P_Verb))-1
                    final_verb_set = list(set(final_verb_set) | set(VerbSet))
                rvs_rules.VerbSet = final_verb_set

                activated_values = rvs_rules.custom_activation(
                                    x=feature_pred[i],
                                    P_Verb=P_Verb)
                
                field_name = "K_"+(str)(K_Value)
                data_output[field_name][index+i] = np.argmax(activated_values)+1
        return data_output

    def get_RVS_Verb(self,feature_extractor,K_range,index,P_Noun_Verb,P_Verb,X,Noun,data_output,rvs_rules):
        feature_pred = feature_extractor.predict(X)
        K = [i for i in range(K_range[0],K_range[1]+1)]
        
        for i in range(len(feature_pred)):        
            for z in K:
                rvs_rules.VerbSet = np.array(
                    rvs_rules.rvs_generator.RVSGen(
                        Noun_Pred=Noun[index+i],
                        K_Value=z,
                        P_Noun_Verb=P_Noun_Verb,
                        P_Verb=P_Verb))-1
                
                activated_values = rvs_rules.custom_activation(
                    x=feature_pred[i],
                    P_Verb=P_Verb)
                
                field_name = "K_"+(str)(z)
                data_output[field_name][index+i] = np.argmax(activated_values)+1

        return data_output

    def predict_verb(self,rvs_rules,verb_predictor,use_RVS,K_range,total_samples,nouns_with_path):
        Nouns = pd.read_csv("data/Splits/test_split1.csv")["Noun"]
        #Nouns = np.load("data/results/test_reports/Nouns.npz",allow_pickle=True)
        #learn_optical_flow.
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
        
        access_order = [i for i in range(total_samples)]
        batch_size = 100
        scale_factor = 1
        fix_frames = 5
        input_shape = (120,320,1)
        
        base_model = verb_predictor.get_layer('dense_1').output 
        feature_extractor = keras.Model(
            inputs = verb_predictor.input,
            outputs = base_model)

        rvs_rules = RVS_Implement()
        P_Noun,P_Verb,P_Noun_Verb = rvs_rules.set_verb_rules()
        print("Verb rules obtained")
        data_output = self.set_data_output(total_samples,K_range)

        
        for i in range(0,total_samples-batch_size,batch_size):
            if num_batches%5 or num_batches%10==0:
                print("Files read:",i,", Ongoing batch size:",batch_size,", Batches completed:",num_batches)

            try:
                X_Value,Verb = data_loader.read_any_flow(
                    access_order,
                    start_index = i,
                    end_index = i + batch_size)
                pred = verb_predictor.predict(X_Value)
                
                #data_output = self.get_RVS(
                #                        feature_extractor=feature_extractor,Nouns=Nouns['b'],
                #                        index=i,K_range=[1,19],
                #                        P_Noun_Verb=P_Noun_Verb,P_Verb=P_Verb,
                #                       X=X_Value,data_output=data_output,rvs_rules=rvs_rules)            
    
                #data_output = self.verb_from_top_5(
                #                        feature_extractor=feature_extractor,Nouns=Nouns['b'],
                #                        index=i,K_range=[1,19],
                #                        P_Noun_Verb=P_Noun_Verb,P_Verb=P_Verb,P_Noun=P_Noun,
                #                        X=X_Value,data_output=data_output,rvs_rules=rvs_rules)
                
                data_output = self.get_RVS_Verb(
                                        data_output=data_output,
                                        rvs_rules=rvs_rules,
                                        feature_extractor=feature_extractor,
                                        K_range=[1,19],index=i,
                                        P_Noun_Verb=P_Noun_Verb,P_Verb=P_Verb,
                                        X=X_Value,Noun=Nouns)

            
            except:
                print("Error at file index",i)
                err_ctr+=1
            
            if err_ctr>=5:
                exit()
            
            
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
                #data_output = self.get_RVS(
                #                        feature_extractor=feature_extractor,Nouns=Nouns['b'],
                #                        index=i,K_range=[1,19],
                #                        P_Noun_Verb=P_Noun_Verb,P_Verb=P_Verb,
                #                        X=X_Value,data_output=data_output,rvs_rules=rvs_rules)


                #data_output = self.verb_from_top_5(
                #                        feature_extractor=feature_extractor,Nouns=Nouns['b'],
                #                        index=i,K_range=[1,19],
                #                        P_Noun_Verb=P_Noun_Verb,P_Verb=P_Verb,P_Noun=P_Noun,
                #                        X=X_Value,data_output=data_output,rvs_rules=rvs_rules)
                # 
                data_output = self.get_RVS_Verb(
                                        data_output=data_output,
                                        rvs_rules=rvs_rules,
                                        feature_extractor=feature_extractor,
                                        K_range=[1,19],index=i,
                                        P_Noun_Verb=P_Noun_Verb,P_Verb=P_Verb,
                                        X=X_Value,Noun=Nouns)

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
        data_output.to_csv("Predicted Verbs.csv",index=False)

        return np.array(Verb_Predicted_top1),np.array(Verb_Predicted_top5)
        
def test_my_model():
    total_samples = pd.read_csv("data/Splits/test_split1.csv")["FileName"].shape[0]

    t1 = Test_Experiments()
    rvs_rules=RVS_Implement()
    
    #m1 = Model()
    #noun_predictor = m1.Time_Distributed_Model()
    
    noun_predictor = rvs_rules.get_noun_model("model_weight_633s.h5")
    verb_predictor = rvs_rules.get_verb_model()
    
    #Nouns,top_5 = t1.predict_noun(noun_predictor=noun_predictor,total_samples=total_samples)
    
    Nouns,Verbs = t1.predict_action(
        noun_predictor=noun_predictor,
        verb_predictor=verb_predictor,
        total_samples=total_samples)
    
    np.savez("Actions.npz",a = Nouns,b=Verbs)
    
    #session.close()
    #"""
    #verb_predictor = rvs_rules.get_verb_model()
    #verb_predictor.summary()
    #Verbs,top_5 = t1.predict_verb(
    #    rvs_rules,
    #    verb_predictor,
    #    use_RVS=True,
    #    K_range=[1,19],
    #    total_samples=total_samples,
    #    nouns_with_path="data/results/test_reports/Nouns.npz")
    #np.savez("data/results/test_reports/Verbs.npz",a=Verbs,b=top_5)

#Results.to_csv("data/results/Results.csv")
#"""
test_my_model()