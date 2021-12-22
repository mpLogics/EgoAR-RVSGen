import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import CuDNNLSTM,Dense,Dropout,LSTM,Flatten,Conv2D,GlobalAveragePooling2D,TimeDistributed,Input,Activation,MaxPooling2D,ConvLSTM2D


import numpy as np
import os
import math
import pandas as pd
import random
import datetime


from Data import LoadData,Data_Access
from visualization import Visualizer
from RVSGen import GenVerbSpace as GVS

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class learn_optical_flow():
    def __init__(self):
        self.input_shape = None
        self.num_classes_total = 19
        self.temporal_extractor = None
        self.train_test_split = (("1","1"))
        self.batch_preprocess_size = 510
        self.Epochs=60
        self.fix_frames = 10
        self.val_seq_size = 5
        self.plot_makker = Visualizer()
    
    def convLSTM_model(self):
        model = Sequential()
        model.add(ConvLSTM2D(filters = 32, kernel_size = (5, 5), return_sequences = True, data_format = "channels_last", input_shape = (5, 120, 320, 1)))
        model.add(Dropout(0.5))
        model.add(ConvLSTM2D(filters = 16, kernel_size = (3, 3), return_sequences = True))
        model.add(Dropout(0.2))
        model.add(ConvLSTM2D(filters = 8, kernel_size = (3, 3), return_sequences = False))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(19, activation = "softmax"))
        model.summary()
        #opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.temporal_extractor = model

    def check_prev_trainings(self,model_name,modality):
        try:
            saved_model = keras.models.load_model(model_name)
            performance_metrics = np.load("data/performance_metrics/" + modality + "/Metrics.npz")
        except Exception:
            print("Saved model could not be read.")
            return None,0,[],[],[],[]
        
        #performance_metrics = np.load("data/performance_metrics/Metrics.npz")
        #self.model = keras.models.load_model("Noun_Predictor")
        
        epochs_completed = performance_metrics['a'].shape[0]
        Loss_per_epoch=[]
        Accuracy_per_epoch=[]
        Val_Loss_per_epoch=[]
        Val_Acc_per_epoch=[]

        for i in range(performance_metrics['a'].shape[0]):
            Loss_per_epoch.append(performance_metrics['a'][i])
            Accuracy_per_epoch.append(performance_metrics['b'][i])
            Val_Loss_per_epoch.append(performance_metrics['c'][i])
            Val_Acc_per_epoch.append(performance_metrics['d'][i])

        return saved_model,epochs_completed,Loss_per_epoch,Accuracy_per_epoch,Val_Loss_per_epoch,Val_Acc_per_epoch

    def getCorrected(self,Y):
        Y_corrected = np.copy(Y)
        return Y_corrected-1

    def train(self):
        L1 = LoadData()
        modality="OF"
        L1.train_test_splitNo = self.train_test_split 
        L1.batch_size = self.batch_preprocess_size
        totalSamples = L1.getTotal()
        print("Total samples = ",totalSamples)
        accessor = Data_Access()
        accessor.modality="OF"
        Loss_per_epoch=[]
        Accuracy_per_epoch=[]
        Val_Loss_per_epoch=[]
        Val_Acc_per_epoch=[]
        access_order = accessor.build_order()
        train_succ=False
        saved,epochs_completed,Loss_per_epoch,Accuracy_per_epoch,Val_Loss_per_epoch,Val_Acc_per_epoch = self.check_prev_trainings(modality="OF",model_name="Verb_Predictor")
        if saved==None:
            pass
        else:
            self.temporal_extractor = saved
        print("Epochs completed =",epochs_completed)

        for epochs in range(epochs_completed+1,self.Epochs+1):    
            if epochs!=1:
                self.plot_makker.plot_metrics(m_path="data/performance_metrics/"+ modality +"/Metrics.npz",Epoch=epochs-1)
            print("\nEpoch",epochs)
            i = 0
            num_batches = 0
            X_Value=[]
            Y_Value=[]
            plotter_flag = False
            Loss=[]
            Accuracy=[]
            Val_Loss=[]
            Val_Acc=[]

            while i<totalSamples-1:

                try:
                    X_Value,Y_Value,Val_Frame,Val_Verb = L1.read_flow(i,access_order,self.num_classes_total)
                    print(X_Value.shape)
                    print(Y_Value.shape)
                    print(Val_Frame.shape)
                    print(Val_Verb.shape)
                except Exception:
                    print("File read unsuccessful at index:",i)
                    break

                print(Val_Frame)
                print(Val_Verb)
                
                i+=self.num_classes_total 
                
                # Logs
                print("\nClasses covered in batch: ",(np.unique(np.array(Y_Value))).shape[0])
                print("Batch(es) read: ",num_batches)
                print("Files read = ",i)                   

                num_batches+=1
                
                Y_corrected = self.getCorrected(np.array(Y_Value))
                    
                
                Y = tf.convert_to_tensor(Y_corrected)
                Y_val_corrected = self.getCorrected(np.array(Val_Verb))
                
                
                Y_val = tf.convert_to_tensor(Y_val_corrected)
                
                # Training batch
                X = np.reshape(X_Value,(self.num_classes_total*2,self.fix_frames-self.val_seq_size,120,320,1))
                X_val = np.reshape(Val_Frame,(self.num_classes_total*2,self.val_seq_size,120,320,1))
                
                history = self.temporal_extractor.fit(X,Y,epochs=1,validation_data=(X_val,Y_val))
                train_succ=True
                
                if train_succ==True:
                    # Collecting Metrics
                    Loss.append(history.history['loss'][0])
                    Accuracy.append(history.history['accuracy'][0])
                    Val_Loss.append(history.history['val_loss'][0])
                    Val_Acc.append(history.history['val_accuracy'][0])
                
                    # Displaying Metrics
                    print("Average Loss: ",np.mean(np.array(Loss)))
                    print("Average Accuracy: ",np.mean(np.array(Accuracy)))
                    print("Average Validation Loss: ",np.mean(np.array(Val_Loss)))
                    print("Average Validation Accuracy: ",np.mean(np.array(Val_Acc)))
                    train_succ=False
                else:
                    print("Training unsuccessful!")
                
                X_Value=[]
                Y_Value=[]
                Val_Verb=[]
                Val_Frame=[]
                
                try:
                    if (num_batches+1)%30 == 0 and plotter_flag == False:
                        self.plot_makker.makePlot(Loss,caption = "Loss Curve",sloc = "data/Graphs/OF/Loss_vs_Epoch_" + (str)(num_batches) + ".png")
                        print((str)(i) + " examples trained")
                        plotter_flag=True
                        
                except Exception:
                    print("Plot saving unsuccessful!")
                
            Loss_per_epoch.append(np.mean(np.array(Loss)))
            Accuracy_per_epoch.append(np.mean(np.array(Accuracy)))
            Val_Loss_per_epoch.append(np.mean(np.array(Val_Loss)))
            Val_Acc_per_epoch.append(np.mean(np.array(Val_Acc)))
            
            np.savez("data/performance_metrics/" + modality + "/Metrics.npz",
            a = Loss_per_epoch,b=Accuracy_per_epoch,
            c = Val_Loss_per_epoch,d=Val_Acc_per_epoch)

            self.temporal_extractor.save("Verb_Predictor")
            print("Model save successful!")
        
        self.plot_makker.makePlot(Loss_per_epoch,caption = "Loss Curve",sloc="OF_Loss_vs_Epoch_"+ (str)(epochs)+ ".png")
        try:
            self.temporal_extractor.save('OF_Verb.h5')
            print("Model trained successfully")
        except Exception:
            print("Model save unsuccessful")