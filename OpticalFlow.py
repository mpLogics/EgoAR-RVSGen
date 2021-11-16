import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import CuDNNLSTM,Dense,Dropout,LSTM,Flatten,Conv2D,GlobalAveragePooling2D,TimeDistributed,Input,Activation


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
    
    def build_temporal_model(self):
        model = Sequential()
        #Izda.add(TimeDistributed(
        #    Convolution2D(40,3,3,border_mode='same'), input_shape=(sequence_lengths, 1,8,10)))
        model.add(
            TimeDistributed(
                Conv2D(32, (7, 7), padding='same', strides = 2),
                input_shape=(5, 240, 640, 1)))
        model.add(Activation('relu'))

        model.add(TimeDistributed(Conv2D(64, (5, 5), padding='same', strides = 2)))
        model.add(Activation('relu'))

        #model.add(TimeDistributed(MaxPooling2D((2,2), data_format = 'channels_first', name='pool1')))
        
        model.add(TimeDistributed(Conv2D(128, (5, 5), padding='same', strides = 2)))
        model.add(Activation('relu'))    
        
        model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same')))
        model.add(Activation('relu'))
        
        model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', strides = 2)))
        model.add(Activation('relu'))
        
        model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same')))
        model.add(Activation('relu'))
        
        model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', strides = 2)))
        model.add(Activation('relu'))    

        model.add(TimeDistributed(Conv2D(256, (3, 3), padding='same')))
        model.add(Activation('relu'))
        
        model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', strides = 2)))
        model.add(Activation('relu'))    
        
        #model.add(TimeDistributed(MaxPooling2D((2,2), data_format = 'channels_first', name='pool1')))    
        
        #model.add(TimeDistributed(Conv2D(32, (1, 1), data_format = 'channels_first')))
        #model.add(Activation('relu'))    
        
        model.add(TimeDistributed(Flatten()))
        
        #model.add(TimeDistributed(Dense(512, name="first_dense" )))
        
        #model.add(LSTM(num_classes, return_sequences=True))
        model.add(CuDNNLSTM(512 , return_sequences=True))
        #model.add(CuDNNLSTM(512 , return_sequences=True))
        #model.add(CuDNNLSTM(512 , return_sequences=True))
        model.add(CuDNNLSTM(512))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dense(19,activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'] )
        #metrics=['accuracy'])
        model.summary()
        self.temporal_extractor = model

    """
    def build_temporal_model(self):
        input_shape_network = (self.fix_frames-self.val_seq_size, 240, 640, 1)
        print(input_shape_network)
        model = Sequential()
        #model.add(Input(shape=(self.fix_frames-self.val_seq_size, 240, 640, 1)))
        model.add(TimeDistributed(Conv2D(16, (30,30),strides = (5,5)),input_shape=input_shape_network))
        #model.add(TimeDistributed(Conv2D(16,(5,5))))
        model.add(TimeDistributed(GlobalAveragePooling2D()))
        #model.add(CuDNNLSTM(5))
        #model.add(Dropout(0.2))
        #model.add(Flatten())
        model.add(Dense(19,activation="softmax"))
        
        #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_rate=1e-6)
        #optimizer = Adam(learning_rate=lr_schedule)

        model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'] )
        model.summary()
        self.temporal_extractor = model

    
    def build_temporal_model(self):
        #import si
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, LSTM, Flatten

        # Set Model
        classifier = Sequential()
        classifier.add(LSTM(128,input_shape=(480,640*2)))
        classifier.add(Flatten())
        classifier.add(Dense(19,activation="softmax"))


        # Compile model
        classifier.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        classifier.summary()
        self.temporal_extractor = classifier
    """
    def check_prev_trainings(self,model_name,modality):
        try:
            performance_metrics = np.load("data/performance_metrics/" + modality + "/Metrics.npz")
            saved_model = keras.models.load_model("model_name")
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
            print("\nEpoch:",epochs)
            i = 0
            num_batches = 0
            crt_batch = 0
            X_Value=[]
            Y_Value=[]
            diff = 0
            plotter_flag = False
            Loss=[]
            Accuracy=[]
            Val_Loss=[]
            Val_Acc=[]
            Val_Noun=[]
            Val_Frame=[]

            while i<totalSamples-1:

                X_Value,Y_Value,Val_Frame,Val_Verb = L1.read_flow(i,access_order,self.num_classes_total)
                print(X_Value.shape)
                print(Y_Value.shape)
                print(Val_Frame.shape)
                print(Val_Verb.shape)

                """
                try:
                    X_Value,Y_Value,Val_Frame,Val_Verb = L1.read_flow(i,access_order,self.num_classes_total)
                    print(X_Value.shape)
                    print(Y_Value.shape)
                    print(Val_Frame.shape)
                    print(Val_Verb.shape)
                except Exception:
                    print("Error reading files from index: ",i)
                """
                
                i+=self.num_classes_total
                
                # Logs
                print("\nClasses covered in batch: ",(np.unique(np.array(Y_Value))).shape[0])
                print("Batch(es) read: ",num_batches)
                print("Files read = ",i)                   

                num_batches+=1
                
                # Setting X and Y for training
                #X = np.array(X_Value)
                #X = X_Value
                #X_val = np.array(Val_Frame)
                #X_val = Val_Frame
                
                Y_test = []
                Y_corrected = self.getCorrected(np.array(Y_Value))
                for i in range(Y_corrected.shape[0]):
                    Y_test.append((Y_corrected[i],Y_corrected[i],Y_corrected[i],Y_corrected[i],Y_corrected[i]))
                    #for j in range(self.val_seq_size):
                        #Y_test.append(Y_corrected[i])
                    
                
                Y_test = np.array(Y_test)
                #print("Training set Y",Y_test.shape)
                Y = tf.convert_to_tensor(Y_corrected)
                #Y = Y_test
                Y_val_corrected = self.getCorrected(np.array(Val_Verb))
                
                
                Y_val_test=[]
                for i in range(Y_val_corrected.shape[0]):
                    Y_val_test.append((Y_val_corrected[i],Y_val_corrected[i],Y_val_corrected[i],Y_val_corrected[i],Y_val_corrected[i]))
                    #for j in range(self.val_seq_size):
                    #    Y_val_test.append(Y_val_corrected[i])    
                
                Y_val_test = np.array(Y_val_test)
                #print("Validation set Y",Y_val_test.shape)
                
                
                
                Y_val = tf.convert_to_tensor(Y_val_corrected)
                #Y_val = Y_val_test
                
                # Training batch
                X = np.reshape(X_Value,(self.num_classes_total,self.fix_frames-self.val_seq_size,240,640,1))
                X_val = np.reshape(Val_Frame,(self.num_classes_total,self.val_seq_size,240,640,1))
                
                #print(Y)
                #print(Y_Value)
                #print(Y_val)
                #print(Y_val_test)
                #print(Y_test)
                history = self.temporal_extractor.fit(X,Y,epochs=50,validation_data=(X_val,Y_val))
                train_succ=True
            
                #print("Unsuccessful training for",i)
                    #train_succ=False
                
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
                
                X_Value=[]
                Y_Value=[]
                Val_Verb=[]
                Val_Frame=[]
                crt_batch=0
                try:
                    if (num_batches+1)%30 == 0 and plotter_flag == False:
                        self.plot_makker.makePlot(Loss,caption = "Loss Curve",sloc = "data/Graphs/Loss_vs_Epoch_" + (str)(num_batches) + ".png")
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
        
        self.plot_makker.makePlot(Loss_per_epoch,caption = "Loss Curve",sloc="Loss_vs_Epoch_"+ (str)(epochs)+ ".png")
