import numpy as np
from numpy.core.numeric import NaN
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.ops.variables import trainable_variables
from Data import LoadData
from RVSGen import GenVerbSpace as GVS
import pandas as pd
import os
import random
from visualization import Visualizer
import datetime

class Model():
    def __init__(self):
        self.input_shape = (299,299,3)
        self.classes = 51
        self.base_trainable = False
        self.include_top = False
        self.modelWeights = "imagenet"
        self.pooling = None
        self.activation = "softmax"
        self.inputTensor = None 
        self.model_modality = None
        self.temporal_extractor = None
        self.spatial_extractor = None
        
    def buildModel(self):
        base_model = keras.applications.InceptionV3(
            include_top=self.include_top,weights=self.modelWeights,
            input_tensor=self.inputTensor,input_shape=self.input_shape,
            pooling=self.pooling,classes=self.classes)
        
        base_model.trainable = self.base_trainable
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(units=self.classes,name="Predictions",activation="softmax")(x)
        #outputs = keras.layers.Dense(units=self.classes,
        #                             name="Predictions",
        #                             activation = "softmax",
        #                             kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5,l2=1e-4),
        #                             bias_regularizer=keras.regularizers.l2(1e-4),
        #                             activity_regularizer=keras.regularizers.l2(1e-5))(x)
    
        print("Total classes = ",self.classes)
        self.spatial_extractor = keras.Model(base_model.input,outputs)
        loss_func = keras.losses.SparseCategoricalCrossentropy()
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.spatial_extractor.compile(optimizer,loss_func,metrics=["accuracy"])
    
        self.temporal_extractor = Sequential()
        #Izda.add(TimeDistributed(
        #    Convolution2D(40,3,3,border_mode='same'), input_shape=(sequence_lengths, 1,8,10)))
        self.temporal_extractor.add(TimeDistributed(Conv2D(32, (7, 7), padding='same', strides = 2),
                input_shape=(5, 240, 640, 1)),activation='relu')
        self.temporal_extractor.add(Activation('relu'))

        self.temporal_extractor.add(TimeDistributed(Conv2D(64, (5, 5), padding='same', strides = 2)))
        self.temporal_extractor.add(Activation('relu'))
        
        self.temporal_extractor.add(TimeDistributed(Conv2D(128, (5, 5), padding='same', strides = 2)))
        self.temporal_extractor.add(Activation('relu'))    
        
        self.temporal_extractor.add(TimeDistributed(Conv2D(128, (3, 3), padding='same')))
        self.temporal_extractor.add(Activation('relu'))
        
        self.temporal_extractor.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', strides = 2)))
        self.temporal_extractor.add(Activation('relu'))
        
        self.temporal_extractor.add(TimeDistributed(Conv2D(256, (3, 3), padding='same')))
        self.temporal_extractor.add(Activation('relu'))
        
        self.temporal_extractor.add(TimeDistributed(Conv2D(256, (3, 3), padding='same', strides = 2)))
        self.temporal_extractor.add(Activation('relu'))    

        self.temporal_extractor.add(TimeDistributed(Conv2D(256, (3, 3), padding='same')))
        self.temporal_extractor.add(Activation('relu'))
        
        model.add(TimeDistributed(Conv2D(512, (3, 3), padding='same', strides = 2)))
        model.add(Activation('relu'))    
        
        model.add(TimeDistributed(Flatten()))
        model.add(CuDNNLSTM(512 , return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(512))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2)) 
        model.add(Dense(19,activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'] )
        model.summary()
        self.temporal_extractor = model
        return model,loss_func,optimizer

class Filter():
    def __init__(self):
        self.K_Range = (5,20)
        self.K_Default = 0
        self.totalSamples = GVS.getTotalSamples()
        self.Noun = pd.read_csv(self.Noun_path)
        self.P_Noun = GVS.calProbNouns
        self.P_Verb = GVS.calProbVerbs
        self.P_Noun_Verb = GVS.calProbVerbs(totalSamples=self.totalSamples)
        self.Verb = GVS.getVerbSet()
    
    
    #def applyFilter(self):
    #    GVS.RVSGen(self.P_Noun_Verb,self.P_Noun,self.P_Verb,V,Noun_Pred,K_Value,self.Verb)
    #self.Noun = pd.read_csv





class Train():
    def __init__(self):
        self.train_test_split = (("1","1"))
        self.model_save_path = ("saved_models")
        self.batch_preprocess_size = 25
        self.Epochs = 60
        self.classes = np.count_nonzero(GVS().getNounSet())
        self.input_shape = (120,160,3)
        self.base_trainable = False
        self.include_top = False    
        self.modelWeights = "imagenet"
        self.pooling = None
        self.activation = "softmax"
        self.inputTensor = None
        self.fix_frames = 15
        self.model = "None"
        self.num_classes_total = 51
        self.plot_makker = Visualizer()
    
    def getCorrected(self,Y):
        Y_corrected = np.copy(Y)
        for i in range(Y.shape[0]):
            if Y[i]<=15:
                Y_corrected[i]-=1
            elif Y[i]>15 and Y[i]<=43:
                Y_corrected[i]-=2
            else:
                Y_corrected[i]-=3
        return Y_corrected
    

    def check_prev_trainings(self,model_name,modality):
        try:
            performance_metrics = np.load("data/performance_metrics/" + modality + "/Metrics.npz")
            saved_model = keras.models.load_model("model_name")
        except Exception:
            print("Saved model could not be read.")
            return 1,[],[],[],[]
        
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
        
        #self.model = keras.models.load_model("Noun_Predictor")

        return saved_model,epochs_completed,Loss_per_epoch,Accuracy_per_epoch,Val_Loss_per_epoch,Val_Acc_per_epoch

    def custom_train_model(self,loss_func,optimizer):
        L1 = LoadData()
        L1.train_test_splitNo = self.train_test_split 
        L1.batch_size = self.batch_preprocess_size
        totalSamples = L1.getTotal()
        print("Total samples = ",totalSamples)
        Loss_per_epoch=[]
        Accuracy_per_epoch=[]
        Val_Loss_per_epoch=[]
        Val_Acc_per_epoch=[]
        access_order = Data_Access().build_order()
        #print(self.model.summary())
        train_succ=False
        self.model,epochs_completed,Loss_per_epoch,Accuracy_per_epoch,Val_Loss_per_epoch,Val_Acc_per_epoch = self.check_prev_trainings()
        
        print("Epochs completed =",epochs_completed)
        
        

        for epochs in range(epochs_completed+1,self.Epochs+1):    
            self.plot_makker.plot_metrics(m_path="data/performance_metrics/Metrics.npz",Epoch=epochs-1)
            print("\nEpoch:",epochs)
            i = 0
            num_batches=0
            crt_batch = 0
            Frame=[]
            Y_Noun=[]
            diff=0
            plotter_flag = False
            Loss=[]
            Accuracy=[]
            Val_Loss=[]
            Val_Acc=[]
            Val_Noun=[]
            Val_Frame=[]

            while i<totalSamples-1:
                if np.isnan(Frame).any():
                    print("Nan encountered. at file index",i)
                
                try:
                    Frame,Y_Noun,Val_Frame,Val_Noun = L1.read_frames(i,access_order,self.num_classes_total)
                except Exception:
                    print("Error reading files from index: ",i)
                
                i+=self.num_classes_total
                
                
                #if crt_batch == self.batch_preprocess_size  or i == totalSamples-1 or True:
                
                # Logs
                print("\nClasses covered in batch: ",(np.unique(np.array(Y_Noun))).shape[0])
                print("Batch(es) read: ",num_batches)
                print("Files read = ",i)                   

                num_batches+=1
                
                # Setting X and Y for training
                X = np.array(Frame)
                X_val = np.array(Val_Frame)
                

                Y_corrected = self.getCorrected(np.array(Y_Noun))
                Y = tf.convert_to_tensor(Y_corrected)
                
                Y_val_corrected = self.getCorrected(np.array(Val_Noun))
                Y_val = tf.convert_to_tensor(Y_val_corrected)
                
                
                # Training batch
                try:
                    history = self.model.fit(X,Y,epochs=1,validation_data=(X_val,Y_val))
                    train_succ=True
                except Exception:
                    print("Unsuccessful training for",i)
                    train_succ=False
                
                if train_succ==True:
                    # Collecting Metrics
                    Loss.append(history.history['loss'])
                    Accuracy.append(history.history['accuracy'])
                    Val_Loss.append(history.history['val_loss'])
                    Val_Acc.append(history.history['val_accuracy'])
                
                    # Displaying Metrics
                    print("Average Loss: ",np.mean(np.array(Loss)))
                    print("Average Accuracy: ",np.mean(np.array(Accuracy)))
                    print("Average Validation Loss: ",np.mean(np.array(Val_Loss)))
                    print("Average Validation Accuracy: ",np.mean(np.array(Val_Acc)))
                
                Frame=[]
                Y_Noun=[]
                Val_Noun=[]
                Val_Frame=[]
                crt_batch=0
                try:
                    if (num_batches+1)%30==0 and plotter_flag==False:
                        self.plot_makker.makePlot(Loss,caption = "Loss Curve",sloc = "data/Graphs/Loss_vs_Epoch_" + (str)(num_batches) + ".png")
                        print((str)(i) + " examples trained")
                        plotter_flag=True
                except Exception:
                    print("Plot saving unsuccessful!")
                
            Loss_per_epoch.append(np.mean(np.array(Loss)))
            Accuracy_per_epoch.append(np.mean(np.array(Accuracy)))
            Val_Loss_per_epoch.append(np.mean(np.array(Val_Loss)))
            Val_Acc_per_epoch.append(np.mean(np.array(Val_Acc)))
            
            np.savez("data/performance_metrics/Metrics.npz",
            a = Loss_per_epoch,b=Accuracy_per_epoch,
            c = Val_Loss_per_epoch,d=Val_Acc_per_epoch)

            self.model.save("Noun_Predictor")
            print("Model save successful!")
        
        self.plot_makker.makePlot(Loss_per_epoch,caption = "Loss Curve",sloc="Loss_vs_Epoch_"+ (str)(epochs)+ ".png")
        try:
            model.save('model_checkpoints/RGB_Noun.h5')
            print("Model trained successfully")
        except Exception:
            print("Model save unsuccessful")