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
        self.input_shape = (120,160,3)
        self.classes = np.count_nonzero(GVS().getNounSet())
        self.base_trainable = False
        self.include_top = False
        self.modelWeights = "imagenet"
        self.pooling = None
        self.activation = "softmax"
        self.inputTensor = None 
        
    def buildModel(self):
        base_model = keras.applications.InceptionV3(
            include_top=self.include_top,weights=self.modelWeights,
            input_tensor=self.inputTensor,input_shape=self.input_shape,
            pooling=self.pooling,classes=self.classes,
            classifier_activation="softmax",
        )
        
        base_model.trainable = self.base_trainable
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(units=self.classes,
                                     name="Predictions",
                                     kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5,l2=1e-4),
                                     bias_regularizer=keras.regularizers.l2(1e-4),
                                     activity_regularizer=keras.regularizers.l2(1e-5))(x)
    
        print("Total classes = ",self.classes)
        model = keras.Model(inputs,outputs)
        loss_func = keras.losses.SparseCategoricalCrossentropy()
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer,loss_func)
        model.summary()
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

    
    def custom_train_model(self,loss_func,optimizer):
        L1 = LoadData()
        L1.train_test_splitNo = self.train_test_split 
        L1.batch_size = self.batch_preprocess_size
        totalSamples = L1.getTotal()
        print("Total samples = ",totalSamples)
        #print("Batch_size = ",self.batch_preprocess_size)
        Loss_per_epoch=[]

        for epochs in range(1,self.Epochs+1):    
            
            i = -1
            num_batches=0
            crt_batch = 0
            Frame=[]
            Y_Noun=[]
            diff=0
            access_order = [i for i in range(8299)]
            random.shuffle(access_order)
            plotter_flag = False
            Loss=[]

            while i<totalSamples-1:
                if diff==0:
                    i+=1
                    try:
                        #RGB,Noun,num_frames,interval_size = L1.load_file(access_order[i])
                        RGB,Noun = L1.load_file(access_order[i])
                        frame_indices = random.sample(population=[i for i in range(len(RGB))],k=self.fix_frames)
                    except Exception:
                        print("File index" + (str)(i) + " could not be read.")
                        i+=1

                diff,crt_batch,Frame,Y_Noun = L1.random_frame_load(diff,self.batch_preprocess_size,
                                                            crt_batch,
                                                            Frame,Y_Noun,
                                                            RGB,Noun,
                                                            len(frame_indices),
                                                            frame_indices)
                
                if np.isnan(Frame).any():
                    print("Nan encountered. at file index",i)

                if crt_batch == self.batch_preprocess_size  or i == totalSamples-1:
                    print("\nClasses covered in batch: ",np.count_nonzero(np.unique(np.array(Y_Noun))))
                    num_batches+=1
                    X = np.array(Frame)
                    Y = tf.convert_to_tensor(np.array(Y_Noun)-1)
                    print("Epoch",epochs,": Batch(es) read: ",num_batches)
                    print("Epoch",epochs,": Files read = ",i)                   
                    
                    self.model.fit(X,Y,epochs=self.Epochs)
                    
                    with tf.GradientTape() as Tape:
                        y_pred = self.model(X,training=True)
                        loss = loss_func(Y,y_pred)
                    gradients = Tape.gradient(loss,self.model.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))  
                    plotter_flag=False
                    Loss.append(loss)
                    Prediction_values = np.argmax(y_pred,axis=1)
                    #Printing logs
                    print("Epoch",epochs,": Batch ",num_batches," training complete.")
                    print("Epoch",epochs,": Loss Value: ",loss)
                    print("Epoch",epochs,": Avg Loss: ",np.mean(np.array(Loss)))
                    print("Epoch",epochs,": Length of loss = ",len(Loss))
                    print("Epoch",epochs,": Accuracy: ",(np.sum(Prediction_values==Y)/self.batch_preprocess_size)*100)
                    
                    Frame=[]
                    Y_Noun=[]
                    crt_batch=0
                
                if (num_batches+1)%30==0 and plotter_flag==False:
                    Visualizer.makePlot(Loss,caption = "Loss Curve",sloc = "data/Graphs/Loss_vs_Epoch_" + (str)(num_batches) + ".png")
                    print((str)(i) + " examples trained")
                    plotter_flag=True
                
            Loss_per_epoch.append(np.mean(np.array(Loss)))
            
            if epochs%2==0:
                try:
                    filename = "model_checkpoints/RGB_"+(str)(epochs)+".h5"
                    model.save(filename)
                except Exception:
                    print("Model Checkpoint save unsuccessful!")
        
        Visualizer.makePlot(Loss_per_epoch,caption = "Loss Curve",sloc="Loss_vs_Epoch_"+ (str)(epochs)+ ".png")
        try:
            model.save('model_checkpoints/RGB_Noun.h5')
            print("Model trained successfully")
        except Exception:
            print("Model save unsuccessful")