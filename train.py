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
        
    def buildModel(self):
        base_model = keras.applications.InceptionV3(
            include_top=self.include_top,weights=self.modelWeights,
            input_tensor=self.inputTensor,input_shape=self.input_shape,
            pooling=self.pooling,classes=self.classes)
        
        base_model.trainable = self.base_trainable
        inputs = keras.Input(shape=self.input_shape)
        print(base_model.output)
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
        model = keras.Model(inputs,outputs)
        loss_func = keras.losses.SparseCategoricalCrossentropy()
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer,loss_func,metrics=["accuracy"])
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

class Data_Access():
    def __init__(self):
        self.df = pd.read_csv("data/Splits/train_split1.csv")
        self.range_classes = 53
        self.random_flag = True
        self.num_classes_total = 51
    
    def get_corrected(self,index):
        if index==16 or index==44:
            return -1
        elif index <=15:
            index-=1
        elif index >15 and index<=43:
            index-=2
        else:
            index-=3
        return index

    def get_index_lists(self,df):

        num_samples_list = []
        IndexLists = []
        IgnoreSet = [16,44]
        
        for i in range(1,54):
            if i in IgnoreSet:
                pass
            else:
                num_samples_list.append((i,len(list(df.get_group(i).index))))
                IndexLists.append(list(df.get_group(i).index))
        
        return num_samples_list,IndexLists

    def shuffle_indices(self,IndexLists):
        
        for i in range(len(IndexLists)):
            random.shuffle(IndexLists[i])
        
        return IndexLists

    def get_access_order(self,IndexLists,sorted_indices,sorted_classes):
        print("Obtaining access order")
        print("Random generation - Flag: ",self.random_flag)
        access_order=[]
        marked_indices = []
        if self.random_flag:
            index_lists = self.shuffle_indices(IndexLists)
        else:
            index_lists = IndexLists
            
        old_min_samples=0
        for k in range(sorted_indices.shape[0]):
            min_samples = sorted_indices[k]
            for j in range(old_min_samples,min_samples):
                for i in range(1,self.range_classes+1):
                    if i not in marked_indices:
                        corrected_sample_value = self.get_corrected(i)
                        if corrected_sample_value!=-1:
                            access_order.append(index_lists[corrected_sample_value][j])
            marked_indices.append(sorted_classes[k])
            old_min_samples=min_samples
        return access_order
    
    def build_order(self):
        df1 = pd.read_csv("data/Splits/train_split1.csv")
        df2 = df1.groupby(by="Noun")
        num_samples_list,IndexLists = self.get_index_lists(df2)
        num_samples=[]
        class_samples=[]
        
        for i in range(len(num_samples_list)):
            num_samples.append(num_samples_list[i][1])
            class_samples.append(num_samples_list[i][0])
            
        dtype = [('class', int), ('num_frames', int)]
        values = [num_samples_list]
        class_sample_pairs = np.array(values, dtype=dtype)
        sorted_class_sample_pairs = np.sort(class_sample_pairs,order='num_frames')
        sorted_indices = np.array([sorted_class_sample_pairs[0][i][1] for i in range(len(values[0]))])
        sorted_classes = np.array([sorted_class_sample_pairs[0][i][0] for i in range(len(values[0]))])
        
        return self.get_access_order(IndexLists,sorted_indices,sorted_classes)



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
    

    def custom_train_model(self,loss_func,optimizer):
        L1 = LoadData()
        L1.train_test_splitNo = self.train_test_split 
        L1.batch_size = self.batch_preprocess_size
        totalSamples = L1.getTotal()
        print("Total samples = ",totalSamples)
        Loss_per_epoch=[]
        access_order = Data_Access().build_order()
        
        
        for epochs in range(1,self.Epochs+1):    
            i = 0
            num_batches=0
            crt_batch = 0
            Frame=[]
            Y_Noun=[]
            diff=0
            plotter_flag = False
            Loss=[]
            print(self.model.summary())
            while i<totalSamples-1:
                """
                if diff==0:
                    i+=1
                    RGB,Noun = L1.load_file(access_order[i],modality="RGB")
                    try:
                        #RGB,Noun,num_frames,interval_size = L1.load_file(access_order[i])
                        RGB,Noun = L1.load_file(access_order[i],modality="RGB")
                        frame_indices = random.sample(population=[i for i in range(len(RGB))],k=self.fix_frames)
                        #print("Length of RGB: ",len(RGB))
                        #print("Length of Noun: ",len(Noun))
                        #print("Frame indices: ",frame_indices)
                        #print("Noun: ",Noun)
                    except Exception:
                        print("File index" + (str)(i) + " could not be read.")
                        i+=1

                diff,crt_batch,Frame,Y_Noun = L1.random_frame_load(diff,self.batch_preprocess_size,
                                                            crt_batch,
                                                            Frame,Y_Noun,
                                                            RGB,Noun,
                                                            len(frame_indices),
                                                            frame_indices)
                
                """
                if np.isnan(Frame).any():
                    print("Nan encountered. at file index",i)

                Frame,Y_Noun = L1.read_frames(i,access_order,self.num_classes_total)
                
                i+=self.num_classes_total
                if crt_batch == self.batch_preprocess_size  or i == totalSamples-1 or True:
                    print("\nClasses covered in batch: ",(np.unique(np.array(Y_Noun))).shape[0])
                    num_batches+=1
                    X = np.array(Frame)
                    Y_corrected = self.getCorrected(np.array(Y_Noun))
                    print(Y_corrected)
                    Y = tf.convert_to_tensor(Y_corrected)
                    print("Epoch",epochs,": Batch(es) read: ",num_batches)
                    print("Epoch",epochs,": Files read = ",i)                   
                    
                    self.model.fit(X,Y,epochs=self.Epochs,validation_split=0.2)
                    """
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
                    """
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