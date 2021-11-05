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
        #inputs = keras.Input(shape=self.input_shape)
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
        model = keras.Model(base_model.input,outputs)
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
    

    def check_prev_trainings(self):
        
        try:
            performance_metrics = np.load("data/performance_metrics/Metrics.npz")
            saved_model = keras.models.load_model("Noun_Predictor")
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