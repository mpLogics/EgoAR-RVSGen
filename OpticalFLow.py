import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import CuDNNLSTM,Dense,Dropout, LSTM,Flatten
import numpy as np
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from Data import LoadData

class Data_Access():
    def __init__(self):
        self.df = pd.read_csv("data/Splits/train_split1.csv")
        self.range_classes = 19
        self.random_flag = True
        self.num_classes_total = 19
    
    def get_index_lists(self,df):

        num_samples_list = []
        IndexLists = []
        #IgnoreSet = [16,44]
        
        for i in range(1,20):
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
        df2 = df1.groupby(by="Verb")
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
        
class learn_optical_flow():
    def __init__(self):
        self.input_shape = None
        self.num_classes_total = 19
        self.temporal_extractor = None
    
    def build_temporal_model(self):
        self.temporal_extractor = Sequential()
        self.temporal_extractor.add(CuDNNLSTM(10,input_shape=(480,640),return_sequences=True))
        self.temporal_extractor.add(Dropout(0.2))
        self.temporal_extractor.add(Flatten())
        self.temporal_extractor.add(Dense(4,activation="softmax"))
        self.temporal_extractor.compile( loss='sparse_categorical_crossentropy',
                optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),
                metrics=['accuracy'] )
        self.temporal_extractor.summary()
    
    def check_prev_trainings(self,model_name,modality):
        try:
            performance_metrics = np.load("data/performance_metrics/" + modality + "/Metrics.npz")
            saved_model = keras.models.load_model("model_name")
        except Exception:
            print("Saved model could not be read.")
            return 0,[],[],[],[]
        
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

    def train(self):
        L1 = LoadData()
        modality="OF"
        L1.train_test_splitNo = self.train_test_split 
        L1.batch_size = self.batch_preprocess_size
        totalSamples = L1.getTotal()
        print("Total samples = ",totalSamples)
        Loss_per_epoch=[]
        Accuracy_per_epoch=[]
        Val_Loss_per_epoch=[]
        Val_Acc_per_epoch=[]
        access_order = Data_Access().build_order()
        print(self.model.summary())
        train_succ=False
        self.temporal_extractor,epochs_completed,Loss_per_epoch,Accuracy_per_epoch,Val_Loss_per_epoch,Val_Acc_per_epoch = self.check_prev_trainings(modality="OF",model_name="Verb_Predictor")
        
        print("Epochs completed =",epochs_completed)

        for epochs in range(epochs_completed+1,self.Epochs+1):    
            if epochs!=1:
                self.plot_makker.plot_metrics(m_path="data/performance_metrics/"+ modality +"/Metrics.npz",Epoch=epochs-1)
            print("\nEpoch:",epochs)
            i = 0
            num_batches=0
            crt_batch = 0
            X_Value=[]
            Y_Value=[]
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
                    X_Value,Y_Value,Val_Frame,Val_Verb = L1.read_frames(i,access_order,self.num_classes_total,modality="OF")
                except Exception:
                    print("Error reading files from index: ",i)
                
                i+=self.num_classes_total
                
                # Logs
                print("\nClasses covered in batch: ",(np.unique(np.array(Y_Value))).shape[0])
                print("Batch(es) read: ",num_batches)
                print("Files read = ",i)                   

                num_batches+=1
                
                # Setting X and Y for training
                X = np.array(X_Value)
                X_val = np.array(Val_Frame)
                

                #Y_corrected = self.getCorrected(np.array(Y_Noun))
                Y = tf.convert_to_tensor(Y_Value)
                
                #Y_val_corrected = self.getCorrected(np.array(Val_Noun))
                Y_val = tf.convert_to_tensor(Val_Verb)
                
                
                # Training batch
                try:
                    history = self.temporal_extractor.fit(X,Y,epochs=1,validation_data=(X_val,Y_val))
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
                
                X_Value=[]
                Y_Value=[]
                Val_Verb=[]
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
            
            np.savez("data/performance_metrics/" + modality + "/Metrics.npz",
            a = Loss_per_epoch,b=Accuracy_per_epoch,
            c = Val_Loss_per_epoch,d=Val_Acc_per_epoch)

            self.temporal_extractor.save("Verb_Predictor")
            print("Model save successful!")
        
        self.plot_makker.makePlot(Loss_per_epoch,caption = "Loss Curve",sloc="Loss_vs_Epoch_"+ (str)(epochs)+ ".png")

        


"""
classifier = Sequential()
classifier.add(CuDNNLSTM(64,input_shape=(480,640),return_sequences=True))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(4,activation="softmax"))
classifier.compile( loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001, decay=1e-6),
        metrics=['accuracy'] )
classifier.summary()

session.close()
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

classifier.fit(x=X,y=Y,epochs=10)
session.close()


class learn_optical_flow():
    def __init__(self):
        self.A=1
        self.input_shape=None
    
    def model():
        classifier = Sequential()
        classifier.add(CuDNNLSTM(128,input_shape=self.input_shape),return_sequences=True)
        classifier.add(Dropout(0.2))
        classifier.add(CuDNNLSTM(128))
        classifier.add(Dense(64,activation="relu"))
        classifier.add(Dropout(0.2))
        classifier.add(Dense(19,activation="softmax"))
        classifier.compile( loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'] )
        classifier.summary()

    def train():
    

"""