import numpy as np
import tensorflow as tf
#from tf.python.keras.models import Model, load_model
from tensorflow.python.keras.models import Model, load_model
from tensorflow.keras.applications import inception_v3
#from keras.models import Sequential
from tensorflow.keras.layers import Input, GlobalAveragePooling2D,Lambda, LSTM,TimeDistributed,Dense,Activation
from Data import LoadData,Data_Access
from visualization import Visualizer


class Model():
    def __init__(self):
        self.RGB_input_shape = (240,320,3)
        self.RGB_classes = 53
        self.base_trainable = False
        self.include_top = False
        self.modelWeights = "imagenet"
        self.pooling = None
        self.activation = "softmax"
        self.inputTensor = None 
        self.model_modality = None
        self.temporal_extractor = None
        self.spatial_extractor = None
        self.fixed_frames = 10
        
    def Time_Distributed_Model(self):
        video = Input(
            shape=(
                self.fixed_frames, 
                self.RGB_input_shape[0],
                self.RGB_input_shape[1],
                self.RGB_input_shape[2]),
                name='video_input')
        
        base_model = inception_v3.InceptionV3(
            include_top=False,
            weights="imagenet",
            classes=self.RGB_classes)
        
        base_model.trainable = self.base_trainable
        encoded_frame = TimeDistributed(Lambda(lambda x: base_model(x)))(video)
        encoded_pool = TimeDistributed(GlobalAveragePooling2D())(encoded_frame)
        encoded_vid = LSTM(256)(encoded_pool)
        ops = Dense(128, activation='relu')(encoded_vid)
        outputs = layers.Dense(self.RGB_classes)(ops)
        activation = Activation("softmax")(outputs)
        model = Model(inputs=[video],outputs=activation)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
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
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.spatial_extractor.compile(optimizer,loss_func,metrics=["accuracy"])
    
        return loss_func,optimizer

class Train():
    def __init__(self):
        self.train_test_split = (("1","1"))
        self.model_save_path = ("saved_models")
        self.Epochs = 60
        self.input_shape = (240,320,3)
        self.base_trainable = False
        self.include_top = False    
        self.modelWeights = "imagenet"
        self.pooling = None
        self.activation = "softmax"
        self.inputTensor = None
        self.fix_frames = 10
        self.model = "None"
        self.num_classes_total = 53
        self.plot_makker = Visualizer()
    
    def getCorrected(self,Y):
        return Y - 1
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
            saved_model = load_model(model_name)
        except Exception:
            print("Saved model could not be read.")
            return None,0,[],[],[],[]
        
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
        print("6 values")
        return saved_model, epochs_completed, Loss_per_epoch, Accuracy_per_epoch, Val_Loss_per_epoch, Val_Acc_per_epoch

    def custom_train_model(self):
        L1 = LoadData()
        L1.train_test_splitNo = self.train_test_split 
        L1.fix_frames = self.fix_frames
        #L1.batch_size = self.batch_preprocess_size
        totalSamples = L1.getTotal()
        print("Total samples = ",totalSamples)
        Loss_per_epoch=[]
        Accuracy_per_epoch=[]
        Val_Loss_per_epoch=[]
        Val_Acc_per_epoch=[]
        da = Data_Access()
        da.random_flag = False
        access_order = da.build_order()
        train_succ=False
        saved_model,epochs_completed,Loss_per_epoch,Accuracy_per_epoch,Val_Loss_per_epoch,Val_Acc_per_epoch = self.check_prev_trainings(model_name="Noun_Predictor",modality="RGB")
        if saved_model==None:
            pass
        else:
            self.model=saved_model
        
        self.model.save("Noun_Predictor")
        print("Epochs completed =",epochs_completed)
        self.model.summary()
        
        for epochs in range(epochs_completed+1,self.Epochs+1):    
            #self.plot_makker.plot_metrics(m_path="data/performance_metrics/Metrics.npz",Epoch=epochs-1)
            print("\nEpoch:",epochs)
            i = 0
            num_batches=0
            Frame=[]
            Y_Noun=[]
            plotter_flag = False
            Loss=[]
            Accuracy=[]
            Val_Loss=[]
            Val_Acc=[]
            Val_Noun=[]

            for i in range(0,totalSamples-(self.num_classes_total*3),self.num_classes_total*3):
                if np.isnan(Frame).any():
                    print("Nan encountered. at file index",i)
                
                try:
                    X_train,Y_Noun,X_Val,Val_Noun = L1.read_rgb(i,access_order)
                    #Frame,Y_Noun,Val_Frame,Val_Noun = L1.read_frames(i,access_order,self.num_classes_total)
                    #X_train,Y_Noun,X_Val,Val_Noun = L1.read_rgb(i,access_order)
                except Exception:
                    print("Error reading files from index: ",i)
                
                #i+=(self.num_classes_total*3)
                #i+=self.num_classes_total
                
                
                #if crt_batch == self.batch_preprocess_size  or i == totalSamples-1 or True:
                
                # Logs
                print("\nClasses covered in batch: ",(np.unique(np.array(Y_Noun))).shape[0])
                print("Batch(es) read: ",num_batches)
                print("Files read = ",i)                   

                num_batches+=1
                
                # Setting X and Y for training
                #X = np.array(Frame)
                #X_val = np.array(Val_Frame)
                

                Y_corrected = self.getCorrected(np.array(Y_Noun))
                Y = tf.convert_to_tensor(Y_corrected)
                
                Y_val_corrected = self.getCorrected(np.array(Val_Noun))
                Y_val = tf.convert_to_tensor(Y_val_corrected)
                
                
                
                # Training batch
                try:
                    history = self.model.fit(np.array(X_train),Y,epochs=1,validation_data=(np.array(X_Val),Y_val))
                    train_succ=True
                except Exception:
                    print("Unsuccessful training for",i)
                    train_succ=False

                if train_succ:
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
                
                try:
                    if (num_batches+1)%30==0 and plotter_flag==False:
                        self.plot_makker.makePlot(Loss,caption = "Loss Curve",sloc = "data/Graphs/Loss_vs_Epoch_" + (str)(num_batches) + ".png")
                        print((str)(i) + " examples trained")
                        plotter_flag=True
                except Exception:
                    print("Plot saving unsuccessful!")
                
            for i in range(num_batches*self.num_classes_total*3,totalSamples,4):
                try:
                    X_train,Y_Noun = L1.read_any_rgb(access_order,start_index=i,end_index=i+3)
                    X_Val,Val_Noun = L1.read_any_rgb(access_order,start_index=i+3,end_index=i+4)
                except Exception:
                    print("Error reading files from index: ",i)
                
                # Logs
                print("\nClasses covered in batch: ",(np.unique(np.array(Y_Noun))).shape[0])
                print("Batch(es) read: ",num_batches)
                print("Files read = ",i)                   

                num_batches+=1
                
                # Setting X and Y for training
                X = np.array(X_train)
                X_val = np.array(X_Val)
                

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

                if train_succ:
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
            
            Loss_per_epoch.append(np.mean(np.array(Loss)))
            Accuracy_per_epoch.append(np.mean(np.array(Accuracy)))
            Val_Loss_per_epoch.append(np.mean(np.array(Val_Loss)))
            Val_Acc_per_epoch.append(np.mean(np.array(Val_Acc)))
            
            np.savez("data/performance_metrics/Metrics.npz",
            a = Loss_per_epoch,b=Accuracy_per_epoch,
            c = Val_Loss_per_epoch,d=Val_Acc_per_epoch)

            self.model.save("Noun_Predictor")
            print("Model save successful!")
        
        #self.plot_makker.makePlot(Loss_per_epoch,caption = "Loss Curve",sloc="Loss_vs_Epoch_"+ (str)(epochs)+ ".png")
        try:
            model.save('model_checkpoints/RGB_Noun.h5')
            print("Model trained successfully")
        except Exception:
            print("Model save unsuccessful")