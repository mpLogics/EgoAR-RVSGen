import tensorflow as tf
from keras import layers, models, applications
from keras.models import Sequential

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
        
        base_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            weights=self.modelWeights,
            classes=self.RGB_classes)
        
        base_model.trainable = self.base_trainable
        encoded_frame = TimeDistributed(Lambda(lambda x: base_model(x)))(video)
        encoded_pool = TimeDistributed(GlobalAveragePooling2D())(encoded_frame)
        encoded_vid = LSTM(256)(encoded_pool)
        ops = Dense(128, activation='relu')(encoded_vid)
        outputs = Dense(self.RGB_classes)(ops)
        activation = Activation("softmax")(outputs)
        model = tf.keras.models.Model(inputs=video,outputs=activation)
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
    

    def check_prev_trainings(self,model_weights):
        try:
            self.model.load_weights(model_weights)
            print("Saved weights restored!")
        except Exception:
            print("Saved weights could not be read.")

    def custom_train_model(self):
        #m2 = Model()
        L1 = LoadData()
        L1.train_test_splitNo = self.train_test_split 
        L1.fix_frames = self.fix_frames
        totalSamples = L1.getTotal()
        print("Total samples = ",totalSamples)
        #da = Data_Access()
        #da.random_flag = True
        #access_order = da.build_order()
        #self.model.summary()
        self.check_prev_trainings(model_weights="model_weights.h5")
        
        for epochs in range(self.Epochs+1):    
            da = Data_Access()
            da.random_flag = True
            access_order = da.build_order()
            
            print(access_order[:51])

            print("\nEpoch:",epochs)
            i = 0
            num_batches=0

            for i in range(0,totalSamples-(self.num_classes_total*3),self.num_classes_total*3):
                
                try:
                    X_train,Y_Noun,X_Val,Val_Noun = L1.read_rgb(i,access_order)
                except Exception:
                    print("Error reading files from index: ",i)
                
                if (np.unique(np.array(Y_Noun))).shape[0]<=50:
                    break
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
                    history = self.model.fit(np.array(X_train),Y,epochs=1)#,validation_data=(np.array(X_Val),Y_val))
                except Exception:
                    print("Unsuccessful training for",i)

            self.model.save_weights('model_weights.h5')
            #self.model.save("Noun_Predictor")
            print("Model weights save successful!")
        
        #self.plot_makker.makePlot(Loss_per_epoch,caption = "Loss Curve",sloc="Loss_vs_Epoch_"+ (str)(epochs)+ ".png")
        try:
            model.save('model_checkpoints/RGB_Noun.h5')
            print("Model trained successfully")
        except Exception:
            print("Model save unsuccessful")