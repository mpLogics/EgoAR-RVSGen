from cv2 import INTER_MAX
from Data import LoadData
import sys
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from train import Model,Train
import json
from OpticalFlow import learn_optical_flow

def getMinFrameValue():
    min_value = 10000
    min_arg = 0
    L1 = LoadData()
    totalSamples = L1.getTotal()

    for i in range(totalSamples):
        try:
            RGB,Noun,num_frames,interval_size = L1.load_file(i)
        except Exception:
            print("File index ",i," could not be read.")
        if len(RGB)<min_value:
            min_value = len(RGB)
            min_arg = i
        for j in range(len(RGB)):
            if np.isnan(np.array(RGB[j])).any():
                print("Nan values at file index ",i+1)
        print("Files read: ",i)

    print("Min frames at file index : ",i)
    print("Min number of frames : ",min_value)


config_file = open("config.json")
config_values = json.load(config_file)["Configuration Values"]

#"""
m1 = Model()
m1.RGB_input_shape = (config_values["train"]["input_shape_x"],
                config_values["train"]["input_shape_y"],
                config_values["train"]["input_shape_z"]
                )
m1.base_trainable = config_values["train"]["base_trainable"]
#m1.pooling = config_values["train"]["pooling"]
m1.modelWeights = config_values["train"]["modelWeights"]
m1.activation = config_values["train"]["activation"]
m1.include_top = config_values["train"]["include_top"]
model = m1.Time_Distributed_Model()
#model,loss_func,optimizer = m1.buildModel()

t1 = Train()
t1.fix_frames = config_values["train"]["frames_to_be_extracted"]
#t1.batch_preprocess_size = config_values["train"]["batch_preprocess_size"]
t1.Epochs = config_values["train"]["Epochs"]
t1.model = model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#t1.trainModel(model,loss_func,optimizer)

print("Training metrics")
print("Epochs: ",t1.Epochs)

t1.custom_train_model()
session.close()
#"""

#Training Optical Flow
"""
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

m2 = learn_optical_flow()
#m2.build_temporal_model()
m2.convLSTM_model()
#m2.debug()
m2.train()


session.close()
"""