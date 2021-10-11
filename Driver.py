from preprocessing import SetAnnotations,PreProcessing
import pandas as pd
import json
from MessageLogging import StoreLogs
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

config_file = open("config.json")
config_values = json.load(config_file)["Configuration Values"]

try:
    if config_values["setannotation"]["runValue"]==True:
        base_path = input("Enter path to annotations folder. \n(For default path, enter DEFAULT \n (default path - current_dir/action_annotation/): ")
        SA = SetAnnotations()
        SA.totCombs=3
        if base_path.upper()=="DEFAULT":
            pass
        else:
            SA.base = base_path
        SA.run()

    if config_values["preprocess"]["runValue"]==True:
        train = []
        test = []

        for i in range(3):
            train.append(pd.read_csv("Splits/train_split" + (str)(i+1) +".csv"))
            test.append(pd.read_csv("Splits/test_split" + (str)(i+1) +".csv"))
        preProcessed_Data = PreProcessing()
        preProcessed_Data.preProcess(train,test)    
        preProcessed_Data.batch_size = 15
        preProcessed_Data.tiers = 3
        preProcessed_Data.totCombs = 3

    if config_values["visualize"]["runValue"]==True:
        from visualization import Visualizer        
        v1 = Visualizer()
        v1.file_path=(str)(config_values["visualize"]["basepath"] + config_values["visualize"]["filepath"])
        print(v1.file_path)
        v1.ExtractFrames()
    
    if config_values["train"]["runValue"]==True:
        from train import Model,Train

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        m1 = Model()
        m1.input_shape = (config_values["train"]["input_shape_x"],
                        config_values["train"]["input_shape_y"],
                        config_values["train"]["input_shape_z"]
                        )
        m1.base_trainable = config_values["train"]["base_trainable"]
        m1.pooling = config_values["train"]["pooling"]
        m1.modelWeights = config_values["train"]["modelWeights"]
        m1.activation = config_values["train"]["activation"]
        m1.include_top = config_values["train"]["include_top"]
        model,loss_func,optimizer = m1.buildModel()
        
        t1 = Train()
        t1.batch_preprocess_size = config_values["train"]["batch_preprocess_size"]
        t1.Epochs = config_values["train"]["Epochs"]
        t1.trainModel(model,loss_func,optimizer)
        
        session.close()

except Exception as Argument:
    L1 = StoreLogs()
    L1.fileName="Logs.txt"
    L1.mode="a"
    L1.msg="Error Detected"
    print(Exception)
    print(Argument)
    L1.des=str(Argument)
    L1.LogMessages()
