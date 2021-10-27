import numpy as np
import scipy.io as sio
import pandas as pd
import cv2
import random
import os

class LoadData():
    
    def __init__(self):
        self.base_input_path = "data/preprocessed_data/"
        self.train_test_splitNo = (("1","1"))
        self.train_split = "data/Splits/train_split" + self.train_test_splitNo[0] + ".csv"
        self.test_split = "data/Splits/test_split" + self.train_test_splitNo[1] + ".csv"
        self.batch_size = 25
        self.train = pd.read_csv(self.train_split)
        self.test = pd.read_csv(self.test_split)
        self.input_shape = (299,299)
        self.sample_rate = 0.1
        self.fix_frames = 10
        self.num_classes_total = 51


    def load_file(self,i,modality):
        file_path = "data/preprocessed_data/" + modality + "/" + self.train["FileName"][i] + ".npz"
        modal = np.load(file_path,allow_pickle=True)["a"]
        Annotation = np.load(file_path,allow_pickle=True)["c"]
        print("File Name: ",self.train["FileName"][i])
        print(modal.shape)
        print(modal[0].shape)
        print(Annotation)
        return modal,Annotation
    
    def read_frames(self,i,access_order,num_classes_total):    
        Frame=[]
        Y_Noun=[]
        for j in range(i,i+num_classes_total):
            RGB,Noun = self.load_file(access_order[j],modality="RGB")
            frame_indices = random.sample(population=[i for i in range(len(RGB))],k=self.fix_frames)
            for count in range(self.fix_frames):
                RGB_resized = cv2.resize(src=RGB[frame_indices[count]],dsize=self.input_shape)
                RGB_normalized = cv2.normalize(RGB_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                print(RGB_normalized)
                print(Noun[frame_indices[count]])
                Frame.append(RGB_normalized)
                Y_Noun.append((int)(Noun[frame_indices[count]]))
        return Frame, Y_Noun

    def getTotal(self):
        return self.train.shape[0]

    def getClasses(self):
         return np.unique((self.train)["Noun"])
    
    def getInputShape(self):
        return self.input_shape

    def getAnnotations(self):
        return self.train["Noun"],self.train["Verb"]