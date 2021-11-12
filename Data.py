import numpy as np
import scipy.io as sio
import pandas as pd
import cv2
import random
import os
import math

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
        
        if modality=="RGB":
            modal = np.load(file_path,allow_pickle=True)["a"]
            Annotation = np.load(file_path,allow_pickle=True)["c"]
        else:
            file_in = np.load(file_path,allow_pickle=True)
            mag = np.array([element for (i,element) in enumerate(file_in['a'])]) 
            ang = np.array([element for (i,element) in enumerate(file_in['b'])]) 
            

            s1 = mag.shape[0]
            s2 = mag.shape[1]
            s3 = mag.shape[2]

            modal = np.zeros((s1,s2,s3*2))
            modal[:,:,:s3] = mag
            modal[:,:,s3:] = ang

            Annotation = np.load(file_path,allow_pickle=True)["d"]
        return modal,Annotation
    
    def get_frame_order(self,frames,modality):
        if modality=="OF":
            length = frame.shape[0]
        else:
            lenth = len(frames)
        interval_size = math.floor(length/self.fix_frames)
        j=0
        frame_indices=[]
        for i in range(self.fix_frames):
            frame_indices.append(j)
            j+=interval_size
        return frame_indices
    
    def read_flow(self,i,access_order,num_classes_total):
        Modal=[]
        Annotation=[]
        Val_Frame=[]
        Val_Noun=[]
        
        for j in range(i,i+num_classes_total):
            Modal,Annotation = self.load_file(access_order[j],modality="OF")
            #frame_indices = random.sample(population=[i for i in range(len(RGB))],k=self.fix_frames)
            frame_indices = self.get_frame_order(Modal,modality="OF")
            for count in range(self.fix_frames):
                #RGB_resized = cv2.resize(src=Modal[frame_indices[count]],dsize=self.input_shape)
                #RGB_normalized = cv2.normalize(RGB_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                if count==4:
                    Val_Frame.append(RGB_normalized)
                    Val_Noun.append((int)(Annotation[frame_indices[count]]))
                
                else:
                    Modal.append(RGB_normalized)
                    Annotation.append((int)(Annotation[frame_indices[count]]))
        
        return Modal, Annotation,Val_Frame,Val_Noun

    def read_frames(self,i,access_order,num_classes_total):    
        #random.seed(a=2)
        
        Frame=[]
        Y_Noun=[]
        Val_Frame=[]
        Val_Noun=[]
        
        for j in range(i,i+num_classes_total):
            RGB,Noun = self.load_file(access_order[j],modality="RGB")
            #frame_indices = random.sample(population=[i for i in range(len(RGB))],k=self.fix_frames)
            frame_indices = self.get_frame_order(RGB)
            for count in range(self.fix_frames):
                RGB_resized = cv2.resize(src=RGB[frame_indices[count]],dsize=self.input_shape)
                RGB_normalized = cv2.normalize(RGB_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                if count==4:
                    Val_Frame.append(RGB_normalized)
                    Val_Noun.append((int)(Noun[frame_indices[count]]))
                
                else:
                    Frame.append(RGB_normalized)
                    Y_Noun.append((int)(Noun[frame_indices[count]]))
        
        return Frame, Y_Noun,Val_Frame,Val_Noun

    def getTotal(self):
        return self.train.shape[0]

    def getClasses(self):
         return np.unique((self.train)["Noun"])
    
    def getInputShape(self):
        return self.input_shape

    def getAnnotations(self):
        return self.train["Noun"],self.train["Verb"]