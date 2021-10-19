import numpy as np
import scipy.io as sio
import pandas as pd
import cv2
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

    
    def load_file(self,i):
        file_path = "data/preprocessed_data/" + self.train["FileName"][i] + ".mat"
        RGB = sio.loadmat(file_path)["RGB"][0]
        Noun = sio.loadmat(file_path)["Noun"][0]    
        #num_frames = (int)(len(RGB)*(self.sample_rate)) + 1
        #interval_size = round((int)(len(RGB)/num_frames))
        return RGB,Noun #num_frames,interval_size

    
    def random_frame_load(self,diff,max_batch_size,crt_batch,Frame,Y_Noun,RGB,Noun,num_frames,frame_indices):

        for count in range(num_frames-diff):
            if crt_batch+1<=max_batch_size:
                    crt_batch+=1
                    diff+=1
                    RGB_resized = cv2.resize(src=RGB[frame_indices[count]],dsize=self.input_shape)
                    RGB_normalized = cv2.normalize(RGB_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    Frame.append(RGB_normalized)
                    Y_Noun.append((int)(Noun[frame_indices[count]]))
            else:
                    break
                
        if diff==num_frames:
            diff = 0
        return diff,crt_batch,Frame,Y_Noun    

    def load_data(self,diff,max_batch_size,crt_batch,Frame,Y_Noun,RGB,Noun,num_frames,interval_size):
        j = interval_size*diff
        for count in range(num_frames-diff):
            if crt_batch+1<=max_batch_size:
                crt_batch+=1
                diff+=1
                RGB_resized = cv2.resize(src=RGB[j],dsize=self.input_shape)
                #RGB_normalized = np.zeros(self.input_shape)
                #print(RGB_normalized)
                RGB_normalized = cv2.normalize(RGB_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #RGB_normalized = cv2.normalize(RGB_resized, RGB_normalized,-1,1,cv2.NORM_MINMAX)
                #print(RGB_normalized)
                Frame.append(RGB_normalized)
                Y_Noun.append((int)(Noun[j]))
            else:
                break
            j+=interval_size
        if diff==num_frames:
            diff = 0
        return diff,crt_batch,Frame,Y_Noun

    def load_train_RGBFrame(self,begin_batch):
        Frame=[]
        Y_Noun=[]
        #if begin_batch%self.batch_size==0:
        #    print("Samples trained: "+(str)(begin_batch))


        for i in range(begin_batch,begin_batch + self.batch_size):
            try:
                file_path = "data/preprocessed_data/" + self.train["FileName"][i] + ".mat"
                RGB = sio.loadmat(file_path)["RGB"][0]
                Noun = sio.loadmat(file_path)["Noun"][0]
            
                if len(RGB)%10==0:
                    num_frames = (int)(len(RGB)/10)
                else:
                    num_frames = (int)(len(RGB)/10) + 1
                interval_size = (int)(len(RGB)/num_frames)

                for j in range(0,len(RGB),interval_size):
                    RGB_resized = cv2.resize(src=RGB[j],dsize=(160,120))
                    RGB_normalized = np.zeros((120,160))
                    RGB_normalized = cv2.normalize(RGB_resized, RGB_normalized,0,255,cv2.NORM_MINMAX)
                    Frame.append(RGB_normalized)
                    Y_Noun.append((int)(Noun[j]))
            except Exception:
                print("File index could not be read: ",i)

        return np.array(Frame), np.array(Y_Noun)
    
    def load_test_RGBFrame(self):
        Frame=[]
        Y_Noun=[]
        
        for i in range((self.begin_batch,self.begin_batch + self.batch_size)):
            file_path = "data/preprocessed_data/" + self.test["FileName"][i] + ".mat"
            try: 
                RGB = sio.loadmat(file_path)["RGB"][0]
                Noun = sio.loadmat(file_path)["Noun"]
                for j in range(self.train["FileName"].shape[0]):
                    Frame.append(RGB[j])
                    Y_Noun.append(Noun[j])
            except Exception:
                print(file_path)
        
        return Frame,Y_Noun, Frame[0].shape
    
    def getTotal(self):
        return self.train.shape[0]

    def getClasses(self):
         return np.unique((self.train)["Noun"])
    
    def getInputShape(self):
        return self.input_shape

    def getAnnotations(self):
        return self.train["Noun"],self.train["Verb"]