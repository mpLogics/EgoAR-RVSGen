import numpy as np
import scipy.io as sio
import pandas as pd
import cv2
import random
import os
import math

class Data_Access():
    
    def __init__(self):
        
        self.df = pd.read_csv("data/Splits/train_split1.csv")
        
        self.range_classes_noun = 53
        self.num_classes_noun_total = 51
        
        self.range_classes_verb = 19
        self.num_classes_verb_total = 19
        
        self.random_flag = True
        self.modality = "RGB"
    
    def get_corrected(self,index):
        
        if self.modality=="OF":
            return index-1
        else:
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
        
        if self.modality=="OF":
            IgnoreSet=[]
            range_annot = self.range_classes_verb
        else:
            IgnoreSet=[16,44]
            range_annot = self.range_classes_noun
        
        num_samples_list = []
        IndexLists = []

        for i in range(1,range_annot+1):
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
        
        if self.modality=="RGB":
            marked_indices = [16,44]
            range_classes = self.range_classes_noun
        else:
            marked_indices = []
            range_classes = self.range_classes_verb
        
        if self.random_flag:
            index_lists = self.shuffle_indices(IndexLists)
        else:
            index_lists = IndexLists
            
        old_min_samples=0
        
        for k in range(sorted_indices.shape[0]):
            min_samples = sorted_indices[k]
            for j in range(old_min_samples,min_samples):
                for i in range(1,range_classes+1):
                    if i not in marked_indices:
                        corrected_sample_value = self.get_corrected(i)
                        if corrected_sample_value!=-1:
                            access_order.append(index_lists[corrected_sample_value][j])
            marked_indices.append(sorted_classes[k])
            old_min_samples=min_samples
        return access_order
    
    def build_order(self):
        df1 = pd.read_csv("data/Splits/train_split1.csv")
        if self.modality=="RGB":
            df2 = df1.groupby(by="Noun")
        else:
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

class LoadData():
    
    def __init__(self):
        self.base_input_path = "data/preprocessed_data/"
        self.train_test_splitNo = (("1","1"))
        self.train_split = "data/Splits/train_split" + self.train_test_splitNo[0] + ".csv"
        self.test_split = "data/Splits/test_split" + self.train_test_splitNo[1] + ".csv"
        self.train = pd.read_csv(self.train_split)
        self.test = pd.read_csv(self.test_split)
        self.mode = "train"
        self.input_shape = (299,299)
        self.sample_rate = 0.1
        self.fix_frames = 5
        self.num_classes_total = 53
        
    def get_any_matrix(self,Mag,Angle,Encoding):
        interval_size = math.floor(len(Mag)/self.fix_frames)
        Annotations=[]
        j = 0
        
        Mag[0] = Mag[0]#[120:240,160:320]
        Angle[0] = Angle[0]#[120:240,160:320]
        prev_matrix = np.concatenate([Mag[0],Angle[0]],axis=1)
        init_matrix = np.reshape(prev_matrix,(1,prev_matrix.shape[0],prev_matrix.shape[1]))

        for k in range(1,self.fix_frames):
            if j<=0:
                j+=interval_size
            
            Mag[j] = Mag[j]
            Angle[j] = Angle[j]
            
            prev_matrix = np.concatenate([Mag[j],Angle[j]],axis=1)
            temp = np.reshape(prev_matrix,(1,prev_matrix.shape[0],prev_matrix.shape[1]))
            init_matrix = np.concatenate([init_matrix,temp])
            
            j+=interval_size    
        
        if self.mode=="train":
            Annotations.append((int)(Encoding[0]))
            return init_matrix,np.array(Annotations)
        else:
            return init_matrix

    def load_file(self,i,modality):
        if self.mode=="train":
            file_path = "data/preprocessed_data/" + modality + "/" + self.train["FileName"][i] + ".npz"
        else:
            file_path = "data/preprocessed_data/" + modality + "/" + self.test["FileName"][i] + ".npz"
        
        if modality=="RGB":
            modal = np.load(file_path,allow_pickle=True)["a"]
            if self.mode=="train":
                Annotation = np.load(file_path,allow_pickle=True)["c"]
                return modal,Annotation
            else:
                return modal            
        
        else:
            file_in = np.load(file_path,allow_pickle=True)
            mag = file_in['a']
            ang = file_in['b']
            
            if self.mode=="train":
                encoding = file_in['d']
                return mag,ang,encoding
            else:
                return mag,ang
    

    def get_frame_order(self,frames,modality):
        
        if modality=="OF":
            length = frames.shape[0]
        else:
            length = len(frames)
        interval_size = math.floor(length/self.fix_frames)
        j=0
        frame_indices=[]
        for i in range(self.fix_frames):
            frame_indices.append(j)
            j+=interval_size
        return frame_indices
    
    def read_val_flow(self,i,access_order,num_classes,multiply_factor):
        if self.mode=="test":
            Mag,Ang = self.load_file(access_order[i],modality="OF")
            prev_matrix = self.get_any_matrix(Mag,Ang,Encoding=[])
        else:
            Mag,Ang,Encoding = self.load_file(access_order[i],modality="OF")
            prev_matrix,prev_Annot = self.get_any_matrix(Mag,Ang,Encoding)
        final_matrix = np.reshape(
                prev_matrix,(
                    (1,
                    prev_matrix.shape[0],
                    prev_matrix.shape[1],
                    prev_matrix.shape[2])))

        # Obtaining data for the batch
        for j in range(i+1,i+(num_classes*multiply_factor)):
            
            if self.mode=="test":
                Mag,Ang = self.load_file(access_order[j],modality="OF")
                prev_matrix = self.get_any_matrix(Mag,Ang,Encoding=[])
            else:
                Mag,Ang,Encoding = self.load_file(access_order[j],modality="OF")
                init_matrix,init_Annot = self.get_any_matrix(Mag,Ang,Encoding)

            prev_matrix = np.reshape(
                init_matrix,(
                    (1,
                    init_matrix.shape[0],
                    init_matrix.shape[1],
                    init_matrix.shape[2])))
            
            final_matrix = np.concatenate([final_matrix,prev_matrix])
            if self.mode=="train":
                prev_Annot = np.concatenate([prev_Annot,init_Annot])
        
        if self.mode=="train":
            # Obtaining validation data for the batch
            m = i+(num_classes*multiply_factor)
            Mag,Ang,Encoding = self.load_file(access_order[m],modality="OF")
            prev_val_matrix,prev_val_Annot = self.get_any_matrix(Mag,Ang,Encoding)
            final_val_matrix = np.reshape(prev_val_matrix,((1,prev_val_matrix.shape[0],prev_val_matrix.shape[1],prev_val_matrix.shape[2])))

            # Obtaining training data for the batch
            for j in range(m+1,m+num_classes):
                
                Mag,Ang,Encoding = self.load_file(access_order[j],modality="OF")
                init_val_matrix,init_val_Annot = self.get_any_matrix(Mag,Ang,Encoding)

                prev_val_matrix = np.reshape(init_val_matrix,((1,init_val_matrix.shape[0],init_val_matrix.shape[1],init_val_matrix.shape[2])))
                final_val_matrix = np.concatenate([final_val_matrix,prev_val_matrix])
                prev_val_Annot = np.concatenate([prev_val_Annot,init_val_Annot])
            
            return final_matrix,prev_Annot,final_val_matrix,prev_val_Annot
        else:
            return final_matrix

    def read_any_rgb(self,access_order,start_index,end_index):    
        Y_Noun=[]
        Frame_Seq=[]
        
        for j in range(start_index,end_index):
            Frame=[]
            RGB,Noun = self.load_file(access_order[j],modality="RGB")
            frame_indices = self.get_frame_order(RGB,modality="RGB")
            
            for count in range(self.fix_frames):
                RGB_resized = cv2.resize(
                    src=RGB[frame_indices[count]],
                    dsize=self.input_shape)
                
                RGB_normalized = cv2.normalize(
                    RGB_resized, 
                    None, 
                    alpha=0, 
                    beta=1, 
                    norm_type=cv2.NORM_MINMAX, 
                    dtype=cv2.CV_32F)
                Frame.append(RGB_normalized)
                
                RGB_val = np.array(Frame)
                RGB_val = np.reshape(
                    RGB_val,(
                        1,RGB_val.shape[0],
                        RGB_val.shape[1],
                        RGB_val.shape[2],
                        RGB_val.shape[3]))
            
        
            Frame_Seq.append(RGB_val)
            Y_Noun.append(Noun[0])

        return Frame_Seq, Y_Noun
    
    def read_rgb(self,i,access_order):
        start_idx = i
        end_idx = i + 2*self.num_classes_total
        Train_Frame = []
        Train_Noun = []
        Train_Frame,Train_Noun = self.read_any_rgb(access_order,start_index=start_idx,end_index = end_idx)

        start_idx = end_idx
        start_idx = end_idx + self.num_classes_total
        Val_Frame = []
        Val_Noun = []
        Val_Frame,Val_Noun = self.read_any_rgb(access_order,start_index=i,end_index = i + self.num_classes_total)

        return Train_Frame,Train_Noun,Val_Frame,Val_Noun
    
    def read_frames(self,i,access_order,num_classes_total):    
        Y_Noun=[]
        Val_Frame=[]
        Val_Noun=[]
        
        Frames=[]
        for j in range(i,i+num_classes_total):
            Frame=[]
            if self.mode=="test":
                RGB = self.load_file(access_order[j],modality="RGB")
            else:
                RGB,Noun = self.load_file(access_order[j],modality="RGB")
            
            frame_indices = self.get_frame_order(RGB,modality="RGB")
            for count in range(self.fix_frames):
                RGB_resized = cv2.resize(
                    src=RGB[frame_indices[count]],
                    dsize=self.input_shape)
                
                RGB_normalized = cv2.normalize(
                    RGB_resized, 
                    None, 
                    alpha=0, 
                    beta=1, 
                    norm_type=cv2.NORM_MINMAX, 
                    dtype=cv2.CV_32F)
                
                if self.mode=="train":
                    #Y_Noun.append((int)(Noun[frame_indices[count]]))
                    if count==4:
                        Val_Frame.append(RGB_normalized)
                        Val_Noun.append((int)(Noun[frame_indices[count]]))    
                    Frame.append(RGB_normalized)
                else:
                    Frame.append(RGB_normalized)
                
                RGB_val = np.array(Frame)
                RGB_val = np.reshape(
                    RGB_val,(
                        1,RGB_val.shape[0],
                        RGB_val.shape[1],
                        RGB_val.shape[2],
                        RGB_val.shape[3]))
            Frames.append(RGB_val)
            Y_Noun.append(Noun[j])
        if self.mode=="test":
            return Frames
        else:
            return Frames, Y_Noun,Val_Frame,Val_Noun

    def getTotal(self):
        return self.train.shape[0]

    def getClasses(self):
         return np.unique((self.train)["Noun"])
    
    def getInputShape(self):
        return self.input_shape

    def getAnnotations(self):
        return self.train["Noun"],self.train["Verb"]