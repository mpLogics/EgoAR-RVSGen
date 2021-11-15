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
        self.batch_size = 25
        self.train = pd.read_csv(self.train_split)
        self.test = pd.read_csv(self.test_split)
        self.input_shape = (299,299)
        self.sample_rate = 0.1
        self.fix_frames = 10
        self.num_classes_total = 51
        
    def get_matrix(self,Mag,Angle,Encoding):
        interval_size = math.floor(len(Mag)/self.fix_frames)
        Annotations=[]
        j = 0
        
        Mag[0] = Mag[0][120:360,160:480]
        Angle[0] = Angle[0][120:360,160:480]
        prev_matrix = np.concatenate([Mag[0],Angle[0]],axis=1)
        init_matrix = np.reshape(prev_matrix,(1,prev_matrix.shape[0],prev_matrix.shape[1]))
        

        Mag[1] = Mag[1][120:360,160:480]
        Angle[1] = Angle[1][120:360,160:480]
        prev_val = np.concatenate([Mag[1],Angle[1]],axis=1)
        prev_val = np.reshape(prev_val,(1,prev_val.shape[0],prev_val.shape[1]))
        j+=interval_size
        
        for k in range(2,self.fix_frames):
            print("K = ",k)
            #print("Here Sanity Check",Mag[j].shape)
            #print("Here Sanity Check",Angle[j].shape)
            Mag[j] = Mag[j][120:360,160:480]
            Angle[j] = Angle[j][120:360,160:480]
            
            print("Here -1",Mag[j].shape)
            print("Here 0",Angle[j].shape)
            
            #if k==self.fix_frames+1:
            if k % 2 !=0:
                init_val = np.concatenate([Mag[j],Angle[j]],axis=1)
                temp_init_val = np.reshape(init_val,(1,init_val.shape[0],init_val.shape[1]))
                prev_val = np.concatenate([prev_val,temp_init_val])

            else:
                prev_matrix = np.concatenate([Mag[j],Angle[j]],axis=1)
                #print("Here 2",Mag[j].shape)
                #print("Here 3",Angle[j].shape)
                #print("Here 4",prev_matrix.shape)
                temp = np.reshape(prev_matrix,(1,prev_matrix.shape[0],prev_matrix.shape[1]))
                init_matrix = np.concatenate([init_matrix,temp])
            
            j+=interval_size    
        
        Annotations.append((int)(Encoding[0]))
        return init_matrix,np.array(Annotations),prev_val,np.array([(int)(Encoding[0])])

            #prev_matrix = np.concatenate([Mag[j],Angle[j]],axis=1)
            #temp = np.reshape(prev_matrix,(1,prev_matrix.shape[0],prev_matrix.shape[1]))
            #init_matrix = np.concatenate([temp,init_matrix])
            #j+=interval_size    
        #Annotations.append((int(Encoding[0])))
        #return init_matrix,np.array(Annotations),init_val,val_annot
    

    def load_file(self,i,modality):
        file_path = "data/preprocessed_data/" + modality + "/" + self.train["FileName"][i] + ".npz"
        
        if modality=="RGB":
            modal = np.load(file_path,allow_pickle=True)["a"]
            Annotation = np.load(file_path,allow_pickle=True)["c"]
        else:
            file_in = np.load(file_path,allow_pickle=True)
            mag = file_in['a']
            #ang = np.array([element for (i,element) in enumerate(file_in['b'])]) 
            ang = file_in['b']
            encoding = file_in['d']
            return mag,ang,encoding
            #modal = np.zeros((s1,s2,s3*2))
            #modal[:,:,:s3] = mag
            #modal[:,:,s3:] = ang
        return modal,Annotation
    

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
    
    def read_flow(self,i,access_order,num_classes_total):
        Frames=[]
        Y=[]
        Val_Frame=[]
        Val_Noun=[]
        
        Mag,Ang,Encoding = self.load_file(access_order[i],modality="OF")
        print(i)
        print(access_order[i])
        print(Encoding[0])
        
        prev_matrix,prev_Annot,prev_val,prev_val_annot = self.get_matrix(Mag,Ang,Encoding)
        final_matrix = np.reshape(prev_matrix,((1,prev_matrix.shape[0],prev_matrix.shape[1],prev_matrix.shape[2])))
        final_val = np.reshape(prev_val,((1,prev_val.shape[0],prev_val.shape[1],prev_val.shape[2])))

        for j in range(i+1,i+num_classes_total):
            #Modal,Annotation = self.load_file(access_order[j],modality="OF")
            #frame_indices = self.get_frame_order(Modal,modality="OF")
        
            Mag,Ang,Encoding = self.load_file(access_order[j],modality="OF")
            init_matrix,init_Annot,prev_val,init_val_annot = self.get_matrix(Mag,Ang,Encoding)
            init_val = np.reshape(prev_val,((1,prev_val.shape[0],prev_val.shape[1],prev_val.shape[2])))
            final_val = np.concatenate([final_val,init_val])
            prev_val_annot = np.concatenate([prev_val_annot,init_val_annot])
            
            #init_matrix,init_Annot,val_matrix,val_annot = get_matrix(Mag,Ang,Encoding)
            prev_matrix = np.reshape(init_matrix,((1,init_matrix.shape[0],init_matrix.shape[1],init_matrix.shape[2])))
            final_matrix = np.concatenate([final_matrix,prev_matrix])
            prev_Annot = np.concatenate([prev_Annot,init_Annot])

            #for count in range(self.fix_frames):
            #    if count==4:
            #        Val_Frame.append(Modal[frame_indices[count]])
            #        Val_Noun.append((int)(Annotation[frame_indices[count]]))
            #    
            #    else:
            #        Frames.append(Modal[frame_indices[count]])
            #        Y.append((int)(Annotation[frame_indices[count]]))
        
        Frames = final_matrix
        Y = prev_Annot
        Val_Frame = final_val
        Val_Annotation = np.array(prev_val_annot)
        
        return Frames, Y, Val_Frame, Val_Annotation

    def read_frames(self,i,access_order,num_classes_total):    
        Frame=[]
        Y_Noun=[]
        Val_Frame=[]
        Val_Noun=[]
        
        for j in range(i,i+num_classes_total):
            RGB,Noun = self.load_file(access_order[j],modality="RGB")
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