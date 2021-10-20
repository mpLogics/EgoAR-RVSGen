import os
import pandas as pd
import re
import numpy as np
import cv2
import scipy.io as sio
from MessageLogging import StoreLogs

class SetAnnotations():

    def __init__(self):
        self.root="data/"
        self.base = self.root+"action_annotation/"
        self.totCombs=3
   
    def getXY(self,Data):
        X=[]
        Y1=[]
        Y2=[]
        Y3=[]
        Batch=[]
        for i in range(Data.shape[0]):
            splits = re.split(" ",Data[0][i]) 
            temp = splits[0]
            label1 = int(splits[1])
            label2 = int(splits[2])
            label3 = int(splits[3])
            X.append(temp)
            Batch.append(0)
            Y1.append(label1)
            Y2.append(label2)
            Y3.append(label3)
        return X,Y1,Y2,Y3,Batch
    
    def txtToCSV(self):
        
        AllFiles=[]
        
        try:
            os.scandir(self.root + "Splits/")
        except FileNotFoundError:
            print("Directory not found, creating directory")
            os.mkdir(os.path.join(os.getcwd(),self.root + "Splits"))
            
        
        for i in range(self.totCombs):
            
            Train_Data = pd.read_csv(self.base + "/train_split" + (str)(i+1) + ".txt",header = None,delimiter='\t')
            X_dash,Y_dash1,Y_dash2,Y_dash3,Batch = self.getXY(Train_Data)
                
            Split = pd.DataFrame(list(zip(X_dash,Y_dash1,Y_dash2,Y_dash3,Batch)))
            Split.columns=["FileName","Action","Verb","Noun","Batch_No"]    
            Split.to_csv(self.root + "Splits/train_split"+(str)(i+1)+".csv",index=False)
            
            AllFiles.append(X_dash)
            
            Test_Data = pd.read_csv(self.base + "/test_split" + (str)(i+1) + ".txt",header = None,delimiter='\t')
            X_dash,Y_dash1,Y_dash2,Y_dash3,Batch = self.getXY(Test_Data)
            
            Split = pd.DataFrame(list(zip(X_dash,Y_dash1,Y_dash2,Y_dash3,Batch)))
            Split.columns=["FileName","Action","Verb","Noun","Batch_No"]
            Split.to_csv(self.root+"Splits/test_split"+(str)(i+1)+".csv",index=False) 
    
    def run(self):
        self.txtToCSV()
        print("CSV Annotations Derivation Complete")

class PreProcessing():
    def __init__(self):
        self.root="data/"
        self.Split_path = self.root + "Splits/"
        self.totCombs = 3
        self.video_root_path = self.root + "video_clips"
        self.batch_size=15
        self.tiers=3
        self.preprocess_save_path = self.root + "preprocessed_data/"
        self.ext=".npz"
    
    def FindLabels(self,videoName,train,test):
        File_found_flag = False
        yVals = np.zeros((1,3))
        for i in range(self.totCombs):  
            if np.count_nonzero(train[i]["FileName"]==videoName) == 0:
                if np.count_nonzero(test[i]["FileName"]==videoName) == 0:
                    break
                else:
                    df = test[i].loc[test[i]["FileName"].str.contains(videoName)]
                    df = df.reset_index()
                    yVals[0][0] = df['Action'][0]
                    yVals[0][1] = df['Verb'][0]
                    yVals[0][2] = df['Noun'][0]
                    File_found_flag = True
            else:
                df = train[i].loc[train[i]["FileName"].str.contains(videoName)]
                df = df.reset_index()
                yVals[0][0] = df['Action'][0]
                yVals[0][1] = df['Verb'][0]
                yVals[0][2] = df['Noun'][0]
                File_found_flag = True
        return yVals,File_found_flag
    
    """
    def alotBatch(self,videoName,batch_no,train,test):
        File_found_flag = False
        yVals = np.zeros((1,3))
        for i in range(self.totCombs):  
            if np.count_nonzero(train[i]["FileName"]==videoName) == 0:
                if np.count_nonzero(test[i]["FileName"]==videoName) == 0:
                    break
                else:
                    test[i].loc[test[i]["FileName"].str.contains(videoName),['Batch_No']] = batch_no
                    df = test[i].loc[test[i]["FileName"].str.contains(videoName)]
                    df = df.reset_index()
                    yVals[0][0] = df['Action'][0]
                    yVals[0][1] = df['Verb'][0]
                    yVals[0][2] = df['Noun'][0]
                    File_found_flag = True
            else:
                train[i].loc[train[i]["FileName"].str.contains(videoName),['Batch_No']] = batch_no
                df = train[i].loc[train[i]["FileName"].str.contains(videoName)]
                df = df.reset_index()
                yVals[0][0] = df['Action'][0]
                yVals[0][1] = df['Verb'][0]
                yVals[0][2] = df['Noun'][0]
                File_found_flag = True
                    #df_train.to_csv("Splits/train_split"+(str)(j+1)+".csv",index=False)
        return train,test,yVals,File_found_flag 
    """
    

    def obtainOpticalFlow(self,frame,prev_gray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,None,0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #mask[..., 0] = angle * 180 / np.pi / 2
        #mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        #h,s,v = cv2.split(mask)
        #rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        #return rgb
        return magnitude,angle
        

    def storeData(self,file,file_path,Y):
        RGB=[]
        Mag=[]
        Ang=[]
        Action=[]
        Verb=[]
        Noun=[]
        Name=[]
        
        cap = cv2.VideoCapture(file_path)
        ret2, first_frame = cap.read()
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        #mask = np.zeros_like(first_frame)
        #mask[..., 1] = 255

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
                
            mag,angle = self.obtainOpticalFlow(frame,prev_gray)            
            RGB.append(frame)
            Mag.append(mag)
            Ang.append(angle)
            
            Action.append(Y[0][0])
            Verb.append(Y[0][1])
            Noun.append(Y[0][2])
            Name.append(file)
        
        cap.release()
        cap = cv2.VideoCapture(file_path)
        frames_removed = (int)(0.1*len(RGB))
        df = pd.DataFrame()
        df["FileName"]=Name
        df["RGB"]=RGB
        df["Magnitude"]=Mag
        df["Angle"]=Ang
        df["Action"]=np.array(Action)
        df["Verb"]=np.array(Verb)
        df["Noun"]=np.array(Noun)
        return df,frames_removed
    
    def searchFileExists(self,videoName):
        value = videoName + self.ext        
        if value in os.listdir(self.preprocess_save_path):
            return True
        else:
            return False

    def save_as_file(self,df,videoName,frames_removed):

        RGB = np.array(df["RGB"])[0][frames_removed:-frames_removed]
        Action = np.array(df["Action"])[0][frames_removed:-frames_removed]
        Magnitude = np.array(df["Magnitude"])[0][frames_removed:-frames_removed]
        Noun = np.array(df["Noun"])[0][frames_removed:-frames_removed]
        Verb = np.array(df["Verb"])[0][frames_removed:-frames_removed]
        Angle = np.array(df["Angle"])[0][frames_removed:-frames_removed]

        if self.ext==".mat":
            mdic_RGB = {"RGB": RGB,
                    "Action": Action,
                    "Noun": Noun}

            mdic_OF = {"Magnitude": Magnitude,
                    "Angle": Angle,
                    "Action": Action,
                    "Verb": Verb}  
            
            sio.savemat(self.root+"preprocessed_data/RGB/"+videoName+self.ext, mdic_RGB)
            sio.savemat(self.root+"preprocessed_data/OF/"+videoName+self.ext, mdic_OF)
        
        else:
            np.savez(self.root+"preprocessed_data/RGB/"+videoName+self.ext,
            a = RGB,
            b = Action,
            c = Noun)

            np.savez(self.root+"preprocessed_data/OF/"+videoName+self.ext,
            a = Magnitude,
            b = Angle,
            c = Action,
            d = Verb)
        

    def preProcess(self,train,test):
        Old_Files_Read_Complete=False
        try:
            os.scandir(self.root+"preprocessed_data/")
        except FileNotFoundError:
            os.mkdir(self.root+"preprocessed_data")
        
        df = pd.DataFrame()
        
        for direc in os.listdir(self.video_root_path):
            subL1_path = self.video_root_path + "/" + direc
            for subdirec in os.listdir(subL1_path):
                subL2_path = subL1_path + "/" + subdirec
                for file in os.listdir(subL2_path):
                    file_path = subL2_path +"/" + file
                    videoName = re.split(".mp4",file)[0]
                    
                    #train,test,
                    Y,File_Found = self.FindLabels(videoName,train,test)
                    
                    if File_Found == True: 
                        if Old_Files_Read_Complete:
                            df,frames_removed = self.storeData(videoName,file_path,Y)
                            save_as_file(df,videoName,frames_removed)
                        else:
                            if self.searchFileExists(videoName)==False:
                                print("Old Files read successful, will now begin saving new files.")
                                print("Last file read: "+videoName+self.ext)
                                Old_Files_Read_Complete=True
                                df,frames_removed = self.storeData(videoName,file_path,Y)
                                save_as_file(df,videoName,frames_removed)
                    else:
                        pass    
                            
        L1 = StoreLogs()
        L1.fileName="Logs.txt"
        L1.mode="a"
        L1.msg="Success!"
        L1.des="Preprocessing Complete"
        L1.LogMessages()

    """
    def preProcessBatchwise(self,train,test):
        #Considering a 3-tier structure of the dataset
        #Paths=[]
        try:
            os.scandir("batches/")
        except FileNotFoundError:
            os.mkdir("batches")
        
        df = pd.DataFrame()
        ctr = 0
        batch_no = 1
        
        for direc in os.listdir(self.video_root_path):
            subL1_path = self.video_root_path + "/" + direc
            for subdirec in os.listdir(subL1_path):
                subL2_path = subL1_path + "/" + subdirec
                for file in os.listdir(subL2_path):
                    file_path = subL2_path +"/" + file
                    videoName = re.split(".mp4",file)[0]
                    #Paths.append(file_path)
                    #Paths.append(videoName)
                    train,test,Y,File_Found = self.alotBatch(videoName,batch_no,train,test)
                    
                    if File_Found == True:
                        #self.storeData(batch_no,file_path,Y,df)
                        df = pd.concat([df, self.storeData(videoName,file_path,Y)], ignore_index=True)
                        ctr+=1

                    else: 
                        pass
                    
                    if ctr % self.batch_size == 0 and ctr >0 :                        
                        
                        if df.empty:
                            pass
                        
                        else:
                            mdic = {"FileName": np.array(df["FileName"]),
                                    "RGB": np.array(df["RGB"]),
                                    "Magnitude": np.array(df["Magnitude"]),
                                    "Angle": np.array(df["Angle"]),
                                    "Action": np.array(df["Action"]),
                                    "Verb": np.array(df["Verb"]),
                                    "Noun": np.array(df["Noun"])}  
                            
                            sio.savemat("batches/Batch_"+(str)(batch_no)+".mat", mdic)
                            #mat_fname = "batches/matlab_matrix.mat"
                            #df.to_csv("batches/Batch_"+(str)(batch_no)+".csv")
                            batch_no+=1
                            df=pd.DataFrame()
        
        for i in range(self.tiers):
            train[i].to_csv(self.Split_path + "train_split"+(str)(i+1)+".csv",index=False)
            test[i].to_csv(self.Split_path + "test_split"+(str)(i+1)+".csv",index=False)
    """
    
        
        #return Paths