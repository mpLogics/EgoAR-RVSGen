import scipy.io as sio
import cv2
from preprocessing import PreProcessing
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self):
        self.file_path = "video_clips/cropped_clips/OP01-R01-PastaSalad/OP01-R01-PastaSalad-66680-68130-F001597-F001639.mp4"
        self.metric_path = "data/performance_metrics/Metrics.npz"

    def makePlot(self,xVal,yVal = None, caption = "Data",l = 15, b = 10,sloc=None,xlab=None,ylab=None):
        plt.figure(figsize=(l,b))        
        plt.title(caption) 
        
        if yVal.any()==None:
            plt.plot(xVal)
        else:
            plt.plot(xVal,yVal)

        if xlab==None and ylab==None:
            pass
        else:
            plt.xlabel(xlab)
            plt.ylabel(ylab)    

        if sloc==None:
            plt.show()
        else:
            plt.savefig(sloc)

    def plot_metrics(self,m_path,Epoch):
        try:
            Metrics = np.load(m_path)
        except Exception:
            print("While printing metrics, file read unsuccessful")
            print("Getting metrics using default path")
            Metrics=np.load(self.metric_path)
        
        i = [i for i in range(1,Metrics['a'].shape[0]+1)]
        
        self.makePlot(xVal=i,yVal=Metrics['a'],caption="Training Loss vs Epochs",
        xlab="Epochs",ylab="Training Loss",sloc="data/performance_metrics/graphs/Train_Loss_" +(str)(Epoch)+".png")
        
        self.makePlot(xVal=i,yVal=Metrics['b'],caption="Training Accuracy vs Epochs",
        xlab="Epochs",ylab="Training Accuracy",sloc="data/performance_metrics/graphs/Train_Acc_" +(str)(Epoch)+".png")

        self.makePlot(xVal=i,yVal=Metrics['c'],caption="Validation Loss vs Epochs",
        xlab="Epochs",ylab="Validation Loss",sloc="data/performance_metrics/graphs/Val_Loss_" +(str)(Epoch)+".png")

        self.makePlot(xVal=i,yVal=Metrics['d'],caption="Validation Accuracy vs Epochs",
        xlab="Epochs",ylab="Validation Accuracy",sloc="data/performance_metrics/graphs/Val_Acc_" +(str)(Epoch)+".png")

    def ExtractFrames(self):
        
        p1 = PreProcessing()
        cap = cv2.VideoCapture(self.file_path)
        ret2, first_frame = cap.read()
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255
        f_no=1
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
                
            magnitude,angle = p1.obtainOpticalFlow(frame,prev_gray)
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            h,s,v = cv2.split(mask)
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            cv2.imwrite("Visuals/RGB_Frames/Frame_"+(str)(f_no)+".png",frame)
            cv2.imwrite("Visuals/Optical_Flow/Frame_"+(str)(f_no)+".png",rgb)
            f_no+=1
        
        cap.release()
    
