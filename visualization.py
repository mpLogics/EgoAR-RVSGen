import scipy.io as sio
import cv2
from preprocessing import PreProcessing
import numpy as np
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self):
        self.file_path = "video_clips/cropped_clips/OP01-R01-PastaSalad/OP01-R01-PastaSalad-66680-68130-F001597-F001639.mp4"

    def makePlot(xVal,yVal = None, caption = "Data",l = 15, b = 10,sloc=None,xlab=None,ylab=None):
        plt.figure(figsize=(l,b))        
        plt.title(caption) 
        
        if yVal==None:
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
    
