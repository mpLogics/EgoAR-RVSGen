"""
import numpy as np
from RVSGen import GenVerbSpace

reduced_verb_space = GenVerbSpace()
totalSamples = reduced_verb_space.getTotalSamples(mode="train")
P_Noun_Verb = reduced_verb_space.calProbCombinations(totalSamples)
print("Sum of probabilities:",np.sum(np.array(list(P_Noun_Verb.values()))))
"""
import cv2
import numpy as np
def read_val_rgb(self,i,access_order):

    start_idx = i
    end_idx = i + 2*self.num_classes_total
    Train_Frame = []
    Train_Noun = []
    Train_Frame,Train_Noun = read_any_rgb(access_order,start_index=start_idx,end_index = end_idx)

    start_idx = end_idx
    start_idx = end_idx + self.num_classes_total
    Val_Frame = []
    Val_Noun = []
    Val_Frame,Val_Noun = read_any_rgb(access_order,start_index=i,end_index = i + self.num_classes_total)

    return Train_Frame,Train_Noun,Val_Frame,Val_Noun

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