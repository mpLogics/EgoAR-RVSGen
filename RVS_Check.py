"""
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Input,ConvLSTM2D

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
"""
#from OpticalFlow import learn_optical_flow
from RVSGen import GenVerbSpace

#l_of = learn_optical_flow()
#verb_predictor,epochs,Loss_per_epoch,Accuracy_per_epoch,Val_Loss_per_epoch,Val_Acc_per_epoch = l_of.check_prev_trainings(
#    model_name="Verb_Predictor",
#    modality="OF")
#print("Training Loss:",Loss_per_epoch)
#print("Training Accuracy:",Accuracy_per_epoch)
#print("Validation Loss:",Val_Loss_per_epoch)
#print("Validation Accuracy:",Val_Acc_per_epoch)

#x = verb_predictor.output
reduced_verb_space = GenVerbSpace()
Nouns = reduced_verb_space.getNounSet()
Verbs = reduced_verb_space.getNounSet()

print(Nouns)
print(Verbs)
totalSamples = reduced_verb_space.getTotalSamples(mode="train")
P_Noun = reduced_verb_space.calProbNouns(totalSamples=totalSamples)
P_Verb = reduced_verb_space.calProbVerbs(totalSamples=totalSamples)

print(P_Noun)
print(P_Verb)

Verb_Probable = reduced_verb_space.RVSGen(Noun_Pred=Nouns[0],K_Value=10)
print(Verb_Probable)


