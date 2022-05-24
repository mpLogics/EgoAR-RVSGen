# EgoAR-RVSGen

If you use the code or a part of it in your research, please cite the following paper.

Manav Prabhakar and Snehasis Mukherjee, First-person Activity Recognition by Modelling Subject-Action Relevance, In IJCNN 2022, IEEE.

## Introduction
The efficacy of Action Recognition methods depends upon relevance of the action (Verb) with respect to the subject (Noun). Existing methods overlook the suitability of Noun-Verb combination in defining an action. In this work, we propose an algorithm called Reduced Verb Set Generator (RVSGen) to reduce the number of possible verbs related to the actions, based upon the relevance of noun-verb combination.

The major contributions of the proposed method can be summarized as follows:
- We first predict the noun and then the verb, instead of predicting both of them simultaneously, as done in the traditional approaches.
- We propose a novel algorithm called RVSGen (based on the Beam Search Decoder) to reduce the number of verbs, and enhance the computational efficacy of the model.
- We apply a ConvLSTM model to predict the verbs, instead of using a simple CNN.

## Code Usage
- Clone the repository and prepare the EGTEA dataset. (To be completed)


## Technology Stack 

1. Python 
2. Tensorflow
3. Pandas
4. Scikit-Learn



