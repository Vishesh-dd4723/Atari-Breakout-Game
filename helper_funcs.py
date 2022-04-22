import cv2 
import numpy as np
import pandas as pd

def merge_S_A(S, A):
    A_arr = np.ones((S.shape[0], S.shape[1]//2))
    A_bar = A_arr*np.expand_dims((A + 1)*64 - 1, axis=1)
    return np.hstack((S, A_bar)) / 255.


def model_input(S, A):
    S = S/255.    
    A = np.array([i for i in range(A)])*2 - 3
    X = np.array([np.hstack((S, A[i])) for i in range(4)])
    return X