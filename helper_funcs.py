import cv2 
import numpy as np
import pandas as pd

def image_compressor(img, new_size=(75, 80)):

    img_reduced = img[30:195, 5:155,:]
    img_grey = cv2.cvtColor(img_reduced, cv2.COLOR_BGR2GRAY)
    img_compressed = cv2.resize(img_grey, new_size)

    return img_compressed

def list_to_pdSeries(keys, values):
    dic = {}
    for i in range(len(keys)):
        dic[keys[i]] = values[i]
    return pd.Series(dic)


def merge_S_A(S, A):
    A_arr = np.ones((S.shape[0], S.shape[1]//2))
    A_bar = A_arr*np.expand_dims((A + 1)*64 - 1, axis=1)
    return np.hstack((S, A_bar)) / 255.


def model_input(prev, curr, k):
    S = np.reshape(np.array([[prev, curr]] * k), newshape=(k, np.product(prev.shape)*2))
    A = np.array([i for i in range(k)])
    X =  merge_S_A(S, A)
    return np.reshape(X, newshape=((k, 80, 75, 3)))