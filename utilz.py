import os
import pickle as pkl
import pandas as pd

def save_features(data, file_name):
    pkl.dump(data, open(file_name, 'wb'))

def load_features(file_name):
    return pkl.load(open(file_name, 'rb'))

def set_trainable(model, set_para):
    for l in model.layers:
        l.trainable = set_para
    model.trainable = set_para
