import os
import numpy as np

fs = 1000 # Frequency Sampling

def read_signal(path):
    if os.path.isfile(path):
        data = np.loadtxt(path,dtype=np.float64)
        return data
    else:
        print("Error: File Not Exist")

def train():
    pass

def test ():
    pass

def load_train():
    full_path = "data/train"
    train_data = {}
    for file_name in os.listdir(full_path):
        label = "person" + os.path.splitext(file_name)[0][1]
        data = read_signal(os.path.join(full_path,file_name))
        train_data[label] = data
    return train_data

def load_test():
    full_path = "data/test"
    files = os.listdir(full_path)
    test_data = np.empty(len(files), dtype=object)
    for i, file_name in enumerate(files):
        data = read_signal(os.path.join(full_path,file_name))
        test_data[i] = data
    return test_data

