import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.RadiusGraph import RadiusGraph
from datasets.AuxFunction import FFT, add_noise
from tqdm import tqdm
import pickle
# --------------------------------获取数据-----------------------------
signal_size = 1024
root = "E:\data\XJTU_Spurgear"


# label
label = [i for i in range(5)]


class XJTUSpurgearRadius(object):
    num_classes = 5

    def __init__(self, data_dir,InputType,task):
        self.data_dir = data_dir
        self.InputType = InputType
        self.task = task



    def data_preprare(self, test=False):

        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            pass

        if test:
            test_dataset = list_data
            return test_dataset
        else:

            train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
            # train_dataset, val_dataset =list_data[:80],list_data[80:]
            print(len(train_dataset),len(val_dataset))
            return train_dataset, val_dataset

