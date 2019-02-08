import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from Dataset import Dataset

class MappingDataset(Dataset):

    def __init__(self):
        self.classes = ['water','forest','impervious','farm','grass','orchard']
        self.datadir = os.path.abspath(os.path.dirname(__file__))
        self._load()


    def _load(self):

        train_data = pd.read_csv(os.path.join(self.datadir,'../../data/mapping/train.csv'))
        test_data = pd.read_csv(os.path.join(self.datadir,'../../data/mapping/test.csv'))

        le = LabelEncoder()
        train_data['class'] = le.fit(train_data['class'])
        test_data['class'] = le.fit(test_data['class'])

        self.classes = le.classes_

        self.train_data = {'features':train_data.iloc[:,1:], 'labels':train_data.iloc[:,1]}
        self.test_data = {'features':test_data.iloc[:,1:], 'labels':test_data.iloc[:,1]}
