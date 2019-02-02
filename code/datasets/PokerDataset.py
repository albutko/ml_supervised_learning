from Dataset import Dataset
import pandas as pd

class PokerDataset(Dataset):

    def __init__(self):
        self._load()

    def _load(self):

        train_data = pd.read_csv('../data/poker/train.csv')
        test_data = pd.read_csv('../data/poker/test.csv')

        self.train_data = {'features':train_data.iloc[:,:-1], 'labels':train_data.iloc[:,-1]}
        self.test_data = {'features':test_data.iloc[:,:-1], 'labels':test_data.iloc[:,-1]}