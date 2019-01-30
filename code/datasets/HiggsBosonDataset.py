from Dataset import Dataset
import pandas as pd

class HiggsBosonDataset(Dataset):
    
    def load(self, n=None):
        path = '../data/higgs-boson/higgs-boson.csv'

        df = pd.read_csv(path)

        self.data = {
            'features': df.iloc[:,:-1].copy(),
            'label': df['Label'].copy()
        }
