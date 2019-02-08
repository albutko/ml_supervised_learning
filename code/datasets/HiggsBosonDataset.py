import os
from sklearn.model_selection import train_test_split
import pandas as pd

from Dataset import Dataset

class HiggsBosonDataset(Dataset):
    def __init__(self, small=True):
        self.classes =['background','signal']
        self.datadir = os.path.abspath(os.path.dirname(__file__))
        self._load(small)

    def _load(self, small=True):

        df = pd.read_csv(os.path.join(self.datadir,'../../data/higgs/higgs-boson.csv'))


        cols_to_keep = ['DER_mass_MMC', 'DER_mass_transverse_met_lep','DER_mass_vis',
                        'DER_pt_h', 'DER_deltar_tau_lep','DER_pt_tot', 'DER_sum_pt',
                        'DER_pt_ratio_lep_tau','DER_met_phi_centrality', 'PRI_tau_pt',
                        'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta',
                        'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet','Label']

        if small:
            fr = .1
        else:
            fr = 1
        df['Label'] = df['Label']=='s'
        df = df[cols_to_keep].sample(frac=fr, random_state = 100)

        train, test = train_test_split(df, test_size=.25, random_state=100)
        self.train_data = {
            'features': train.iloc[:,:-1],
            'labels': train.iloc[:,-1]
        }

        self.test_data = {
            'features': test.iloc[:,:-1],
            'labels': test.iloc[:,-1]
        }
