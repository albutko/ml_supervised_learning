from Dataset import Dataset
import pandas as pd

class HiggsBosonDataset(Dataset):
    def __init__(self):
        self._load()


    def _load(self):
        path = '../data/higgs/higgs-boson.csv'

        df = pd.read_csv(path)


        cols_to_keep = ['DER_mass_MMC', 'DER_mass_transverse_met_lep','DER_mass_vis',
                        'DER_pt_h', 'DER_deltar_tau_lep','DER_pt_tot', 'DER_sum_pt',
                        'DER_pt_ratio_lep_tau','DER_met_phi_centrality', 'PRI_tau_pt',
                        'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta',
                        'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet']

        df = df[cols_to_keep].sample(frac=1, random_state = 100)

        self.train_data = {
            'features': df.iloc[:200000,:-1],
            'labels': df.iloc[:200000,-1]
        }

        self.test_data = {
            'features': df.iloc[200000:,:-1],
            'labels': df.iloc[200000:,-1]
        }
