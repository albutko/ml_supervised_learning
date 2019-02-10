from datasets import HiggsBosonDataset
from datasets import MappingDataset

def mapping_get_training_data_test():
    d = MappingDataset()
    feats, labels = d.get_train_data()
    print 'Success'

def mapping_get_test_data_test():
    d = MappingDataset()
    feats, labels = d.get_test_data()

    print 'Success'

def higgs_get_training_data_test():
    d = HiggsBosonDataset()
    feats, labels = d.get_train_data()
    print(feats[1,:])
    print 'Success'

def higgs_get_test_data_test():
    d = HiggsBosonDataset()
    feats, labels = d.get_test_data()

    print 'Success'

def main():
    # mapping_get_training_data_test()
    # mapping_get_test_data_test()
    higgs_get_training_data_test()
    higgs_get_test_data_test()

if __name__ == '__main__':
    main()
