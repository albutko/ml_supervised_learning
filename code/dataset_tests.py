from datasets import HiggsBosonDataset
from datasets import PokerDataset

def poker_get_training_data_test():
    d = PokerDataset()
    feats, labels = d.get_train_data()
    print 'Success'

def poker_get_test_data_test():
    d = PokerDataset()
    feats, labels = d.get_test_data()

    print 'Success'

def higgs_get_training_data_test():
    d = HiggsBosonDataset()
    feats, labels = d.get_train_data()
    print 'Success'

def higgs_get_test_data_test():
    d = HiggsBosonDataset()
    feats, labels = d.get_test_data()

    print 'Success'

def main():
    poker_get_training_data_test()
    poker_get_test_data_test()
    higgs_get_training_data_test()
    higgs_get_test_data_test()

if __name__ == '__main__':
    main()
