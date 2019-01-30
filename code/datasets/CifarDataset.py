from Dataset import Dataset

class CifarDataset(Dataset):
    
    def __init__(self, train_data=None, test_data=None):
        super(CifarDataset, self).__init__(train_data, test_data)

    def load(n=None):
        path = '../data/cifar-10-batches-py/'

        train_files = [path + 'data_batch_' + str(f) for f in range(1,6)]
        test_file = path+'test_batch'
        print train_files
        train_data = dict()
        test_data = dict()

        def unpickle(file):
            import cPickle
            try:
                with open(file,'rb') as fo:
                    dict =  cPickle.load(fo)
                return dict
            except:
                print 'Error'

        for file in train_files:
            train_data.update(unpickle(file))

        test_data.update(unpickle(test_file))

        self.train_set = (train_data['data'], train_data['labels'])
        self.test_set = (train_data['data'], train_data['labels'])
