from CifarDataset import CifarDataset


def image_load():
    print 'tryin'

    ds = CifarDataset().load()


    print 'train_data instances: %d\n train_data lables: %d' % (ds.train_data.get('data').shape[0], ds.train_data.get('labels').shape[0])
    print 'train_data instances: %d\n train_data lables: %d' % (ds.test_data.get('data').shape[0], ds.test_data.get('labels').shape[0])

    return ds.train_data.get('data').shape[0] == ds.train_data.get('labels').shape[0] and ds.test_data.get('data').shape[0] == ds.test_data.get('labels').shape[0]

def main():
    image_load()

if __name__ == '__main__':
    main()
