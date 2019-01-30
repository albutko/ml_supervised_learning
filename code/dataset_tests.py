from datasets import HiggsBosonDataset

def image_load():
    print 'tryin'

    ds = HiggsBosonDataset()
    ds.load()
    
    feats = ds.get_features()

    print feats.head()

def main():
    image_load()

if __name__ == '__main__':
    main()
