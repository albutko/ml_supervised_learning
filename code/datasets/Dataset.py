class Dataset:
    def __init__(self, data=None):
        self.data = data

    def load(self):
        pass

    def get_data(self):
        return (self.get_features().copy(), self.get_labels().copy())

    def get_features(self):
        return self.data.get('features').copy()

    def get_labels(self):
        return self.data.get('labels').copy()
