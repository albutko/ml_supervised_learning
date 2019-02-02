class Dataset:
    def __init__(self):
        self.train_data = {}
        self.test_data = {}

    def _load(self):
        pass

    def get_train_data(self):
        return (self.train_data.get('features').copy(), self.train_data.get('labels').copy())

    def get_test_data(self):
        return (self.test_data.get('features').copy(), self.test_data.get('labels').copy())
