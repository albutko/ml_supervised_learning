import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import neighbors
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


class Model:
    def __init__(self):
        self.clf = None
        self.results = {}
        self.metrics = []

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass


class DTree(Model):

    def __init__(self):
        self.clf = DecisionTreeClassifier()
        self.metrics = []

    def train(self):
        self.clf.fit(X_train, y_train)

    def predict(self, X):
        return self.clf.predict(X)

    def evaluate():
        
