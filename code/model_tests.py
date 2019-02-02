from datasets import HiggsBosonDataset
from sklearn.model_selection import train_test_split
from sklearn import tree

def test():
    print 'tryin'

    X, y = HiggsBosonDataset().get_data()

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, shuffle=True)

    clf = tree.DecisionTreeClassifier(min_samples_leaf=100, max_depth=10)

    clf.fit(X_train, y_train)

    print clf.score(X_test,y_test)

def main():
    test()

if __name__ == '__main__':
    main()
