from datasets import HiggsBosonDataset, PokerDataset
from sklearn.model_selection import learning_curve, train_test_split, ParameterGrid, GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def plot_learning_curve_train_size(estimator, X, y, title, ylim=None, cv=3, scoring='accuracy',
                                   train_sizes=np.linspace(.1,1.0,5)):
    """Based on scikit learn documentation: https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html"""

    train_size, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,
                                                           train_sizes=train_sizes)

    plt.figure()
    plt.title(title)
    plt.xlabel('size of training set')
    plt.ylabel(scoring)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.plot(train_size, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_size, test_scores_mean, 'o-', color='g', label='Testing Score')

    plt.legend(loc='best')
    plt.show()




def plot_curve(data):
    """
        Takes JSON style data object and plots on line

        data = {
            'title':
            'xlabel':
            'ylabel':
            'ylim':
            'xlim':
            'x_values':
            'series':[
                        {
                    'metrics':
                    'style':
                    'color':
                    'lable':
                }
            ]
        }

    """
    plt.figure()
    plt.title(data.get('title'))
    plt.xlabel(data.get('xlabel'))
    plt.ylabel(data.get('ylabel'))

    if data.get('ylim') is not None:
        plt.ylim(data.get('ylim'))

    if data.get('xlim') is not None:
        plt.ylim(data.get('xlim'))

    for series in data.get('series'):
        plt.plot(data.get('x_values'), series.get('metrics'),
                 series.get('style','o-'), color=series.get('color','r'), label=series.get('label'))

    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix(y_pred, y_true, classes=None, title='Confusion Matrix', normalize=True):
    classes_int = np.arange(len(classes))

    cm = confusion_matrix(y_pred, y_true, classes_int)
    print cm
    if normalize:
        print cm.sum(axis=1)
        cm = cm.astype('float64')/cm.sum(axis=0)[:,np.newaxis]

        fmt = '.2f'
    else:
        fmt = 'd'
    print cm
    df_cm = pd.DataFrame(cm,index=[c for c in classes],columns=[c for c in classes])

    plt.figure()
    plt.title(title)


    sns.heatmap(df_cm, annot=True, fmt=fmt).set(xlabel='Predicted', ylabel='True')
    plt.show()

def best_hyperparameter_search(estimator, X, y, params, scoring='accuracy', cv=1, graph=False):
    # param_grid = ParameterGrid(params)

    clf = GridSearchCV(estimator, params, cv=cv, scoring=scoring, return_train_score=True)
    clf.fit(X, y)

    results = clf.cv_results_

    print '{} gave the best score with {} mean test {} and mean fit time of {} +- {} secs'.format(clf.best_params_,
                                                                                            clf.best_score_,
                                                                                            scoring,
                                                                                            results['mean_fit_time'][clf.best_index_],
                                                                                            results['std_fit_time'][clf.best_index_])
    if graph:
        p = params.keys()[0]
        data = {
            'title':'{} vs {}\n(CV:{})'.format(scoring,p,cv) ,
            'xlabel':p,
            'ylabel': scoring,
            'x_values': params.get(p),
            'series':[
                        {
                        'metrics':results.get('mean_test_score'),
                        'style': '-',
                        'color': 'g',
                        'label': 'Test Set'
                    },
                        {
                        'metrics':results.get('mean_train_score'),
                        'style': '-',
                        'color': 'r',
                        'label': 'Train Set'
                    }
            ]
        }
        plot_curve(data)


    return clf.best_estimator_
def confusion_matrix_test():
    clf = DecisionTreeClassifier(min_samples_leaf=10)
    ds = HiggsBosonDataset()

    classes = ['back', 'signal']
    X, y = ds.get_train_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    plot_confusion_matrix(y_pred, y_test, classes)


def learning_curve_training_size_test():
    clf = DecisionTreeClassifier(min_samples_leaf=10)
    ds = HiggsBosonDataset()
    plot_learning_curve_train_size(clf, X, y, 'Learning Curve', scoring='f1',  train_sizes=np.linspace(.1,1,10))




def main():
    clf = DecisionTreeClassifier(min_samples_split=190, max_leaf_nodes=25)
    ds = HiggsBosonDataset()

    classes = ['back', 'signal']
    X, y = ds.get_train_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    # params = {'max_leaf_nodes':np.linspace(2,1000,5).astype('int32'),
    #           'max_depth':np.linspace(2, 1000, 10).astype('int32')}

    # params = {'min_samples_split':np.linspace(2,200,20).astype('int32'),
    #           'max_leaf_nodes':np.linspace(2,1000,5).astype('int32')'}

    params = {'criterion':['gini','entropy']}


    best_clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='f1', cv=3, graph=False)
    plot_confusion_matrix(best_clf.predict(X_test), y_test,classes)
if __name__ == '__main__':
    main()
