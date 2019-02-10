from itertools import cycle
import itertools
from sklearn.model_selection import learning_curve, train_test_split, ParameterGrid, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,f1_score, recall_score, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_roc_curve(y_score, y_test, X_test=None, estimator=None, classes=None, file=None, scoring='f1'):
    n_classes = len(classes)
    if n_classes > 2:
        #binarize output
        y_test = label_binarize(y_test,classes=range(n_classes))
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC multi-class\n(Macro AUC: {0:0.2f})'.format(roc_auc_score(y_test,y_score,average='macro')))
        plt.legend(loc="lower right")

        if file is not None:
            plt.savefig('../images/'+file, bbox_inches='tight')


        plt.show()

    else:
        probas = estimator.predict_proba(X_test)
        y_pred = estimator.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, probas[:,1])
        plt.figure()


        plt.title("ROC Curve\n(AUC:{0:0.2f})".format(roc_auc_score(y_test, y_score[:,1])))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--',color='r')

        if file is not None:
            plt.savefig('../images/'+file, bbox_inches='tight')

        plt.show()

# def plot_mlp_iterative_learning_curve(estimator, )

def plot_learning_curve_train_size_cv(estimator, X, y, title, ylim=None, cv=3, scoring='accuracy',
                                   train_sizes=np.linspace(.1,1.0,5), file=None):
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

    if file is not None:
        plt.savefig('../images/'+file, bbox_inches='tight')

    plt.show()

def plot_time_by_parameter(estimator, X_train, y_train, X_test, params, title='', ylim=None, file=None):
    training_time = []
    predict_time = []
    param_vals = []
    param = ''
    for k, v in params.items():
        param_vals = v
        param = k
        for val in v:
            estimator.set_params()
            start = timer()
            estimator.fit(X_train, y_train)
            training_time.append(timer() - start)
            y_t_pred = estimator.predict(X_train)
            start = timer()
            y_test_pred = estimator.predict(X_test)
            predict_time.append(timer() - start)

    plt.figure()
    plt.title('Training and Prediction Time vs Training Size')
    plt.xlabel(k)
    plt.ylabel('Time (secs)')

    if ylim is not None:
        plt.ylim(*ylim)

    plt.plot(param_vals, training_time, '--', color='r', label='Training time')
    plt.plot(param_vals, predict_time, 'o--', color='g', label='Prediction time')

    plt.legend(loc='best')

    plt.show()

def plot_learning_curve_train_size(estimator, X_train, y_train, X_test, y_test, title='Learning Curve', ylim=None, scoring='accuracy',
                                   train_sizes=np.linspace(.1,.99,10), file=None):
    """Based on scikit learn documentation: https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html"""

    train_scores, test_scores = [], []
    train_size = train_sizes*X_train.shape[0]
    train_size.astype('int32')
    training_time = []
    predict_time = []
    for size in train_sizes:
        _,X_t,_,y_t = train_test_split(X_train, y_train, test_size=size)
        start = timer()
        estimator.fit(X_t, y_t)
        training_time.append(timer() - start)
        y_t_pred = estimator.predict(X_t)
        start = timer()
        y_test_pred = estimator.predict(X_test)
        predict_time.append(timer() - start)
        if scoring == 'f1_macro':
            train_scores.append(f1_score(y_t, y_t_pred,average='macro'))
            test_scores.append(f1_score(y_test, y_test_pred,average='macro'))
        elif scoring == 'recall':
            train_scores.append(recall_score(y_t, y_t_pred))
            test_scores.append(recall_score(y_test, y_test_pred))


    plt.figure()
    plt.title('Training and Prediction Time vs Training Size')
    plt.xlabel('size of training set')
    plt.ylabel('Time (secs)')

    if ylim is not None:
        plt.ylim(*ylim)

    plt.plot(train_size, training_time, 'o-', color='r', label='Training time')
    plt.plot(train_size, predict_time, 'o-', color='g', label='Prediction time')

    plt.legend(loc='best')

    plt.show()

    plt.figure()
    plt.title(title)
    plt.xlabel('size of training set')
    plt.ylabel(scoring)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.plot(train_size, train_scores, 'o-', color='r', label='Training Score')
    plt.plot(train_size, test_scores, 'o-', color='g', label='Testing Score')

    plt.legend(loc='best')


    if file is not None:
        plt.savefig('../images/'+file, bbox_inches='tight')

    plt.show()


def plot_boost_estimators_curve(estimator, X_train, y_train, X_test, y_test, title='Performance By Learner', ylim=None, scoring='accuracy', file=None):
    """Based on scikit learn documentation: https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html"""

    train_scores, test_scores = [], []
    num_est = len(estimator.estimators_)
    y_train_preds = estimator.staged_predict(X_train)
    y_test_pred = estimator.staged_predict(X_test)
    for p in y_train_preds:
        if scoring == 'f1_macro':
            train_scores.append(f1_score(y_train, p,average='macro'))

        elif scoring == 'recall':
            train_scores.append(recall_score(y_train, p))


    for p in y_test_pred:
        if scoring == 'f1_macro':
            test_scores.append(f1_score(y_test, p,average='macro'))
        elif scoring == 'recall':
            test_scores.append(recall_score(y_test, p))

    plt.xlabel('Estimators in Ensemble')
    plt.ylabel(scoring)
    plt.plot(range(num_est), train_scores, '-', color='r', label='Training Score')
    plt.plot(range(num_est), test_scores, '-', color='g', label='Testing Score')

    plt.legend(loc='best')


    if file is not None:
        plt.savefig('../images/'+file, bbox_inches='tight')

    plt.show()

def plot_learning_curve_neural_net(nnet, X_train, y_train, X_test, y_test, title='Learning Curve', ylim=None, scoring='accuracy', file=None):
    """Based on scikit learn documentation: https://scikit-learn.org/0.15/auto_examples/plot_learning_curve.html"""
    params = nnet.get_params()
    nnet.set_params(**{'warm_start':True})
    epochs = params.get('max_iter',200)
    train_scores, test_scores, epoch = [], [], []
    for e in range(epochs):
        nnet.partial_fit(X_train, y_train, np.unique(y_train))
        y_train_pred = nnet.predict(X_train)
        y_test_pred = nnet.predict(X_test)
        if e%20 == 0:
            epoch.append(e)
            if scoring == 'f1_macro':
                train_scores.append(f1_score(y_train, y_train_pred,average='macro'))
                test_scores.append(f1_score(y_test, y_test_pred,average='macro'))
            elif scoring == 'recall':
                train_scores.append(recall_score(y_train, y_train_pred))
                test_scores.append(recall_score(y_test, y_test_pred))

    plt.figure()
    plt.title(title)
    plt.xlabel('Training Epoch')
    plt.ylabel(scoring)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.plot(epoch, train_scores, '-', color='r', label='Training Score')
    plt.plot(epoch, test_scores, '-', color='g', label='Testing Score')

    plt.legend(loc='best')


    if file is not None:
        plt.savefig('../images/'+file, bbox_inches='tight')

    plt.show()
def plot_curve(data, file=None):
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

    if file is not None:
        plt.savefig('../images/'+file, bbox_inches='tight')

    plt.show()

def plot_confusion_matrix(y_pred, y_true, classes=None, title='Confusion Matrix', normalize=True, file=None):
    classes_int = np.arange(len(classes))

    cm = confusion_matrix(y_true,y_pred, classes_int)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fmt = '.2f'
    else:
        fmt = 'd'

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if file is not None:
        plt.savefig('../images/'+file, bbox_inches='tight')

    plt.show()

def best_hyperparameter_search(estimator, X, y, params, scoring='accuracy', cv=2, graph=False, file=None):
    # param_grid = ParameterGrid(params)

    clf = GridSearchCV(estimator, params, cv=cv, scoring=scoring, return_train_score=True)
    clf.fit(X, y)

    results = clf.cv_results_

    print '{} gave the best score with {} mean test {}\nmean fit time of {} +- {}secs\nmean predict time {} +- {}secs'.format(clf.best_params_,
                                                                                                                              clf.best_score_,
                                                                                                                              scoring,
                                                                                                                              results['mean_fit_time'][clf.best_index_],
                                                                                                                              results['std_fit_time'][clf.best_index_],
                                                                                                                              results['mean_score_time'][clf.best_index_],
                                                                                                                              results['std_score_time'][clf.best_index_])
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
                        'label': 'Validation Set'
                    },
                        {
                        'metrics':results.get('mean_train_score'),
                        'style': '-',
                        'color': 'r',
                        'label': 'Train Set'
                    }
            ]
        }
        plot_curve(data, file=file)


    return clf.best_estimator_

def print_class_counts(y_train, y_test, normalize=False):


    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    ii = np.nonzero(train_counts)[0]
    if normalize:
        train_counts /= np.sum(train_counts)
        test_counts /= np.sum(test_counts)
    y_tr = np.vstack((ii,train_counts[ii]))
    y_te = np.vstack((ii,test_counts[ii]))
    print('train counts:\n{}\ntest counts:\n{}'.format(y_tr,y_te))
