from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, recall_score
from timeit import default_timer as timer

from datasets import HiggsBosonDataset, MappingDataset
from utils import *

higgs = HiggsBosonDataset()
mapping = MappingDataset()

def main():
    DecisionTreeExperiment(higgs)
    # DecisionTreeBestClassifierTest(higgs)
    # DecisionTreeExperiment(mapping)
    # DecisionTreeBestClassifierTest(mapping)
    # BoostingExperiment(higgs)
    # BoostingExperiment(mapping)
    # BoostingBestClassifierTest(higgs)
    # BoostingBestClassifierTest(mapping)
    # KNNExperiment(higgs)
    # KNNBestClassifierTest(higgs)
    # KNNExperiment(mapping)
    # KNNBestClassifierTest(mapping)
    # NeuralNetExperiment(higgs)
    # NeuralNetBestClassifierTest(higgs)
    # NeuralNetExperiment(mapping)
    # NeuralNetBestClassifierTest(mapping)
    # SVMExperiment(higgs)
    # SVMBestClassifierTest(higgs)
    # SVMExperiment(mapping)
    # SVMBestClassifierTest(mapping)

def DecisionTreeExperiment(dataset):
    print('Running Decision Tree Experiment')

    X, y = dataset.get_train_data()
    classes = dataset.get_classes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

    if dataset.name == 'higgs':
        clf = DecisionTreeClassifier()
        params = {'min_samples_split':np.linspace(100,500,100).astype('int32')}

        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='recall', cv=5, graph=True)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, classes = dataset.classes)
    else:
        clf = DecisionTreeClassifier()
        params = {'min_samples_split':np.linspace(2,500,20).astype('int32')}
        # params = {'max_leaf_nodes':np.linspace(100,1000,50).astype('int32')}
        # params = {'max_leaf_nodes':np.linspace(2,500,20).astype('int32'),
        #           'min_samples_split':np.linspace(2,500,20).astype('int32')}
        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='f1_macro', cv=5, graph=True)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, classes = dataset.classes)

    return clf

def DecisionTreeBestClassifierTest(dataset):
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    classes = dataset.get_classes()
    if dataset.name == 'higgs':
        clf = DecisionTreeClassifier(min_samples_split=320)
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='recall')
        start = timer()
        clf = clf.fit(X_train, y_train)
        print('Fit time:{} secs'.format(timer() - start))
        start = timer()
        y_pred = clf.predict(X_test)
        print('Predict time:{} secs'.format(timer() - start))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)
        print('Train Recall:{}%'.format(recall_score(y_train,clf.predict(X_train),pos_label=1)*100))
        print('Test Recall:{}%'.format(recall_score(y_test,y_pred,pos_label=1)*100))

    else:
        clf = DecisionTreeClassifier(max_leaf_nodes=190)
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='f1_macro')
        start = timer()
        clf = clf.fit(X_train, y_train)
        print('Fit time:{} secs for {} instances'.format(timer() - start,y_train.shape[0]))
        start = timer()
        y_pred = clf.predict(X_test)
        print('Predict time:{} secs for {} instances'.format(timer() - start,y_test.shape[0]))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, classes = dataset.classes)
        print('Train F1 Macro:{}%'.format(f1_score(y_train,clf.predict(X_train),average='macro',labels=range(len(dataset.classes)))*100))
        print('Test F1 Macro:{}%'.format(f1_score(y_test,y_pred,average='macro',labels=range(len(dataset.classes)))*100))


def BoostingExperiment(dataset):
    print('Running Boosting Experiment')

    X, y = dataset.get_train_data()
    classes = dataset.get_classes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    if dataset.name == 'higgs':
        base_est = DecisionTreeClassifier(min_samples_split=500)
        clf = AdaBoostClassifier(base_estimator=base_est)

        clf.fit(X_train, y_train)

        params = {'learning_rate':[0.01,0.1,1,10,100]}

        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='recall', cv=2, graph=True)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, X_test, estimator = clf, classes = dataset.classes)

    else:
        base_est = DecisionTreeClassifier(max_leaf_nodes=50)
        params = {'learning_rate':[0.01,0.1,1,10,100]}
        clf = AdaBoostClassifier(base_estimator=base_est)

        clf.fit(X_train, y_train)

        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='f1_macro', cv=2, graph=True)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)
        plot_roc_curve(y_score, y_test, classes = dataset.classes)

    return clf


def BoostingBestClassifierTest(dataset):
    print('Running Best Boosting Test')
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    classes = dataset.get_classes()

    if dataset.name == 'higgs':
        base_est = DecisionTreeClassifier(min_samples_split=500)
        clf = AdaBoostClassifier(base_estimator=base_est, learning_rate=0.1)
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='recall')
        start = timer()
        clf = clf.fit(X_train, y_train)
        fit_time = timer() - start
        print('Fit time:{} secs for {} instances ({} inst/per sec)'.format(fit_time,y_train.shape[0],y_train.shape[0]/fit_time))
        start = timer()
        y_pred = clf.predict(X_test)
        pred_time = timer() - start
        print('Predict time:{} secs for {} instances ({} inst/per sec)'.format(pred_time,y_test.shape[0],y_test.shape[0]/pred_time))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)
        print('Train Recall:{}%'.format(recall_score(y_train,clf.predict(X_train),pos_label=1)*100))
        print('Test Recall:{}%'.format(recall_score(y_test,y_pred,pos_label=1)*100))

    else:
        base_est = DecisionTreeClassifier(max_leaf_nodes=50)
        clf = AdaBoostClassifier(base_estimator=base_est, learning_rate=1)
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='f1_macro')
        start = timer()
        clf = clf.fit(X_train, y_train)
        fit_time = timer() - start
        plot_boost_estimators_curve(clf, X_train, y_train, X_test, y_test, scoring='f1_macro')
        print('Fit time:{} secs for {} instances ({} inst/per sec)'.format(fit_time,y_train.shape[0]))
        start = timer()
        y_pred = clf.predict(X_test)
        pred_time = timer() - start
        print('Predict time:{} secs for {} instances ({} inst/per sec)'.format(pred_time,y_test.shape[0],y_test.shape[0]/pred_time))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)
        print('Train F1 Macro:{}%'.format(f1_score(y_train,clf.predict(X_train),average='macro',labels=range(len(dataset.classes)))*100))
        print('Test F1 Macro:{}%'.format(f1_score(y_test,y_pred,average='macro',labels=range(len(dataset.classes)))*100))


def KNNExperiment(dataset):
    print('Running KNN Experiment')
    clf = KNeighborsClassifier()

    X, y = dataset.get_train_data()
    classes = dataset.get_classes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state = 23)

    params = {'n_neighbors':np.linspace(60,120,30).astype('int32')}
    if dataset.name == 'higgs':

        params = {'n_neighbors':np.linspace(60,120,30).astype('int32')}

        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='recall', cv=3, graph=True)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, X_test, estimator = clf, classes = dataset.classes)

    else:
        params = {'n_neighbors':np.linspace(1,100,50).astype('int32')}

        clf.fit(X_train, y_train)

        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='f1_macro', cv=3, graph=True)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, classes = dataset.classes)

def KNNBestClassifierTest(dataset):
    print("Running Best KNN Tests")
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    classes = dataset.get_classes()
    if dataset.name == 'higgs':
        clf = KNeighborsClassifier(n_neighbors=97)
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='recall')
        start = timer()
        clf = clf.fit(X_train, y_train)
        fit_time = timer() - start
        print('Fit time:{} secs for {} instances ({} inst/per sec)'.format(fit_time,y_train.shape[0],y_train.shape[0]/fit_time))
        start = timer()
        y_pred = clf.predict(X_test)
        pred_time = timer() - start
        print('Predict time:{} secs for {} instances ({} inst/per sec)'.format(pred_time,y_test.shape[0],y_test.shape[0]/pred_time))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)
        print('Train Recall:{}%'.format(recall_score(y_train,clf.predict(X_train),pos_label=1)*100))
        print('Test Recall:{}%'.format(recall_score(y_test,y_pred,pos_label=1)*100))

    else:
        clf = KNeighborsClassifier(n_neighbors=1)
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='f1_macro')
        start = timer()
        clf = clf.fit(X_train, y_train)
        fit_time = timer() - start
        print('Fit time:{} secs for {} instances ({} inst/per sec)'.format(fit_time,y_train.shape[0],y_train.shape[0]/fit_time))
        start = timer()
        y_pred = clf.predict(X_test)
        pred_time = timer() - start
        print('Predict time:{} secs for {} instances ({} inst/per sec)'.format(pred_time,y_test.shape[0],y_test.shape[0]/pred_time))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)
        print('Train F1 Macro:{}%'.format(f1_score(y_train,clf.predict(X_train),average='macro',labels=range(len(dataset.classes)))*100))
        print('Test F1 Macro:{}%'.format(f1_score(y_test,y_pred,average='macro',labels=range(len(dataset.classes)))*100))

def NeuralNetExperiment(dataset):
    print('Running Neural Net Experiment')

    X, y = dataset.get_train_data()
    classes = dataset.get_classes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state = 23)

    if dataset.name == 'higgs':
        clf = MLPClassifier(hidden_layer_sizes=(100,50))
        params = {'learning_rate_init':[0.001,0.01,0.1,1]}
        params = {'alpha':[0.1,1,10]}
        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='recall', cv=2, graph=False)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, X_test, estimator = clf, classes = dataset.classes)

        clf = MLPClassifier(hidden_layer_sizes=(100,50), alpha=1, max_iter=500)
        plot_learning_curve_neural_net(clf, X_train, y_train, X_test, y_test, title='Learning Curve', scoring='recall')

    else:

        params = {'alpha':[0.01,0.1,1,10],'learning_rate_init':[0.01,0.1,1]}
        clf = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500)

        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='f1_macro', cv=3, graph=False)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, classes = dataset.classes)
        clf = MLPClassifier(hidden_layer_sizes=(100,50), alpha=0.01)
        plot_learning_curve_neural_net(clf, X_train, y_train, X_test, y_test, title='Learning Curve', scoring='recall')

def NeuralNetBestClassifierTest(dataset):
    print("Running Best Neural Net Tests")
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    classes = dataset.get_classes()
    if dataset.name == 'higgs':
        clf = MLPClassifier(hidden_layer_sizes=(100,50), alpha=1, max_iter=500)
        plot_learning_curve_neural_net(clf, X_train, y_train, X_test, y_test, title='Learning Curve', scoring='recall')
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='recall')
        start = timer()
        clf = clf.fit(X_train, y_train)
        fit_time = timer() - start
        print('Fit time:{} secs for {} instances ({} inst/per sec)'.format(fit_time,y_train.shape[0],y_train.shape[0]/fit_time))
        start = timer()
        y_pred = clf.predict(X_test)
        pred_time = timer() - start
        print('Predict time:{} secs for {} instances ({} inst/per sec)'.format(pred_time,y_test.shape[0],y_test.shape[0]/pred_time))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)
        print('Train Recall:{}%'.format(recall_score(y_train,clf.predict(X_train),pos_label=1)*100))
        print('Test Recall:{}%'.format(recall_score(y_test,y_pred,pos_label=1)*100))

    else:
        clf = MLPClassifier((100,50), alpha=0.1, learning_rate_init=0.01, max_iter=500)
        plot_learning_curve_neural_net(clf, X_train, y_train, X_test, y_test, title='Learning Curve', scoring='f1_macro')
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='f1_macro')
        start = timer()
        clf = clf.fit(X_train, y_train)
        fit_time = timer() - start
        print('Fit time:{} secs for {} instances ({} inst/per sec)'.format(fit_time,y_train.shape[0],y_train.shape[0]/fit_time))
        start = timer()
        y_pred = clf.predict(X_test)
        pred_time = timer() - start
        print('Predict time:{} secs for {} instances ({} inst/per sec)'.format(pred_time,y_test.shape[0],y_test.shape[0]/pred_time))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)

        print('Train F1 Macro:{}%'.format(f1_score(y_train,clf.predict(X_train),average='macro',labels=range(len(dataset.classes)))*100))
        print('Test F1 Macro:{}%'.format(f1_score(y_test,y_pred,average='macro',labels=range(len(dataset.classes)))*100))

def SVMExperiment(dataset):
    print('Running SVM Experiment')
    clf = SVC( probability=True)

    X, y = dataset.get_train_data()
    classes = dataset.get_classes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    if dataset.name == 'higgs':
        clf = SVC(probability=True)
        params = {'C':[0.1,1,10,100],'kernel':['rbf'],'gamma':[0.01,0.1,1]}
        # params = {'C':[1,10,100],'kernel':['rbf'],'gamma':[0.1]}
        # params = {'C':[0.1,1,10,100],'kernel':['poly'],'degree':[1,2,3,4]}

        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='recall', cv=2, graph=False)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, classes = dataset.classes)

    else:
        clf = SVC(probability=True)
        # params = {'C':[0.1,1,10,100,1000],'kernel':['rbf'],'gamma':[0.01,0.1,1]}
        params = {'C':[0.1,1,10,100],'kernel':['poly'],'degree':[1,2,3,4]}
        clf = best_hyperparameter_search(clf, X_train, y_train, params, scoring='f1_macro', cv=2, graph=False)
        plot_confusion_matrix(clf.predict(X_test), y_test,classes,normalize=True)
        y_score = clf.predict_proba(X_test)
        plot_roc_curve(y_score, y_test, classes = dataset.classes)



def SVMBestClassifierTest(dataset):
    print("Running Best SVM Tests")
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    classes = dataset.get_classes()
    if dataset.name == 'higgs':
        # clf = SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
        clf = SVC(kernel='poly', C=100, degree=3, probability=True)
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='recall')
        start = timer()
        clf = clf.fit(X_train, y_train)
        fit_time = timer() - start
        print('Fit time:{} secs for {} instances ({} inst/per sec)'.format(fit_time,y_train.shape[0],y_train.shape[0]/fit_time))
        start = timer()
        y_pred = clf.predict(X_test)
        pred_time = timer() - start
        print('Predict time:{} secs for {} instances ({} inst/per sec)'.format(pred_time,y_test.shape[0],y_test.shape[0]/pred_time))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes, file='poly_svm_'+dataset.name+'_roc_test.png')
        print('Train Recall:{}%'.format(recall_score(y_train,clf.predict(X_train),pos_label=1)*100))
        print('Test Recall:{}%'.format(recall_score(y_test,y_pred,pos_label=1)*100))

    else:
        clf = SVC(kernel='poly', C=10, degree=3, probability=True)
        # clf = SVC(kernel='rbf', C=100, gamma=0.01, probability=True)
        plot_learning_curve_train_size(clf, X_train, y_train, X_test, y_test, title='Learning Curve',scoring='f1_macro')
        start = timer()
        clf = clf.fit(X_train, y_train)
        fit_time = timer() - start
        print('Fit time:{} secs for {} instances ({} inst/per sec)'.format(fit_time,y_train.shape[0],y_train.shape[0]/fit_time))
        start = timer()
        y_pred = clf.predict(X_test)
        pred_time = timer() - start
        print('Predict time:{} secs for {} instances ({} inst/per sec)'.format(pred_time,y_test.shape[0],y_test.shape[0]/pred_time))
        plot_confusion_matrix(y_pred, y_test, classes, normalize=True)
        y_score = clf.predict_proba(X_test)

        plot_roc_curve(y_score, y_test, X_test, estimator=clf, classes = dataset.classes)
        print('Train F1 Macro:{}%'.format(f1_score(y_train,clf.predict(X_train),average='macro',labels=range(len(dataset.classes)))*100))
        print('Test F1 Macro:{}%'.format(f1_score(y_test,y_pred,average='macro',labels=range(len(dataset.classes)))*100))


if __name__=='__main__':
    main()
