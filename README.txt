Code can be found at: https://github.com/albutko/ml_supervised_learning
Data is stored in the data directory, so clone the repo and you will have everything

My HW1 implementation was written using python, scikit-learn, matplotlib, numpy, and pandas.

First install packages using pip from requirements.txt with command:
    pip install -r requirements.txt


this should install all necessary packages.
Next, if you want to recreate any of my results, you will do so in code/experiments.py file

In the main() method you will find a list of 20 function calls. There are two types
of functions here:
    1. *****Experiment(dataset)
    2. *****BestClassifier(dataset)

Each of these functions use an if statement to path the analysis based on the dataset so make
sure you are commenting the correct code out in the correct if-statement when necessary

These functions take in a dataset as input: higgs or mapping.

The *Experiment(*dataset*) method will run the cross validation and grid search experiments I used
for hyperparameter tuning and will use matplotlib and the standard output to visualize results.
matplotlib windows must be closed for the program to continue outputting results.

The *BestClassifier(*dataset*) method will train and test the best classifier configuration
for the associated classifier.

To run any of these methods simply uncomment the method calls in main() for the specific dataset
you wish to test and run `python experiments.py` from the code directory on the command line

For some experiments, I searched across multiple parameters, for these I have the different
parameter dictionaries commented out in the Experiment functions. You must uncomment what you
wish to test. Also if there is more than one parameter in the search you must set
graph=False in the `best_hyperparameter_search()` function as it can only graph change due
to one parameter

For SVM because two kernels were tested you must make sure to go into the SVMExperiment and
SVMBestClassifierTest to uncomment the parameter dict or kernel you wish to use.
