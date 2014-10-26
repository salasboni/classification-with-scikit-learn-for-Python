__author__ = 'salasboni'

import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import f1_score, precision_score, roc_curve, auc, roc_auc_score
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import KFold
from sklearn.svm import l1_min_c

def partial_auc(clf, X, y):
    """
    Computes the parital area under the curve, for fals epositive rate less than 0.15
    :param clf: a classifier object
    :param X: The observation set
    :param y: The label set
    """
    probas_ = clf.predict_proba(X)
    fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
    aa = fpr.compress((fpr < .15).flat)
    pAUC = auc(aa, tpr[0:len(aa)])
    return pAUC


def getFileNames(name , pathForData):
    """
    Generates the file names for the data set, given the parameters
    :param  name: the name of the data extraction procedure, eg freq_2, abs, etc
    :param pathForData: path of the data directory
    """
    fileMeasTrControl = pathForData + "trainingSet_Control_" + name + ".txt"
    fileMeasTrCA = pathForData + "trainingSet_CA_" + name + ".txt"
    return fileMeasTrControl, fileMeasTrCA

def getXandY(fileNameCo, fileNameCA):
    """
    Loads the dataset, adn generats the label array.
    :param fileNameCo: the name of the control dataset
    :param fileNameCA: the name of the cases dataset
    """
    XCon = np.loadtxt(fileNameCo)
    XCA = np.loadtxt(fileNameCA)
    n1, nCon = XCon.shape
    n1, nCA = XCA.shape
    X = np.concatenate((XCon, XCA), axis=1)
    y = np.concatenate((np.zeros(shape=(1, nCon)), np.ones(shape=(1, nCA))), axis=1)
    X = X.transpose()
    y = y.transpose()
    y = y.ravel()
    return X, y


def myScoreFunction( probas_, y_test, thr, scoring_function):
    """
    Loads the dataset, adn generats the label array.
    :param probas_: array with probabilities of each observation belonging to each class
    :param y_test: labels of test data
    :param thr: theshold for decision, must be between 0 and 1
    :param scoring_function: score function to optimize, eg f1, precision, etc
    """
    dispatch = {'f1': f1_score, 'precision': precision_score, 'auc': roc_auc_score}
    pp = ( probas_ > thr)
    thisScore = ( dispatch[scoring_function](y_test, pp.astype(int)[:, 1]))
    return thisScore


def myCrossValidation(X, y, kf, scoring_function, grid):
    """
    Performs cross validation, finding both the optimal combinaton of
    hyperparameters for the classifier and the optimal theshold for the
    decision boundary, for a given scoring function.
    :param X: Dataset
    :param y: Labels
    :param kf: Cross validation partition
    :param scoring_function: Scoring function to optimize, eg f1 score, precision, etc.
    :param grid: Grid of values for all the hyperparameters to optimize
    """
    thresholds = np.arange(.3, .7, 0.05)
    bestScore = 0
    bestParameters = {}

    for parameters in list(ParameterGrid(grid)):

        allScores = np.zeros(len(kf))
        allThresholds = np.zeros(len(kf))
        ii = 0

        for train, test in kf:

            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            clf = linear_model.LogisticRegression( penalty = 'l1', **parameters)
            clf.fit(X_train, y_train)
            probas_ = clf.predict_proba(X_test)
            allScores1 = [0]*len(thresholds)

            for i in range(len(thresholds)):
                allScores1[i] = myScoreFunction(probas_, y_test, thresholds[i], scoring_function)

            maxInd = allScores1.index(max(allScores1))
            bestScore, bestThreshold = allScores1[maxInd], thresholds[maxInd]
            allScores[ii] = bestScore
            allThresholds[ii] = bestThreshold
            ii += 1

        totalScore = np.mean(allScores)

        if totalScore > bestScore:
            bestParameters = parameters

    bestScore = np.mean(allScores)
    bestThreshold = np.mean(allThresholds)

    return bestParameters, bestScore, bestThreshold


def clfProcessor(name, pathForData, scoring_function):

    """
    Loads the data, scales the data, defines a grid for the hyperparameters
    for a l1-regularized logistic regression classifier, performs L1-based
    feature selection, and finds the best hyperparameters via cross validation.
    :param name: Name of data extrcation procedure, eg freq_2, abs, etc
    :param pathForData: Path for directory where thr data is
    :param scoring_function: Scoring function to optimize, eg f1, precision, etc.
    :return clf: Object with the optimal classifier
    :return X: Scaled dataset
    :return y: Labels
    :return X_max: Array with maximal values of X, to reproduce transform of the traning set
    :return X_min: Array with minimal values of X, to reproduce transform of the traning set
    """

    # Load training data. Each column is an observation

    fileMeasTrControl, fileMeasTrCA = getFileNames(name, pathForData)
    X, y = getXandY(fileMeasTrControl, fileMeasTrCA)

    # Scaling to [0,1] intervals and saving transform to apply in test data
    X_max = X.max(axis=0)
    X_min = X.min(axis=0)
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # Shuffle data
    n_samples, n2 = X.shape
    order = np.random.permutation(n_samples)
    X = X[order, :]
    y = y[order].astype(np.float)

    # L1 based feature selection
    theTransform = linear_model.LogisticRegression(C=1, penalty='l1', dual=False, class_weight={1: 2})  # LinearSVC
    X = theTransform.fit_transform( X, y)

    # Find minimum C for non-empty model and get grid for cross validation
    cs_log = np.logspace( 0, 4, num=30)
    l1_min = l1_min_c( X, y, loss='log')
    cs = l1_min * cs_log[10:]
    grid = {'C': cs, 'class_weight': [{1: 1}, {1: 2}, {1: 3}, {1: 5}]}

    # Perform grid search cross validation
    kf = KFold( len(y), n_folds = 5)
    bestParameters = myCrossValidation( X, y, kf, scoring_function, grid)
    clf = linear_model.LogisticRegression( penalty = 'l1', **bestParameters)

    return [clf, X, y, X_max, X_min]
