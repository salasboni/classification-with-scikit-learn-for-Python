__author__ = 'salasboni'

import pylab as pl
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import Classifier_module as mymodule

allData = {'freq_2','freq_4','freq_6','sqrt','abs','sq'}
scoring_function = 'precision'
pathForResults = '/Users/IEEE_project/results/'
pathForData = '/Users/IEEE_project/data/'

for name in allData:

    array = mymodule.clfProcessor(name, pathForData, scoring_function)

    clf = array[0]
    X = array[1]
    y = array[2]

    # Fit classifier to data and produce classification report and confusion martix
    clf.fit( X, y)
    y_pred =  clf.predict( X)
    cm = confusion_matrix( y, y_pred)

    print('Classifiation report')
    print(classification_report(y, y_pred))
    print('~~~~~~~~~~~~~')
    print('Confusion matrix')
    print(cm)
    print('~~~~~~~~~~~~~')

    probas_ = clf.fit(X, y).predict_proba(X)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve( y, probas_[:, 1] )
    roc_auc = auc( fpr, tpr)

    # Plot ROC curve and save to png file
    fig = pl.figure()
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC curve')
    pl.legend(loc="lower right")
    nameFig = pathForResults + '/ROC_curve_' + name + '.png'
    pl.savefig( nameFig, bbox_inches = 'tight')


