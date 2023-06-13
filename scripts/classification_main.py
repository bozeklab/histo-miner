#Lucas Sancéré -

# import sys
# sys.path.append('../')  # Only for Remote use on Clusters



import numpy as np
from sklearn import linear_model, ensemble

# import scipy.stats
# import sys
# import pandas as pd
# import mrmr
# import boruta
# from sklearn.ensemble import RandomForestClassifier
# import json
# import os
# import time
# import subprocess





"""
This file is to update fully
We will abandon a bit the mrmr repo to apply all the classifider here 

-> needs to play with the last inferences from hvn to updates these repo
"""

# Choose classification you want to run

run_mrmr = True
run_boruta = True
run_mannwhitney = True




######### CLASSIFIERS ################  /!\   -----> PREDICTION ON TRAINING DATA§ Not good!! Split the set into 2 when more data!!!!!
#For now we just check that the code is working, it is a debugging step

#https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#https://scikit-learn.org/stable/modules/linear_model.html


#### STart by loading the files preeviously saved



####### Classifier using as input all the data
genfeatarray = np.transpose(featarray)

###### RIDGE CLASSIFIER
# Initialize the RidgeClassifier with an alpha value of 0.5
ridge = linear_model.RidgeClassifier(alpha=0.5)
# Fit the classifier to the data
ridge.fit(genfeatarray, clarray)
# Predict the labels for new data
ridge_pred = ridge.predict(genfeatarray)
print('ridge_pred : {}'.format(ridge_pred))
# Print the accuracy of the classifier
print("Accuracy of RIDGE classifier:", ridge.score(genfeatarray, clarray))

##### LOGISTIC REGRESSION
lr = linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')
# Fit the classifier to the data
lr.fit(genfeatarray, clarray)
# Predict the labels for new data
lr_pred = lr.predict(genfeatarray)
print('lr_pred : {}'.format(lr_pred))
# Print the accuracy of the classifier
print("Accuracy of LOGISTIC classifier:", lr.score(genfeatarray, clarray))

##### RANDOM FOREST
forest = ensemble.RandomForestClassifier(n_estimators=100, class_weight='balanced')
# Fit the classifier to the data
forest.fit(genfeatarray, clarray)
# Predict the labels for new data
forest_pred = forest.predict(genfeatarray)
print('forest_pred : {}'.format(forest_pred))
# Print the accuracy of the classifier
print("Accuracy of RANDOM FOREST classifier:", forest.score(genfeatarray, clarray))


if run_mrmr:
    # Generate the matrix with selected feature for mrmr
    mrmrselectedfeatures_idx = Selfeat_mrmr[0]
    mrmrselectedfeatures_idx = sorted(mrmrselectedfeatures_idx)
    featarrar_mrmr = np.transpose(featarray)
    featarray_mrmr = featarrar_mrmr[:, mrmrselectedfeatures_idx]
    print('Shape featarray_mrmr', featarray_mrmr.shape)
    print('Shape clarray', clarray.shape)

    ###### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier with an alpha value of 0.5
    ridge_mrmr = linear_model.RidgeClassifier(alpha=0.5)
    # Fit the classifier to the data
    ridge_mrmr.fit(featarray_mrmr, clarray)
    # Predict the labels for new data
    ridge_mrmr_pred = ridge_mrmr.predict(featarray_mrmr)
    print('ridge_mrmr_pred : {}'.format(ridge_mrmr_pred))
    # Print the accuracy of the classifier
    print("Accuracy of RIDGE MRMR classifier:", ridge_mrmr.score(featarray_mrmr, clarray))

    ##### LOGISTIC REGRESSION
    lr_mrmr = linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')
    # Fit the classifier to the data
    lr_mrmr.fit(featarray_mrmr, clarray)
    # Predict the labels for new data
    lr_mrmr_pred = lr_mrmr.predict(featarray_mrmr)
    print('lr_mrmr_pred : {}'.format(lr_mrmr_pred))
    # Print the accuracy of the classifier
    print("Accuracy of LOGISTIC MRMR classifier:", lr_mrmr.score(featarray_mrmr, clarray))

    ##### RANDOM FOREST
    forest_mrmr = ensemble.RandomForestClassifier(n_estimators=100, class_weight='balanced')
    # Fit the classifier to the data
    forest_mrmr.fit(featarray_mrmr, clarray)
    # Predict the labels for new data
    forest_mrmr_pred = forest_mrmr.predict(featarray_mrmr)
    print('forest_mrmr_pred : {}'.format(forest_mrmr_pred))
    # Print the accuracy of the classifier
    print("Accuracy of RANDOM FOREST MRMR classifier:", forest_mrmr.score(featarray_mrmr, clarray))


if run_boruta:
    # here the matrix with selected feature is already done
    print('Shape selfeat_boruta', Selfeat_boruta.shape)
    print('Shape clarray', clarray.shape)

    ###### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier with an alpha value of 0.5
    ridge_boruta = linear_model.RidgeClassifier(alpha=0.5)
    # Fit the classifier to the data
    ridge_boruta.fit(Selfeat_boruta, clarray)
    # Predict the labels for new data
    ridge_boruta_pred = ridge_boruta.predict(Selfeat_boruta)
    print('ridge_boruta_pred : {}'.format(ridge_boruta_pred))
    # Print the accuracy of the classifier
    print("Accuracy of RIDGE BORUTA classifier:", ridge_boruta.score(Selfeat_boruta, clarray))

    ##### LOGISTIC REGRESSION
    lr_boruta = linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')
    # Fit the classifier to the data
    lr_boruta.fit(Selfeat_boruta, clarray)
    # Predict the labels for new data
    lr_boruta_pred = lr_boruta.predict(Selfeat_boruta)
    print('lr_boruta_pred : {}'.format(lr_boruta_pred))
    # Print the accuracy of the classifier
    print("Accuracy of LOGISTIC BORUTA classifier:", lr_boruta.score(Selfeat_boruta, clarray))

    ##### RANDOM FOREST
    forest_boruta = ensemble.RandomForestClassifier(n_estimators=100, class_weight='balanced')
    # Fit the classifier to the data
    forest_boruta.fit(Selfeat_boruta, clarray)
    # Predict the labels for new data
    forest_boruta_pred = forest_boruta.predict(Selfeat_boruta)
    print('forest_boruta_pred : {}'.format(forest_boruta_pred))
    # Print the accuracy of the classifier
    print("Accuracy of RANDOM FOREST BORUTA classifier:", forest_boruta.score(Selfeat_boruta, clarray))


if run_mannwhitney:
    # Generate the matrix with selected feature for mannwhitney
    mwuselectedfeatures_idx = Orderedp_mannwhitneyu[:Nbr_keptfeat - 1]
    mwuselectedfeatures_idx = sorted(mwuselectedfeatures_idx)
    featarray_mannwhitney = np.transpose(featarray)
    featarray_mannwhitney = featarray_mannwhitney[:, mwuselectedfeatures_idx]
    print('Shape featarray_mannwhitney', featarray_mannwhitney.shape)
    print('Shape clarray', clarray.shape)

    ###### RIDGE CLASSIFIER
    # Initialize the RidgeClassifier with an alpha value of 0.5
    ridge_mannwhitney = linear_model.RidgeClassifier(alpha=0.5)
    # Fit the classifier to the data
    ridge_mannwhitney.fit(featarray_mannwhitney, clarray)
    # Predict the labels for new data
    ridge_mannwhitney_pred = ridge_mannwhitney.predict(featarray_mannwhitney)
    print('ridge_mannwhitney_pred : {}'.format(ridge_mannwhitney_pred))
    # Print the accuracy of the classifier
    print("Accuracy of RIDGE MANN WHITNEY classifier:", ridge_mannwhitney.score(ridge_mannwhitney_pred, clarray))

    ##### LOGISTIC REGRESSION
    lr_mannwhitney = linear_model.LogisticRegression(solver='liblinear', multi_class='ovr')
    # Fit the classifier to the data
    lr_mannwhitney.fit(featarray_mannwhitney, clarray)
    # Predict the labels for new data
    lr_mannwhitney_pred = lr_mannwhitney.predict(featarray_mannwhitney)
    print('lr_mannwhitney_pred : {}'.format(lr_mannwhitney_pred))
    # Print the accuracy of the classifier
    print("Accuracy of LOGISTIC MANN WHITNEY classifier:", lr_mannwhitney.score(lr_mannwhitney_pred, clarray))

    ##### RANDOM FOREST
    forest_mannwhitney = ensemble.RandomForestClassifier(n_estimators=100, class_weight='balanced')
    # Fit the classifier to the data
    forest_mannwhitney.fit(featarray_mannwhitney, clarray)
    # Predict the labels for new data
    forest_mannwhitney_pred = forest_mannwhitney.predict(featarray_mannwhitney)
    print('forest_mannwhitney_pred : {}'.format(forest_mannwhitney_pred))
    # Print the accuracy of the classifier
    print("Accuracy of RANDOM FOREST MANN WHITNEY classifier:", forest_mannwhitney.score(featarray_mannwhitney, clarray))


print('GT class:', y)
print('done')







