'''
Tuan Nguyen & Randal Olson

Summer 2016

Comparing k's in k-fold CV Project

'''

import sys
import csv
import json 
from glob import glob
from collections import defaultdict
from tqdm import tqdm 
import random
import pandas as pd
import itertools
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

#pre-specified models 

RF = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators= 500, n_jobs = -1)) #rmb to change back to 500 

LR = make_pipeline(StandardScaler(), LogisticRegression())

SVC = make_pipeline(StandardScaler(),SVC())        

GBC = make_pipeline(StandardScaler(),GradientBoostingClassifier(n_estimators= 500))

kNN = make_pipeline(StandardScaler(),KNeighborsClassifier())

models = [RF, LR, SVC, GBC, kNN]

kvals = list(range(3,11))+ [15, 20 , 25]

#trying to iterate over all datasets 

with open('scores.csv', 'w') as csvfile:
    fieldnames = ['dataset', 'name', 'k', 'avg_score', 'sd_score', 'cv_scores']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = '\t')
    writer.writeheader()
    for dataset in tqdm(glob('data/*.csv.gz')):
        #small dataset for testing 
        # if 'iris.' not in dataset: 
#             continue
#         print(dataset)
    
        # Read the data set into memory
        input_data = pd.read_csv(dataset, compression='gzip', sep='\t')
        features = input_data.drop('class', axis=1).values.astype(float)
        labels = input_data['class'].values                                                                                                                    
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, stratify=labels, train_size=0.75, test_size=0.25, random_state=77)
    
        for (k, clf) in itertools.product(kvals,models):
            # Create the pipeline for each model
            try:
                cv_scores = cross_validation.cross_val_score(estimator=clf, X=X_train, y=y_train, cv=k)
                avg_score = np.mean(cv_scores)
                sd_score = np.std(cv_scores)
                name = clf.steps[1][1].__class__.__name__
                writer.writerow({'dataset': dataset.split('/')[-1][:-7], 'name': name, 'k':k , 
                'avg_score': avg_score, 'sd_score': sd_score, 'cv_scores':','.join([str(s) for s in cv_scores])}) 
            
            except KeyboardInterrupt:
                sys.exit(1)
            except:
                continue
        
        for clf in models: 
            try:
                clf.fit(X_train, y_train)
                name = clf.steps[1][1].__class__.__name__
                test_score = clf.score(X_test, y_test)
                writer.writerow({'dataset': dataset.split('/')[-1][:-7], 'name': name, 'k': 0, 
                'avg_score': test_score, 'sd_score': 0, 'cv_scores':test_score})                
            except KeyboardInterrupt:
                sys.exit(1)
            except:
                continue       
