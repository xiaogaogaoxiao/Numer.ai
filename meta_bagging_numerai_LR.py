#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:30:10 2017

@author: Hannes
"""

# Meta Bagging adapted from Mike Kim and rewritten in python


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


print("loading data")
training = pd.read_csv('../numerai_datasets/numerai_training_data.csv', header=0)
testing = pd.read_csv('../numerai_datasets/numerai_tournament_data.csv', header=0)
print("\nShape of train {} and test {}".format(training.shape, testing.shape))


ytrain = training["target"]
train = training.drop("target", axis=1)
test = testing.drop("t_id", axis=1)
data = pd.concat([train, test], axis=0)
data = np.asmatrix(data)


trind = range(0,len(train))
teind = range(len(train), len(data))


Xtrain = data[trind, :]
Xtest = data[teind, :]


clf = LogisticRegression(n_jobs=-1)
clf.fit(Xtrain, ytrain)

ypred = clf.predict_proba(Xtest)

tmpC = range(1, 100)
tmpL = len(trind)


# Meta bagging loop
# for now base learner and stacker model are LogReg. consider changing to alternative algo.

for i in tmpC:
  print("\nIteration:", i)
  tmpS1 = np.random.choice(trind, size=tmpL, replace=True)
  tmpS2 = list(set(trind) - set(tmpS1))

  tmpX2 = Xtrain[tmpS2, :]
  tmpY2 = ytrain[tmpS2]

  print("\nTraining BaseLearner:")
  logreg = LogisticRegression(n_jobs=-1).fit(tmpX2, tmpY2)

  tmpX1 = Xtrain[tmpS1, :]
  tmpY1 = ytrain[tmpS1]

  print("\nPredicting MetaFeatures:")
  tmpX2 = logreg.predict_proba(tmpX1)
  tmpX3 = logreg.predict_proba(Xtest)

  print("\nTraining Stacker:")
  logreg2 = LogisticRegression(n_jobs=-1)
  logreg2.fit(np.concatenate((tmpX1, tmpX2), axis=1), tmpY1)

  ypred0 = logreg2.predict_proba(np.concatenate((Xtest, tmpX3), axis=1))
  ypred = ypred + ypred0
  print("--"*30)

ypred = ypred/(i+1)
print("\nDone!")

submission = pd.DataFrame({"t_id": testing["t_id"],
              "probability": ypred[:, 1]})

cols = submission.columns.tolist()
cols = cols[-1:] + cols[:-1]
submission = submission[cols]

submission.to_csv("numerai_submission.csv", index=False)
