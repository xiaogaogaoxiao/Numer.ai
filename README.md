# Numer.ai
Python script of the numer.ai competition for stock market prediction. The strategy here is to rely on an ensembling method called "Meta Bagging" (see:  https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14295)

I rewrote the code from Mike Kims original R implementation into Python using sci-kit learn. Also, I changed the algorithms from random forest and gradient boosting to pure logistic regression for both base and stacker model which works best for the numer.ai data.

t-sne resulted in complete noise and the embeddings were therefore not used as additional features. 
