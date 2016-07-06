import numpy as np
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.cross_validation import train_test_split
from preprocess import AmazonReviews
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb


def main(debug=False):

    # Initiate and preprocess the Amazon reviews
    data = AmazonReviews(debug=debug, is_regression=True)

    print 'Splitting data into train/test split.'
    xtrain, xtest, ytrain, ytest = train_test_split(data.X, data.Y, train_size=0.80)

    print 'Training model.'
    gbm = xgb.XGBRegressor()
    gbm.fit(xtrain, ytrain, verbose=True)
    ypred = gbm.predict(xtest)
    print r2_score(ytest, ypred)

    print gbm.feature_importances_


main(False)
