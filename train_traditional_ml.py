# Import libraries
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle


def load_data(data_path):
    train_df = pd.read_csv(data_path)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    annovar_train = train_df.loc[:, train_df.columns != 'db'].copy()
    # Delete columns with highest NaN values
    headers_to_drop = ['LINSIGHT_rankscore', 'M-CAP_rankscore', 'MutPred_rankscore', 'MVP_rankscore']
    annovar_train.drop(headers_to_drop, axis=1, inplace=True)
    # Drop NAN values from rankscores
    headers = [col for col in annovar_train.columns if 'rankscore' in col]
    annovar_train = annovar_train.dropna(subset=headers).reset_index(drop=True).copy()
    # Drop NAN values from final variables
    headers = ['PVS1', 'PS1', 'PS2', 'PS3', 'PS4', 'PM1', 'PM2', 'PM3', 'PM4', 'PM5', 'PM6', 'PP1', 'PP2', 'PP3', 'PP4',
               'PP5', 'BA1', 'BS1', 'BS2', 'BS3', 'BS4', 'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7']
    annovar_train = annovar_train.dropna(subset=headers).reset_index(drop=True).copy()
    # Delete colums not providing any information
    drop_headers = annovar_train.columns.values
    new_headers = []
    dropped_headers = []
    for head in drop_headers:
        if len(np.unique(annovar_train[head])) == 1:
            annovar_train.drop(head, axis=1, inplace=True)
            dropped_headers.append(head)
            print('Deleted column without information: ' + str(head))
        else:
            new_headers.append(head)
    annovar_train.drop(['#CHROM', 'POS', 'Arm', 'Band', 'Sub-band', 'Subsub-band'], axis=1, inplace=True)
    features_of_interest = annovar_train.columns.difference(['gt'])
    x = np.array(annovar_train[features_of_interest])
    y = np.array(annovar_train['gt'].astype(int))
    return x, y

if __name__=='__main__':
    # Read vcf file
    data_path = "data/train.csv"
    X_train, y_train = load_data(data_path)
    data_path = "data/test.csv"
    X_test, y_test = load_data(data_path)

    print('Training data cases: ', len(X_train), ' Test data cases: ', len(X_test))

    """### ALL CLASSIFICATION TRAINING"""
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)

    # Models to be tried

    models = {'Logistic': LogisticRegression(random_state=42, max_iter=1000), 'SVM':SVC(random_state=42, max_iter= 10000),
              'DecisionTree': DecisionTreeClassifier(random_state=42), 'RF': RandomForestClassifier(random_state=42),
              'XGB': XGBClassifier(random_state=42)}

    hyperpar = {'Logistic': {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                             'C':  [100, 10, 1.0, 0.1, 0.01],
                             'tol': [1e-3, 1e-4, 1e-5]},
                'SVM': {'C': [0.01, 0.1, 1, 10, 100],
                        'kernel': ['poly', 'rbf', 'sigmoid', 'linear'],
                        'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
                        'tol': [1e-3, 1e-4, 1e-2]},
                'DecisionTree': {'max_depth': list(range(2, 20)),
                                 'criterion': ['gini', 'entropy']},
                'RF': {'max_depth': list(range(2, 10)),
                       'criterion': ['gini', 'entropy'],
                       'max_features': ["auto", "log2", "None"],
                       'n_estimators': [100, 200, 300, 400, 500, 600],
                       'bootstrap': [True, False]},
                'XGB': {'n_estimators': [100, 200, 300, 400, 500, 600],
                        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5],
                        'max_depth': list(range(2, 10)),
                        'booster': ['gbtree', 'gblinear', 'dart']}}

    final_A = {}
    final_P = {}
    final_R = {}
    final_F1 = {}

    for architecture in models:
      cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
      model = models[architecture]

      space = hyperpar[architecture]
      search = RandomizedSearchCV(estimator=model, param_distributions=space, scoring='f1', cv=cv_inner, refit=True, n_iter=30, n_jobs=-1, verbose=3)

      result = search.fit(X_train, y_train)
      # get the best performing model fit on the whole training set
      best_model = result.best_estimator_
      path = "/resluts/" + architecture + ".h5"
      pickle.dump(best_model, open(path, 'wb'))
      # evaluate model on the hold out dataset
      yhat = best_model.predict(X_test)
      # Precision
      P = metrics.precision_score(y_test,yhat)
      final_P[architecture] = P
      # Recall
      R = metrics.recall_score(y_test,yhat)
      final_R[architecture] = R
      # F1 score
      F1 = metrics.f1_score(y_test,yhat)
      final_F1[architecture] = F1
      # Accuracy
      acc = metrics.accuracy_score(y_test, yhat)
      final_A[architecture] = acc
        # report progress
      print('#####################################################################')
      print(architecture)
      print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
      print('#####################################################################')