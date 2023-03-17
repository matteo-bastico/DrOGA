import os
import json
import torch
import random
import numpy as np
import pandas as pd
from datasets.base_dataset import DatasetBase
from models.cnn_model import CNNBinaryClassification, CNNBinaryClassificationSkip
from models.base_model import MLP
from sklearn import metrics
import pickle
import xgboost

random.seed(42)

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
    dropped_headers = []
    headers_dropped_train = ['MUT', 'PVS1', 'PS2', 'PM3', 'PM4', 'PM6', 'PP1', 'PP4', 'BS3', 'BS4', 'BP2', 'BP3',
                             'BP5', 'BP7', 'CLNSIG10', 'CLNSIG13']
    annovar_train = annovar_train.drop(headers_dropped_train, axis=1)
    annovar_train.drop(['#CHROM', 'POS', 'Arm', 'Band', 'Sub-band', 'Subsub-band'], axis=1, inplace=True)
    dataset = DatasetBase(annovar_train, labels='gt')
    features_of_interest = annovar_train.columns.difference(['gt'])
    x = np.array(annovar_train[features_of_interest])
    y = np.array(annovar_train['gt'].astype(int))
    print("Test Samples %i" % len(dataset))
    return dataset, annovar_train.columns, x, y


def get_metrics(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    tp = y_pred_tag[y_test == 1].sum().float()
    fp = y_pred_tag[y_test == 0].sum().float()

    tot_p = (y_test == 1).sum()
    tot_n = (y_test == 0).sum()

    fn = tot_p - tp
    tn = tot_n - fp

    return tp, fp, fn, tn

def eval_ml(model_name, model, X_test, y_test):
    if model_name == 'SVM':
        y_pred = np.argmax(model_SVM.predict_proba(X_test), axis=1)
    else:
        y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred) * 100
    # Precision
    P = metrics.precision_score(y_test, y_pred) * 100
    # Recall
    R = metrics.recall_score(y_test, y_pred) * 100
    # F1 score
    F1 = metrics.f1_score(y_test, y_pred) * 100

    print("Accuracy: %.2f" % acc)
    print("Precision: %.2f" % P)
    print("Recall: %.2f" % R)
    print("F1: %.2f" % F1)
    return

def eval_dl(model_name, model, testloader, device):
    # Validation loss
    test_steps = 0
    total = 0
    correct = 0
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0

    model.eval()

    tp_features = []
    tn_features = []
    fn_features = []
    fp_features = []

    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            X_batch, y_batch = data
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if model_name != 'mlp': X_batch = X_batch.view(X_batch.shape[0], 1, -1)
            # forward
            y_pred = model(X_batch.double())
            test_steps += 1

            y_pred_tag = torch.round(torch.sigmoid(y_pred))

            # TP
            if y_pred_tag == y_batch == 1:
                tp_features.append(X_batch)
            # TN
            elif y_pred_tag == y_batch == 0:
                tn_features.append(X_batch)
            # FN
            elif y_pred_tag != y_batch == 1:
                fn_features.append(X_batch)
            # TN
            elif y_pred_tag != y_batch == 0:
                fp_features.append(X_batch)

            correct += (y_pred_tag == y_batch.unsqueeze(1)).sum().item()
            total += y_batch.size(0)

            tp, fp, fn, tn = get_metrics(y_pred, y_batch.unsqueeze(1))
            test_tp += tp.item()
            test_fp += fp.item()
            test_fn += fn.item()
            test_tn += tn.item()

    precision = test_tp / (test_tp + test_fp)
    recall = test_tp / (test_tp + test_fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("Accuracy: %.2f" % (correct / total * 100))
    print("Precision: %.2f" % (precision * 100))
    print("Recall: %.2f" % (recall * 100))
    print("F1: %.2f" % (f1 * 100))
    return

if __name__ == '__main__':
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Load Data
    data_path = "data/test.csv"
    dataset, features_names, X_test, y_test = load_data(data_path)
    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)

    # LOAD MLP MODEL
    path = 'weights/MLP'
    checkpoint = "checkpoint_000999"
    with open(os.path.join(path, 'params.json')) as config_file:
        config = json.load(config_file)

    h_sizes = [70]  # Input features
    hiddens = config['hiddens']
    start = config['start_hidden_exp']
    max_exp = int(hiddens / 2) + start
    for i in range(hiddens):
        if i <= int(hiddens / 2):
            h_sizes.append(2 ** (start + i))
        else:
            h_sizes.append(2 ** ((max_exp) - (i - int(hiddens / 2))))

    # Set model from config
    model_mlp = MLP(h_sizes)
    model_mlp.to(device)

    # Checkpoint
    checkpoint_dir = os.path.join(path, checkpoint)
    model_state, _ = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
    model_mlp.load_state_dict(model_state)

    # LOAD CNN MODEL
    path = 'weights/CNN'
    with open(os.path.join(path, 'params.json')) as config_file:
        config = json.load(config_file)

    model_cnn = CNNBinaryClassification(input_features=70, ch1=config['ch1'], ch2=config['ch2'],
                                    fc1=config['fc1'], fc2=config['fc2'], fc3=config['fc3'])
    model_cnn.to(device)

    # Checkpoint
    model_state = torch.load('weights/CNN/weights.pt')
    # Small changes
    model_cnn.load_state_dict(model_state)

    # LOAD CNN-SKIP MODEL
    path = 'weights/CNN_SKIP'
    with open(os.path.join(path, 'params.json')) as config_file:
        config = json.load(config_file)

    model_cnn_skip = CNNBinaryClassificationSkip(input_features=70, ch1=config['ch1'], ch2=config['ch2'],
                                        fc1=config['fc1'], fc2=config['fc2'], fc3=config['fc3'],
                                        sk1=config['sk1'], sk2=config['sk2'])
    model_cnn_skip.to(device)
    # Checkpoint
    model_state = torch.load('weights/CNN_SKIP/weights.pt')
    model_cnn_skip.load_state_dict(model_state)

    # LOAD LR MODEL
    filename = 'weights/Logistic.h5'
    model_LR = pickle.load(open(filename, 'rb'))
    # LOAD DT MODEL
    filename = 'weights/DecisionTree.h5'
    model_DT = pickle.load(open(filename, 'rb'))
    # LOAD RF MODEL
    filename = 'weights/RF.h5'
    model_RF = pickle.load(open(filename, 'rb'))
    # LOAD XGB MODEL
    filename = 'weights/XGB.h5'
    model_XGB = pickle.load(open(filename, 'rb'))
    # LOAD SVM MODEL
    filename = 'weights/SVM.h5'
    model_SVM = pickle.load(open(filename, 'rb'))

    # Evaluate ML models
    print('LR')
    eval_ml('LR',model_LR, X_test, y_test)
    print('-----------------------------')
    print('DT')
    eval_ml('DT',model_DT, X_test, y_test)
    print('-----------------------------')
    print('RF')
    eval_ml('RF',model_RF, X_test, y_test)
    print('-----------------------------')
    print('XGB')
    eval_ml('XGB',model_XGB, X_test, y_test)
    print('-----------------------------')
    print('SVM')
    eval_ml('SVM',model_SVM, X_test, y_test)

    # Evaluate DL models
    print('-----------------------------')
    print('MLP')
    eval_dl('mlp',model_mlp, testloader, device)
    print('-----------------------------')
    print('CNN')
    eval_dl('cnn',model_cnn, testloader, device)
    print('-----------------------------')
    print('CNN-skip')
    eval_dl('cnn_skip',model_cnn_skip, testloader, device)

