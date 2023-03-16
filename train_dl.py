from functools import partial
import numpy as np
import pandas as pd
import math
import os
import torch
import torch.optim as optim
from torch.utils.data import random_split

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.ax import AxSearch

from datasets.base_dataset import  DatasetBase
from models.cnn_model import CNNBinaryClassification, CNNBinaryClassificationSkip
from models.base_model import MLP
from torch.nn import BCEWithLogitsLoss
from utils.FocalLoss import FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from warmup_scheduler import GradualWarmupScheduler

# Random Seeds
torch.random.manual_seed(42)
np.random.seed(42)

def get_metrics(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    tp = y_pred_tag[y_test == 1].sum().float()
    fp = y_pred_tag[y_test == 0].sum().float()

    tot_p = (y_test == 1).sum()
    tot_n = (y_test == 0).sum()

    fn = tot_p - tp
    tn = tot_n - fp

    return tp, fp, fn, tn


def train_with_config(architecture, max_num_epochs, model, criterion, optimizer, scheduler, trainloader, valloader, device):

    for epoch in range(max_num_epochs):
        running_loss = 0.0
        epoch_steps = 0


        model.train()
        for i, data in enumerate(trainloader, 0):
            X_batch, y_batch = data
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if (architecture == 'cnn') or (architecture=='cnn-skip'): X_batch = X_batch.view(X_batch.shape[0], 1, -1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = model(X_batch.double())
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0
        val_tn = 0

        model.eval()
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                X_batch, y_batch = data
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if (architecture == 'cnn') or (architecture=='cnn-skip'): X_batch = X_batch.view(X_batch.shape[0], 1, -1)
                # forward
                y_pred = model(X_batch.double())
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                val_loss += loss.cpu().numpy()
                val_steps += 1

                y_pred_tag = torch.round(torch.sigmoid(y_pred))
                correct += (y_pred_tag == y_batch.unsqueeze(1)).sum().item()
                total += y_batch.size(0)

                tp, fp, fn, tn = get_metrics(y_pred, y_batch.unsqueeze(1))
                val_tp += tp.item()
                val_fp += fp.item()
                val_fn += fn.item()
                val_tn += tn.item()

        precision = val_tp / (val_tp + val_fp)
        recall = val_tp / (val_tp + val_fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        if scheduler is not None:
            scheduler.step(epoch)

        # Checkpoint every 100 epochs
        if epoch % 100 == 99:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total, precision=precision,
                    recall=recall, f1=f1)


def train(architecture, config, dataset, max_num_epochs, checkpoint_dir=None, data_dir=None):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    if architecture == 'mlp':
        h_sizes = [70] # Input features
        hiddens = config['hiddens']
        start = config['start_hidden_exp']
        max_exp = int(hiddens/2) + start
        for i in range(hiddens):
            if i <= int(hiddens/2):
                h_sizes.append(2**(start+i))
            else:
                h_sizes.append(2**((max_exp) - (i - int(hiddens/2))))

        # Set model from config
        model = MLP(h_sizes)
    elif architecture=='cnn':
        # Set model from config
        model = CNNBinaryClassification(input_features=70, ch1=config['ch1'], ch2=config['ch2'],
                                        fc1=config['fc1'], fc2=config['fc2'], fc3=config['fc3'])
    elif architecture == 'cnn-skip':
        model = CNNBinaryClassificationSkip(input_features=70, ch1=config['ch1'], ch2=config['ch2'],
                                            fc1=config['fc1'], fc2=config['fc2'], fc3=config['fc3'],
                                            sk2=config['sk2'], sk1=config['sk1'])

    model.to(device)

    # Set Criterion : BCE or Focal Loss
    assert config['criterion'] == "BCEWithLogitsLoss" or config['criterion'] == "FocalLoss"

    if config['criterion'] == "BCEWithLogitsLoss":
        pos_weight = torch.DoubleTensor([config['pos_weight']]).to(device)
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    elif config['criterion'] == "FocalLoss":
        pos_weight = torch.DoubleTensor([config['pos_weight']]).to(device)
        gamma = torch.DoubleTensor([config['gamma']]).to(device)
        criterion = FocalLoss(gamma=gamma, alpha=pos_weight, logits=True, reduce=True)

    # Set Optimizer : SGD or Adam
    assert config['optimizer'] == "SGD" or config['optimizer'] == "Adam"

    if config['optimizer'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])

    # Set Scheduler
    assert config['scheduler'] == 'Step' or config['scheduler'] == 'Cosine'

    if config['scheduler'] == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, max_num_epochs - config['warmup_steps'])
    if config['scheduler'] == 'Step':
        step_size = math.ceil((max_num_epochs - config['warmup_steps']) / 3)
        scheduler = StepLR(optimizer, step_size)

    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=config['warmup_steps'],
                                              after_scheduler=scheduler)

    # Checkpoint
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Create train-val
    test_abs = int(len(dataset) * 0.8)
    train_subset, val_subset = random_split(
        dataset, [test_abs, len(dataset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    train_with_config(max_num_epochs, model, criterion, optimizer, scheduler_warmup, trainloader, valloader, device)

    print("Finished Training")


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
    dataset = DatasetBase(annovar_train, labels='gt')

    return dataset


if __name__=='__main__':

    max_num_epochs = 1000
    grace_period = 100
    num_samples = 100
    reduction_factor = 4
    architecture = 'cnn' # cnn or mlp or cnn-skip

    if architecture == 'mlp':
        config = {
            "hiddens" : tune.choice([1, 3, 5, 7]),
            "start_hidden_exp" : tune.choice([4, 5, 6, 7]),
            "criterion": "FocalLoss",
            "pos_weight": tune.quniform(1, 3, 0.05),
            "gamma": tune.quniform(1, 4, 0.1), # Only for Focal loss,
            "optimizer": "Adam",
            "lr": tune.uniform(1e-8, 1e-6),
            "weight_decay": tune.uniform(0, 0.1),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "scheduler": 'Cosine',
            "warmup_steps": tune.quniform(0, 100, 10)
        }
    elif architecture == 'cnn':
        config = {
            "ch1": tune.choice([8, 16]),
            "ch2": tune.choice([32, 64]),
            "fc1": tune.choice([512, 256]),
            "fc2": tune.choice([256, 128, 64]),
            "fc3": tune.choice([64, 32]),
            # "sk2": tune.choice([4, 8, 16]),
            # "sk1": tune.choice([32, 64]),
            "criterion": "FocalLoss",
            "pos_weight": tune.quniform(1, 3, 0.05),
            "gamma": tune.quniform(1, 4, 0.1),  # Only for Focal loss,
            "optimizer": "Adam",
            "lr": tune.uniform(1e-8, 1e-6),
            "weight_decay": tune.uniform(0, 0.1),
            "batch_size": tune.choice([32, 64, 128, 256]),
            "scheduler": 'Cosine',
            "warmup_steps": tune.quniform(0, 100, 10)
        }
    elif architecture == 'cnn-skip':
        config = {
            "ch1": tune.choice([8, 16, 32, 128]),
            "ch2": tune.choice([16, 32, 128, 256]),
            "fc1": tune.choice([2048, 1024, 512, 256, 128]),
            "fc2": tune.choice([1024, 512, 256, 128, 64]),
            "fc3": tune.choice([512, 256, 128, 64, 32]),
            "sk2": tune.choice([8, 16, 32]),
            "sk1": tune.choice([32, 64, 128]),
            "criterion": tune.choice(["BCEWithLogitsLoss", "FocalLoss"]),
            "pos_weight": tune.quniform(1, 3, 0.05),
            "gamma": tune.quniform(1, 4, 0.1),  # Only for Focal loss,
            "optimizer": tune.choice(["SGD", "Adam"]),
            "lr": tune.uniform(1e-9, 1e-6),
            "momentum": tune.uniform(0.9, 0.99),  # Only for SGD
            "weight_decay": tune.uniform(0, 0.1),
            "batch_size": tune.choice([8, 16, 32, 64, 128, 256]),
            "scheduler": tune.choice(['Step', 'Cosine']),
            "warmup_steps": tune.quniform(0, 100, 10)
        }

    # Load Data
    data_path = "data/train.csv"
    dataset = load_data(data_path)

    # Hyperparams search scheduler
    # A training result attr to use for comparing time. -> training_iteration is DEFAULT
    scheduler = ASHAScheduler(
        max_t=max_num_epochs, # max time units per trial.
        grace_period=grace_period, # only stop trials at least this old in time.
        reduction_factor=reduction_factor) # Used to set halving rate and amount.

    # Search algorithm
    ax_search = AxSearch(
        # Optional: parameter_constraints
        # Optional: outcome_constraints
    )

    # Limit to 4 concurrent trials
    algo = tune.suggest.ConcurrencyLimiter(ax_search, max_concurrent=4)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration", "precision", "recall", "f1"])

    result = tune.run(
        partial(train, dataset=dataset, max_num_epochs=max_num_epochs),
        name=architecture+"_e" + str(max_num_epochs) + "_g" + str(grace_period) + "_n" + str(num_samples) + "_rf" +
             str(reduction_factor),
        config=config,
        resources_per_trial={"cpu": 8, "gpu": 1},
        num_samples=num_samples, # Number of tests
        metric="f1", # Already in Scheduler
        mode="max", # Already in Scheduler
        scheduler=scheduler,
        search_alg=algo,
        local_dir="/media/gatvprojects/3C6D631202147A77/deepGen/ray_results",
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("f1", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final validation precision: {}".format(
        best_trial.last_result["precision"]))
    print("Best trial final validation recall: {}".format(
        best_trial.last_result["recall"]))
    print("Best trial final validation f1: {}".format(
        best_trial.last_result["f1"]))