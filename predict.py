import argparse
import csv
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader

from data.data import CIFData
from models.model import Net
from models.utils import Normalizer, AverageMeter, mae_metric, class_metric

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('trained_model', help='name of the trained model (weights/trained_model).')
parser.add_argument('dataset_name', help='dataset name (dataset/dataset_name).')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

args = parser.parse_args(sys.argv[1:])
model_path = 'weights/' + args.trained_model
if os.path.isfile(model_path):
    print("=> loading model params '{}'".format(model_path))
    model_checkpoint = torch.load(model_path,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(model_path))
else:
    print("=> no model params found at '{}'".format(model_path))

best_loss = 1e10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    global args, model_args, best_loss, model_path

    # load data
    path = "dataset/" + args.dataset_name
    dataset = CIFData(root_dir=path, max_num_nbr=model_args.max_num_nbr)
    # dataset = CIFData(root_dir=path)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    # build model
    orig_atom_fea_len = dataset[0].orig_atom_fea_len
    model = Net(atom_fea_len=model_args.atom_fea_len,
                n_conv=model_args.n_conv,
                h_fea_len=model_args.h_fea_len, n_h=model_args.n_h,
                classification=True if model_args.task == 'classification' else False,
                n_classes=model_args.n_classes,
                attention=model_args.attention,
                dynamic_attention=model_args.dynamic_attention,
                n_heads=model_args.n_heads,
                max_num_nbr=model_args.max_num_nbr,
                pooling=model_args.pooling,
                p=model_args.dropout_p)
    model.to(device)

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    if model_args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        target_list = [dataset[i].y for i in range(len(dataset))]
        normalizer = Normalizer(torch.tensor(target_list))

    # optionally resume from a checkpoint
    if os.path.isfile(model_path):
        print("=> loading model '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(model_path, checkpoint['epoch'],
                      checkpoint['best_loss']))
    else:
        print("=> no model found at '{}'".format(model_path))
    test(test_loader, model, criterion, normalizer)


def test(test_loader, model, criterion, normalizer):
    test_cif_ids = []
    test_targets = []
    test_preds = []
    if model_args.task == 'classification':
        probabilities = []

    running_loss = AverageMeter()
    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
    for batch_idx, data in enumerate(test_loader, 0):
        with torch.no_grad():
            if model_args.task == 'regression':
                targets = data.y.unsqueeze(1)
                targets_normed = normalizer.norm(targets)
            else:
                targets = data.y.long()
                targets_normed = targets
            data, targets_normed = data.to(device), targets_normed.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets_normed)
            running_loss.update(loss.item(), targets.size(0))

            cif_id = data.cif_id
            test_target = targets

            test_cif_ids += cif_id
            if model_args.task == 'regression':
                test_pred = normalizer.denorm(outputs.data.cpu())
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
            else:
                probability = nn.functional.softmax(outputs, dim=1)
                probability = probability.tolist()

                prediction = outputs.cpu().detach().numpy()
                test_pred = np.argmax(prediction, axis=1)
                test_preds += test_pred.tolist()

                test_targets += test_target.view(-1).tolist()
                probabilities += probability

            if model_args.task == 'regression':
                mae = mae_metric(normalizer.denorm(outputs.data.cpu()), targets)
                mae_errors.update(mae, targets.size(0))
                if batch_idx % model_args.print_space == 0:
                    print('batch_idx: %2d, loss: %.3f, MAE: %.3f' % (
                        batch_idx + 1, running_loss.avg, mae_errors.avg))
            else:
                accuracy = class_metric(outputs, targets)
                accuracies.update(accuracy, targets.size(0))
                if batch_idx % model_args.print_space == 0:
                    print('batch_idx: %2d, loss: %.3f, accuracy: %.3f' % (
                        batch_idx + 1, running_loss.avg, accuracies.avg))

    if model_args.task == 'regression':
        with open('results/regression/test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, pred, target in zip(test_cif_ids, test_preds, test_targets):
                writer.writerow((cif_id, round(pred, 2), target))

        df = pd.read_csv('results/regression/test_results.csv',
                         header=None,
                         names=['CIF_ID', 'Prediction', 'Target'])
        df.to_csv('results/regression/test_results.csv', index=False)
    else:
        with open('results/classification/test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, pred, target, probability in zip(test_cif_ids, test_preds, test_targets, probabilities):
                writer.writerow((cif_id, round(pred, 2), target, probability))

        df = pd.read_csv('results/classification/test_results.csv',
                         header=None,
                         names=['CIF_ID', 'Prediction', 'Target', 'Probabilities'])
        df.to_csv('results/classification/test_results.csv', index=False)

    return running_loss.avg


if __name__ == '__main__':
    main()
