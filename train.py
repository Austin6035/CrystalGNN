import argparse
import os
import sys

import torch.optim as optim
import csv
from torch.optim.lr_scheduler import MultiStepLR
from models.utils import *
from data.data import CIFData
from models.model import Net

# parser
parser = argparse.ArgumentParser(description='Graph Neural Networks')
parser.add_argument('data_src', metavar='PATH', help='data source: data/data_src')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression',
                    help='complete a regression or ''classification task (default: regression)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--pooling', choices=['mean', 'max', 'add'],
                    default='max', help='global pooling layer (default: mean)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: ''0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: ''[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-space', '-p', default=1, type=int,
                    metavar='N', help='print space (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.8, type=float, metavar='N',
                         help='number of training data to be loaded (default 0.6)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--valid-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default '
                              '0.2)')
valid_group.add_argument('--valid-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.2)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim-method', default='Adam', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=4, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--max-num-nbr', default=12, type=int, metavar='N',
                    help='max number of neighbors')
parser.add_argument('--n-classes', default=2, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--patience', default=20, type=int, metavar='N',
                    help='How long to wait after last time validation loss improved.(default=7)')
attention_group = parser.add_mutually_exclusive_group()
attention_group.add_argument('--attention', '-GAT', action='store_true',
                             help='Attention or not.(default: False)')
attention_group.add_argument('--dynamic-attention', '-DA', action='store_true',
                             help='Dynamic attention or not.(default: False)')
parser.add_argument('--n-heads', default=1, type=int, metavar='N',
                    help='Number of multi-head-attentions.(default=1, useful on attention mechanism)')
parser.add_argument('--dropout-p', '-d', default=0, type=float, metavar='N',
                    help='dropout - p.(default=0)')
parser.add_argument('--early-stopping', '-es', action='store_true',
                    help='if early stopping or not (default: False)')
args = parser.parse_args(sys.argv[1:])
best_loss = 1e10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_loss
    path = "dataset/" + args.data_src
    dataset = CIFData(root_dir=path, max_num_nbr=args.max_num_nbr)

    train_loader, valid_loader, test_loader = \
        train_val_test_split(dataset,
                             batch_size=args.batch_size,
                             train_ratio=args.train_ratio,
                             valid_ratio=args.valid_ratio,
                             test_ratio=args.test_ratio,
                             num_workers=args.workers,
                             train_size=args.train_size,
                             valid_size=args.valid_size,
                             test_size=args.test_size)

    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        target_list = [dataset[i].y for i in range(len(dataset))]
        normalizer = Normalizer(torch.tensor(target_list))

    model = Net(atom_fea_len=args.atom_fea_len,
                n_conv=args.n_conv,
                h_fea_len=args.h_fea_len, n_h=args.n_h,
                classification=True if args.task == 'classification' else False,
                n_classes=args.n_classes,
                attention=args.attention,
                dynamic_attention=args.dynamic_attention,
                n_heads=args.n_heads,
                max_num_nbr=args.max_num_nbr,
                pooling=args.pooling,
                p=args.dropout_p)
    model.to(device)

    if args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    if args.optim_method == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim_method == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as optim-method')

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    train_losses = []
    valid_losses = []

    if args.resume:
        checkpoint_path = 'weights/' + args.resume
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.start_epoch, args.epochs):
        print("----------Train Set----------")
        train_loss = train(train_loader, model, criterion, optimizer, epoch, normalizer)
        train_losses.append(train_loss)

        print("----------Valid Set----------")
        valid_loss = validate(valid_loader, model, criterion, epoch, normalizer)
        valid_losses.append(valid_loss)

        scheduler.step()

        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

        if args.early_stopping:
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    print("----------Test Set----------")
    best_checkpoint = torch.load('weights/model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    test_loss = test(test_loader, model, criterion, normalizer, path="test")

    test(train_loader, model, criterion, normalizer, path="train")
    test(valid_loader, model, criterion, normalizer, path="valid")

    # save loss
    with open('results/loss.csv', 'w') as f:
        writer = csv.writer(f)
        for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
            writer.writerow((epoch, train_loss, valid_loss))
    df = pd.read_csv('results/loss.csv',
                     header=None,
                     names=['EPOCH', 'Train_Loss', 'Valid_Loss'])
    df.to_csv('results/loss.csv', index=False)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    running_loss = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        # auc, precision, recall, fscore

    for batch_idx, data in enumerate(train_loader, 0):
        if args.task == 'regression':
            targets = data.y.unsqueeze(1)
            targets_normed = normalizer.norm(targets)
        else:
            targets = data.y.long()
            targets_normed = targets

        data, targets_normed = data.to(device), targets_normed.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets_normed)

        running_loss.update(loss.item(), targets.size(0))

        if args.task == 'regression':
            mae = mae_metric(normalizer.denorm(outputs.data.cpu()), targets)
            mae_errors.update(mae, targets.size(0))
            if batch_idx % args.print_space == 0:
                print('epoch: %2d, batch_idx: %2d, loss: %.3f, MAE: %.3f' % (
                    epoch + 1, batch_idx + 1, running_loss.avg, mae_errors.avg))
        else:
            accuracy = class_metric(outputs, targets)
            accuracies.update(accuracy, targets.size(0))
            if batch_idx % args.print_space == 0:
                print('epoch: %2d, batch_idx: %2d, loss: %.3f, accuracy: %.3f' % (
                    epoch + 1, batch_idx + 1, running_loss.avg, accuracies.avg))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss.avg


def validate(valid_loader, model, criterion, epoch, normalizer):
    running_loss = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
    for batch_idx, data in enumerate(valid_loader, 0):
        with torch.no_grad():
            if args.task == 'regression':
                targets = data.y.unsqueeze(1)
                targets_normed = normalizer.norm(targets)
            else:
                targets = data.y.long()
                targets_normed = targets
            data, targets_normed = data.to(device), targets_normed.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets_normed)
            running_loss.update(loss.item(), targets.size(0))

            if args.task == 'regression':
                mae = mae_metric(normalizer.denorm(outputs.data.cpu()), targets)
                mae_errors.update(mae, targets.size(0))
                if batch_idx % args.print_space == 0:
                    print('epoch: %2d, batch_idx: %2d, loss: %.3f, MAE: %.3f' % (
                        epoch + 1, batch_idx + 1, running_loss.avg, mae_errors.avg))
            else:
                accuracy = class_metric(outputs, targets)
                accuracies.update(accuracy, targets.size(0))
                if batch_idx % args.print_space == 0:
                    print('epoch: %2d, batch_idx: %2d, loss: %.3f, accuracy: %.3f' % (
                        epoch + 1, batch_idx + 1, running_loss.avg, accuracies.avg))

    return running_loss.avg


def test(test_loader, model, criterion, normalizer, path="test"):
    test_cif_ids = []
    test_targets = []
    test_preds = []
    if args.task == 'classification':
        probabilities = []

    running_loss = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
    for batch_idx, data in enumerate(test_loader, 0):
        with torch.no_grad():
            if args.task == 'regression':
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
            if args.task == 'regression':
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

            if args.task == 'regression':
                mae = mae_metric(normalizer.denorm(outputs.data.cpu()), targets)
                mae_errors.update(mae, targets.size(0))
                if path == 'test' and batch_idx % args.print_space == 0:
                    print('batch_idx: %2d, loss: %.3f, MAE: %.3f' % (
                        batch_idx + 1, running_loss.avg, mae_errors.avg))
            else:
                accuracy = class_metric(outputs, targets)
                accuracies.update(accuracy, targets.size(0))
                if path == 'test' and batch_idx % args.print_space == 0:
                    print('batch_idx: %2d, loss: %.3f, accuracy: %.3f' % (
                        batch_idx + 1, running_loss.avg, accuracies.avg))

    if args.task == 'regression':
        with open('results/regression/' + path + '_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, pred, target in zip(test_cif_ids, test_preds, test_targets):
                writer.writerow((cif_id, pred, target))

        df = pd.read_csv('results/regression/' + path + '_results.csv',
                         header=None, names=['CIF_ID', 'Prediction', 'Target'])
        df.to_csv('results/regression/' + path + '_results.csv', index=False)
    else:
        with open('results/classification/' + path + '_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, pred, target, probability in zip(test_cif_ids, test_preds, test_targets, probabilities):
                writer.writerow((cif_id, pred, target, probability))

        df = pd.read_csv('results/classification/' + path + '_results.csv',
                         header=None,
                         names=['CIF_ID', 'Prediction', 'Target', 'Probabilities'])
        df.to_csv('results/classification/' + path + '_results.csv', index=False)

    return running_loss.avg


if __name__ == '__main__':
    main()
