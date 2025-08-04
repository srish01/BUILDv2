import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from PIL import Image
from itertools import chain
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from datetime import datetime
from utils.ood_utils import get_fc_w_b, get_react_logits, dice_calculate_mask, get_scale_logits

# PCA
from sklearn.decomposition import PCA

# confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# tsne
from sklearn.manifold import TSNE
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from custom_imagefolder import ImageFolder


def make_loader(data, args, train='train'):
    if train == 'train':
        return DataLoader(data, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            shuffle=True)
    elif train == 'calibration':
        return DataLoader(data, batch_size=args.cal_batch_size,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_memory,
                        shuffle=True)
    elif train == 'test':
        return DataLoader(data, batch_size=args.test_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            shuffle=False)
    else:
        raise NotImplementedError("'train' must be either train, calibration, or test")

def calibration_dataset(args, dataset, predefine_idx=None):
    if isinstance(dataset, Subset):
        cal_dataset = deepcopy(dataset)
        ys = list(sorted(set(cal_dataset.targets)))
        cal_idx_list, keep_idx_list, cal_names_list, keep_names_list = [], [], [], []
        cal_subset_indices, keep_subset_indices = [], []
        cal_names_list, keep_names_list = [], []
        for y_ in ys:
            # idx is the indices of class y in the subset
            idx = np.where(cal_dataset.targets == y_)[0]
            if args.seed != 0:
                np.random.shuffle(idx)

            # Note _dataset.indices are the true location of the targets in dataset.dataset
            # First choose the indices of true (absolute) indices for class y in dataset.dataset
            cal_idx = cal_dataset.indices[idx]
            keep_idx = cal_dataset.indices[idx]

            # Split the true (absolute) indices
            cal_idx = cal_idx[:args.cal_size]
            keep_idx = keep_idx[args.cal_size:]

            cal_subset_indices.append(idx[:args.cal_size])
            keep_subset_indices.append(idx[args.cal_size:])

            # Save the selected true (absolute) indices
            cal_idx_list.append(cal_idx)
            keep_idx_list.append(keep_idx)




        cal_idx_list = np.concatenate(cal_idx_list)
        keep_idx_list = np.concatenate(keep_idx_list)
        cal_subset_indices = np.concatenate(cal_subset_indices)
        keep_subset_indices = np.concatenate(keep_subset_indices)

        dataset.targets = dataset.targets[keep_subset_indices]
        dataset.indices = keep_idx_list
        cal_dataset.targets = cal_dataset.targets[cal_subset_indices]
        cal_dataset.indices = cal_idx_list

        cal_names = []
        for i in cal_subset_indices:
            cal_names.append(cal_dataset.names)
        cal_dataset.names = cal_names
        keep_names = []
        for i in keep_subset_indices:
            keep_names.append(dataset.names)
        dataset.names = keep_names
        return dataset, cal_dataset

    else:
        cal_dataset = deepcopy(dataset)
        ys = list(sorted(set(cal_dataset.targets)))
        cal_idx_list, keep_idx_list, cal_names_list, keep_names_list = [], [], [], []
        for y_ in ys:
            idx = np.where(cal_dataset.targets == y_)[0]
            if args.seed != 0:
                np.random.shuffle(idx)
            cal_idx = idx[:args.cal_size]
            keep_idx = idx[args.cal_size:]
            name = dataset.names[idx[0]]
            for _ in range(len(cal_idx)):
                cal_names_list.append(name)
            for _ in range(len(keep_idx)):
                keep_names_list.append(name)

            cal_idx_list.append(cal_idx)
            keep_idx_list.append(keep_idx)
        cal_idx_list = np.concatenate(cal_idx_list)
        keep_idx_list = np.concatenate(keep_idx_list)

        cal_dataset.data = cal_dataset.data[cal_idx_list]
        cal_dataset.targets = cal_dataset.targets[cal_idx_list]
        cal_dataset.full_labels = cal_dataset.full_labels[cal_idx_list]
        cal_dataset.names = cal_names_list

        dataset.data = dataset.data[keep_idx_list]
        dataset.targets = dataset.targets[keep_idx_list]
        dataset.full_labels = dataset.full_labels[keep_idx_list]
        dataset.names = keep_names_list
        return dataset, cal_dataset

class Criterion(nn.Module):
    def __init__(self, args, net, reduction='mean'):
        super(Criterion, self).__init__()
        self.args = args
        if args.loss_f == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif args.loss_f == 'bce':
            self.criterion = nn.BCELoss(reduction=reduction)
        elif args.loss_f == 'nll':
            self.criterion = nn.NLLLoss(reduction=reduction)
        else:
            NotImplementedError("Loss {} is not defined".format(args.loss_f))

        # self.seen_classes = net.seen_classes

    def forward(self, x, labels):
        labels = self.convert_lab(labels)
        if self.args.loss_f == 'bce':
            return self.criterion(torch.sigmoid(x), labels)
        elif self.args.loss_f == 'nll':
            return self.criterion(nn.LogSoftmax(dim=1)(x), labels)
        else: # 'ce'
            return self.criterion(x, labels)

    def convert_lab(self, labels):
        if self.args.loss_f == 'bce':
            raise NotImplementedError("BCE is not implemented")
            n_cls = len(self.seen_classes)
            labels = torch.eye(n_cls).to(self.args.device)[labels]
            return labels
        else: # 'ce', 'nll'
            return labels

class Zeroshot:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.correct, self.total, self.total_loss = 0., 0., 0.

    def evaluate(self, x, text_inputs, labels):
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs).type(torch.FloatTensor).to(self.args.device)

            image_features = x / x.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred = similarity.argmax(-1)

        self.correct += pred.eq(labels).sum().item()
        self.total += len(labels)

    def acc(self, reset=True):
        metrics = {}
        metrics['cil_acc'] = self.correct / self.total * 100
        self.reset_eval()
        return metrics

    def reset_eval(self):
        self.correct, self.total, self.total_loss = 0., 0., 0.

class Logger:
    def __init__(self, args, name=None, filename='result'):
        self.init = datetime.now()
        self.args = args
        if name is None:
            self.name = self.init.strftime("%m|%d|%Y %H|%M|%S")
        else:
            self.name = name
        
        # self.filename = f"/{filename}.txt"

        self.args.dir = self.name

        self._make_dir()

    def now(self):
        time = datetime.now()
        diff = time - self.init
        self.print(time.strftime("day-%d-%m-%Y_hr-%H-%M-%S"), f" | Total: {diff}")

    def print(self, *object, sep=' ', end='\n', flush=False, filename='/result.txt', date=False):
        # if filename is None:
        #     filename = self.filename
        if date:
            date = datetime.now().strftime("day-%d-%m-%Y_hr-%H-%M-%S")
            date = f"{date} --- "
            object = (date, *object)
        print(*object, sep=sep, end=end, file=sys.stdout, flush=flush)

        if self.args.print_filename is not None:
            filename = self.args.print_filename
        with open(self.dir() + filename, 'a') as f:
            print(*object, sep=sep, end=end, file=f, flush=flush)

    def _make_dir(self):
        # If provided hdd drive
        if 'hdd' in self.name:
            if not os.path.isdir('/' + self.name):
                os.makedirs('/' + self.name)
        else:
            if not os.path.isdir('./logs'):
                os.mkdir('./logs')
            if not os.path.isdir('./logs/{}'.format(self.name)):
                os.makedirs('./logs/{}'.format(self.name))

    def dir(self):
        if 'hdd' in self.name:
            return '/' + self.name + '/'
        else:
            return './logs/{}/'.format(self.name)

    def time_interval(self):
        self.print("Total time spent: {}".format(datetime.now() - self.init))

def _print_results(task_id, mat, print=print, type='acc'):
    if type == 'acc':
        # Print accuracy
        print(f'> Accuracy', date=False)
        for i in range(task_id + 1):
            print(f"evaluating model {i} -> [", end='', date=False)
            for j in range(task_id + 1):
                acc = mat[i, j]
                if acc != -100:
                    print("{:.2f}   /".format(acc), end='', date=False)
                else:
                    print("\t", end='', date=False)
            print("]    ", end='', date=False)
            aca = mat[i, -1]
            print(f"ACA(t= {i})={aca:.2f}", date=False)
        aia = mat[-1, -1]
        print(f"> AIA: {aia:.2f}", date=False)
        return aca, aia
    # elif type == 'det_acc':
    #     print(f'> Task id Prediction Accuracy', date=False)
    #     for i in range(task_id + 1):
    #         print(f"evaluating model {i} -> [", end='', date=False)
    #         for j in range(task_id + 1):
    #             acc = mat[i, j]
    #             if acc != -100:
    #                 print("{:.2f}   /".format(acc), end='', date=False)
    #             else:
    #                 print("\t", end='', date=False)
    #         print("]    ", end='', date=False)
    #         print(f"ADA(t= {i})={mat[i, -1]:.2f}", date=False)  # Average Detection Accuracy
    #     print(f"> AIDA: {mat[-1, -1]:.2f}", date=False) # Average Incremental Detection Accuracy
    elif type == 'forget':
        print('> Forgetting:', date=False)
        print("[", end='', date=False)
        # Print forgetting and average incremental accuracy
        for i in range(task_id + 1):
            forgetting_i = mat[-1, i]
            if forgetting_i != -100:
                print(f"{forgetting_i:.2f}  /", end='', date=False)
            else:
                print("\t", end='', date=False)
        print("]", end='', date=False)
        avg_forget = None
        if task_id > 0:
            avg_forget = np.mean(mat[-1, :task_id])
            print(f"Avg Forgetting: {avg_forget:.2f}", date=False)
        return avg_forget

    else:
        raise NotImplementedError("Type must be either 'acc', 'forget' or 'det_acc'")



class Tracker:
    def __init__(self, args):
        self.print = args.logger.print
        self.mat = np.zeros((args.n_tasks * 2 + 1, args.n_tasks * 2 + 1)) - 100

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(self.mat[task_id, :p_task_id + 1])

        # Compute forgetting
        for i in range(task_id):
            self.mat[-1, i] = self.mat[i, i] - self.mat[task_id, i]

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if self.print:
            print = self.print
        return _print_results(task_id=task_id, mat=self.mat, print=print, type=type)

        # if print is None: print = self.print
        # if type == 'acc':
        #     # Print accuracy
        #     print(f'> Accuracy', date=False)
        #     for i in range(task_id + 1):
        #         print(f"Trained on task {task_id} -> [", end='', date=False)
        #         for j in range(task_id + 1):
        #             acc = self.mat[i, j]
        #             if acc != -100:
        #                 print("{:.2f}   /".format(acc), end='', date=False)
        #             else:
        #                 print("\t", end='', date=False)
        #         print("]    ", end='', date=False)
        #         print(f"ACA(t={task_id})={self.mat[i, -1]:.2f}", date=False)
        #     print(f"> AIA: {self.mat[-1, -1]:.2f}", date=False)
        # elif type == 'forget':
        #     print('> Forgetting:', date=False)
        #     print("[", end='', date=False)
        #     # Print forgetting and average incremental accuracy
        #     for i in range(task_id + 1):
        #         forgetting_i = self.mat[-1, i]
        #         if forgetting_i != -100:
        #             print(f"{forgetting_i:.2f}  /", end='', date=False)
        #         else:
        #             print("\t", end='', date=False)
        #         print("]", end='', date=False)
        #     if task_id > 0:
        #         forget = np.mean(self.mat[-1, :task_id])
        #         print(f"Avg Forgetting: {forget:.2f}", date=False)
        # else:
        #     raise NotImplementedError("Type must be either 'acc' or 'forget'")

class AUCTracker:
    def __init__(self, args):
        self.print = args.logger.print
        self.mat = np.zeros((args.n_tasks * 2 + 1, args.n_tasks * 2 + 1)) - 100
        self.n_tasks = args.n_tasks

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(np.concatenate([
                                                        self.mat[task_id, :task_id],
                                                        self.mat[task_id, task_id + 1:self.n_tasks]
                                                        ]))

        # # Compute forgetting
        # for i in range(task_id):
        #     self.mat[-1, i] = self.mat[i, i] - self.mat[task_id, i]

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            # print(f'> AUC/AUPR', date=False)
            for i in range(task_id + 1):
                print(f"evaluating model {i} -> [", end='', date=False)
                for j in range(self.n_tasks):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("]    ", end='', date=False)
                print(f"Avg (t= {i})={self.mat[i, -1]:.2f}", date=False)
                # print("{:.2f}".format(self.mat[i, -1]))
            # Print forgetting and average incremental accuracy
            # for i in range(self.n_tasks):
            #     print("\t", end='')
            print(f"> Inc. : {self.mat[-1, -1]:.2f}", date=False)
            # print("{:.2f}".format(self.mat[-1, -1]))
        else:
            raise NotImplementedError("Type must be 'acc'")

class OWTracker:
    def __init__(self, args):
        self.print = args.logger.print
        self.mat = np.zeros((args.n_tasks * 2 + 1, args.n_tasks * 2 + 1)) - 100
        self.n_tasks = args.n_tasks

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(self.mat[task_id, task_id + 1:self.n_tasks])

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='auc', print=None):
        if print is None: print = self.print
        ow_params = ['auc', 'aupr']
        if type in ow_params:
            # Print accuracy
            print(f'> {type.upper()}', date=False)
            for i in range(task_id + 1):
                print(f"evaluating model {i} -> [", end='', date=False)
                for j in range(self.n_tasks):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("]    ", end='', date=False)
                if self.mat[i, -1] != -100:
                    print(f"Avg {type.upper()}(t= {i})={self.mat[i, -1]:.2f}", date=False)
                else:
                    print("")
            # Print forgetting and average incremental accuracy
            # for i in range(self.n_tasks):
            #     print("\t", end='')
            # print("{:.2f}".format(self.mat[-1, -1]))
            print(f"> Inc. {type.upper()}: {self.mat[-1, -1]:.2f}", date=False)
            return self.mat[-1, -1]
            # print("{:.2f}".format(self.mat[-1, -1]))
        else:
            raise NotImplementedError("Type must be 'auc'")

    # def print_result(self, task_id, type='acc', print=print):
    #     if type == 'acc':
    #         # Print accuracy
    #         for i in range(task_id + 1):
    #             for j in range(self.n_tasks):
    #                 acc = self.mat[i, j]
    #                 if acc != -100:
    #                     print("{:.2f}\t".format(acc), end='')
    #                 else:
    #                     print("\t", end='')
    #             if self.mat[i, -1] != -100:
    #                 print("{:.2f}".format(self.mat[i, -1]))
    #             else:
    #                 print("")
    #         # Print forgetting and average incremental accuracy
    #         for i in range(self.n_tasks):
    #             print("\t", end='')
    #         print("{:.2f}".format(self.mat[-1, -1]))
    #     else:
    #         raise NotImplementedError("Type must be 'acc'")

# def print_result(mat, task_id, type, print=print):
#     if type == 'acc':
#         # Print accuracy
#         for i in range(task_id + 1):
#             for j in range(task_id + 1):
#                 acc = mat[i, j]
#                 if acc != -100:
#                     print("{:.2f}\t".format(acc), end='')
#                 else:
#                     print("\t", end='')
#             print("{:.2f}".format(mat[i, -1]))
#     elif type == 'forget':
#         # Print forgetting and average incremental accuracy
#         for i in range(task_id + 1):
#             acc = mat[-1, i]
#             if acc != -100:
#                 print("{:.2f}\t".format(acc), end='')
#             else:
#                 print("\t", end='')
#         print("{:.2f}".format(mat[-1, -1]))
#         if task_id > 0:
#             forget = np.mean(mat[-1, :task_id])
#             print("Average Forgetting: {:.2f}".format(forget))
#     else:
#         ValueError("Type must be either 'acc' or 'forget'")

def tsne(train_f_cross, train_y_cross, name='tsne',
         n_components=2, verbose=0, learning_rate=1, perplexity=9, n_iter=1000, logger=None):
    """ train_f_cross: X, numpy array. train_y_cross: y, numpy array """
    num_y = len(list(set(train_y_cross)))

    tsne = TSNE(n_components=n_components, verbose=verbose,
                learning_rate=learning_rate, perplexity=perplexity,
                n_iter=n_iter)
    tsne_results = tsne.fit_transform(train_f_cross)

    df_subset = pd.DataFrame(data={'tsne-2d-one': tsne_results[:, 0],
                                    'tsne-2d-two': tsne_results[:, 1]})
    df_subset['y'] = train_y_cross

    plt.figure(figsize=(16,10))
    sn.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sn.color_palette("hls", num_y),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    dir = '' if logger is None else logger.dir()

    plt.savefig(dir + name)
    plt.close()

def plot_confusion(true_lab, pred_lab, label_names=None,
                    task_id=None, p_task_id=None, name='confusion',
                    print=print, logger=None, num_cls_per_task=None):
    classes = sorted(set(np.concatenate([true_lab, pred_lab])))
    if label_names is not None:
        labs = []
        for c in classes:
            labs.append(label_names[c])
    plt.figure(figsize=(15, 14))
    cm = confusion_matrix(true_lab, pred_lab)
    hmap = sn.heatmap(cm, annot=True)
    hmap.set_xticks(np.arange(len(classes)) + 0.5)
    hmap.set_yticks(np.arange(len(classes)) + 0.5)
    if label_names is not None:
        hmap.set_xticklabels(labs, rotation=90)
        hmap.set_yticklabels(labs, rotation=0)

    if num_cls_per_task is not None:
        for j in range(len(classes)):
            if (j + 1) % num_cls_per_task == 0:
                plt.axhline(y=j + 1)
                plt.axvline(x=j + 1)

    dir = '' if logger is None else logger.dir() # if None, save into current folder
    print = logger.print if logger is not None else print

    if task_id is not None:
        plt.savefig(dir + "Total Task {}, current task {} is learned.pdf".format(task_id, p_task_id))
    else:
        plt.savefig(dir + name + '.pdf')
    plt.close()

    if task_id is not None:
        print("{}/{} | upper/lower triangular sum: {}/{}".format(task_id, p_task_id,
                                    np.triu(cm, 1).sum(), np.tril(cm, -1).sum()))
    else:
        print("Upper/lower triangular sum: {}/{}".format(np.triu(cm, 1).sum(),
                                                        np.tril(cm, -1).sum()))

def dist_estimation(data, classes):
    # data, classes: numpy array
    data = data / np.linalg.norm(data, axis=-1, keepdims=True)

    unique_cls = list(sorted(set(classes)))

    mu_list = []
    sigma_list = []
    for i, c in enumerate(unique_cls):
        idx = classes == c
        selected_data = data[idx]

        mu = np.mean(selected_data, axis=0)
        mu_list.append(mu)

        sigma = 0
        selected_data = selected_data - mu
        for s in selected_data:
            s = s.reshape(1, -1)
            sigma += np.transpose(s) @ s
        sigma_list.append(sigma / len(selected_data))
    # sigma /= len(data)
    return mu_list, sigma_list

def maha_distance(inputs, mu, sigma):
    inv_sigma = np.linalg.inv(sigma)
    out = (inputs - mu).dot(inv_sigma)
    # print(out.shape)
    out = np.sum(out * (inputs - mu), axis=1)
    # out = np.dot(out, np.transpose(inputs - mu))
    return out

def md(data, mean, mat, inverse=False):
    if data.ndim == 1:
        data.reshape(1, -1)
    delta = (data - mean)

    if not inverse:
        mat = np.linalg.inv(mat)

    dist = np.dot(np.dot(delta, mat), delta.T)
    return np.sqrt(np.diagonal(dist)).reshape(-1, 1)

from sklearn.metrics import roc_auc_score
def compute_auc(in_scores, out_scores):
    # Return auc e.g. auc=0.95
    if isinstance(in_scores, list):
        in_scores = np.concatenate(in_scores)
    if isinstance(out_scores, list):
        out_scores = np.concatenate(out_scores)

    labels = np.concatenate([np.ones_like(in_scores),
                             np.zeros_like(out_scores)])
    try:
        auc = roc_auc_score(labels, np.concatenate((in_scores, out_scores)))
    except ValueError:
        print("Input contains NaN, infinity or a value too large for dtype('float64').")
        auc = -0.99
    return auc

def compute_aupr_in(in_scores, out_scores):
    if isinstance(in_scores, list):
        in_scores = np.concatenate(in_scores)
    if isinstance(out_scores, list):
        out_scores = np.concatenate(out_scores)

    labels = np.concatenate([np.ones_like(in_scores),
                             np.zeros_like(out_scores)])
    precision_in, recall_in, _ = metrics.precision_recall_curve(labels, np.concatenate((in_scores, out_scores)))
    aupr_in = metrics.auc(recall_in, precision_in)
    
    return aupr_in


def compute_aupr_out(in_scores, out_scores):
    if isinstance(in_scores, list):
        in_scores = np.concatenate(in_scores)
    if isinstance(out_scores, list):
        out_scores = np.concatenate(out_scores)

    labels = np.concatenate([np.zeros_like(in_scores),
                             np.ones_like(out_scores)])
    precision_out, recall_out, _ = metrics.precision_recall_curve(labels, np.concatenate((in_scores, out_scores)))
    aupr_out = metrics.auc(recall_out, precision_out)
    
    return aupr_out

class DeNormalize(object):
    # def __init__(self, mean, std):
    def __init__(self, transform):
        # self.mean = mean
        # self.std = std
        self.mean = transform.transforms[-1].mean # (Tensor)
        self.std = transform.transforms[-1].std # (Tensor)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class MySampler(Sampler):
    """
        Sampler for dataset whose length is different from that of the target dataset.
        This can be particularly useful when we need oversampling/undersampling because
        the target dataset has more/less samples than the dataset of interest.
        Generate indices whose length is same as that of target length.
    """
    def __init__(self, current_length, target_length):
        self.current = current_length
        self.target = target_length

    def __iter__(self):
        self.indices = np.array([], dtype=int)
        while len(self.indices) < self.target:
            idx = np.random.permutation(self.current)
            self.indices = np.concatenate([self.indices, idx])
        self.indices = self.indices[:self.target]
        return iter(self.indices)

    def __len__(self):
        return self.target

class Memory(Dataset):
    """
        Replay buffer. Keep balanced samples. Data must be compatible with Image.
        Currently, MNIST and CIFAR are compatible.
    """
    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size

        self.data_list = {}
        self.targets_list = {}

        self.data, self.targets = [], []

        self.n_cls = len(self.data_list)
        self.n_samples = self.buffer_size

    def update(self, dataset):
        self.args.logger.print("Updating Memory")
        self.transform = dataset.transform

        ys = list(sorted(set(dataset.targets)))
        for y in ys:
            idx = np.where(dataset.targets == y)[0]
            # import pdb
            # pdb.set_trace()
            self.data_list[y] = dataset.data[idx]
            self.targets_list[y] = dataset.targets[idx]

            self.n_cls = len(self.data_list)

        self.n_samples = self.buffer_size // self.n_cls
        for y, data in self.data_list.items():
            idx = np.random.permutation(len(data))
            idx = idx[:self.n_samples]
            self.data_list[y] = self.data_list[y][idx]
            self.targets_list[y] = self.targets_list[y][idx]

        self.data, self.targets = [], []
        for (k, data), (_, targets) in zip(self.data_list.items(), self.targets_list.items()):
            self.data.append(data)
            self.targets.append(targets)
        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)

    def is_empty(self):
        return len(self.data) == 0

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

class Memory_ImageFolder(Dataset):
    """
        Replay buffer. Keep balanced samples. This only works for ImageFolder dataset.
    """
    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size

        self.data_list = {}
        self.targets_list = {}

        self.data, self.targets = [], []

        self.n_cls = len(self.data_list)
        self.n_samples = self.buffer_size

    def is_empty(self):
        return len(self.data) == 0

    def update(self, dataset):
        """
            dataset is a subset. The subset has attributes targets, indices, transform, etc.
            where indices are the absolute indices in the original dataset
            and targets are the targets of the indices in the original dataset.
            This function makes attributes data and targets, where
            data is a list of paths and targets is a list of targets of the corresponding data

            NOTE
            dataset is a Subset
            dataset.dataset is an ImageFolder
        """
        self.args.logger.print("Updating Memory")

        self.loader = dataset.dataset.loader
        self.transform = dataset.dataset.transform

        ys = list(sorted(set(dataset.targets)))
        for y in ys:
            # import pdb
            # pdb.set_trace()
            idx = np.where(dataset.targets == y)[0]
            absolute_idx = dataset.indices[idx]

            if y not in self.data_list.keys():
                self.data_list[y], self.targets_list[y] = [], []

            for i in absolute_idx:
                self.data_list[y].append(dataset.dataset.samples[i][0])
                self.targets_list[y].append(dataset.dataset.samples[i][1])

                self.n_cls = len(self.data_list) # total number of classes in memory

        # number of samples per class
        self.n_samples = self.buffer_size // self.n_cls
        for y, data in self.data_list.items():
            # Choose random samples to keep
            idx = np.random.permutation(len(data))
            idx = idx[:self.n_samples]
            self.data_list[y] = [self.data_list[y][i] for i in idx]
            self.targets_list[y] = [self.targets_list[y][i] for i in idx]

        self.data, self.targets = [], []
        for (k, data), (_, targets) in zip(self.data_list.items(), self.targets_list.items()):
            assert len(data) == len(targets)
            for path, y in zip(data, targets):
                self.data.append(path)
                self.targets.append(y)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.data[index], self.targets[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

def compressor(images, k=10):
    """
        Compress a batch of images using SVD.
        images: tensor shape (B, ch, h, w)
        k: integer, number of leading eigenpairs to keep. Discard d-k pairs.
    """
    b, ch, h, w = images.size()
    assert k <= h

    img_list = []
    for img in images:
        for c in range(ch):
            u, s, v = torch.svd(img[c, :, :])
            u, s, v = u[:, :k], s[:k], v[:, :k]
            img[c] = torch.mm(torch.mm(u, torch.diag(s)), v.t())
        img_list.append(img.view(1, ch, h, w))
    return torch.cat(img_list)

def load_deit_pretrain(args, target_model):
    """
        target_model: the model we want to replace the parameters (most likely un-trained)
    """
    if os.path.isfile('./deit_pretrained/best_checkpoint.pth'):
        checkpoint = torch.load('./deit_pretrained/best_checkpoint.pth', map_location='cpu')
    else:
        raise NotImplementedError("Cannot find pre-trained model")
    target = target_model.state_dict()
    pretrain = checkpoint['model']
    transfer = {k: v for k, v in pretrain.items() if k in target and 'head' not in k}
    target.update(transfer)
    target_model.load_state_dict(target)
    # return args

class ComputeEnt:
    def __init__(self, args):
        self.temp = args.T

    def compute(self, output, keepdim=True):
        """
            output: torch.tensor logit, 2d
        """
        soft = torch.softmax(output, dim=-1)
        if keepdim:
            return -1 * torch.sum(soft * torch.log(soft), dim=-1, keepdim=True)
        else:
            return-1 * torch.sum(soft * torch.log(soft))


def collect_test_scores(args, model, method, train_data, test_data, **kwargs):

    train_loaders = []

    for eval_model in range(args.n_tasks):

        if args.use_finetuned:      # args.use_finetuned when using test scores from thresholds and p-value from validation dataset
            fname = os.path.join(args.load_dir +f'/after_cal/m{eval_model}_test_scores_{method}.npz')
        else:
            fname = os.path.join(args.load_dir + f'/m{eval_model}_test_scores_{method}.npz')

        if os.path.exists(fname):
            args.logger.print(f'=> test scores for {method}: m{eval_model} already available')
            continue
        args.logger.print(f'collecting test scores for {method}: m{eval_model}')

        if method == 'more_fw':
            test_model_name = f'/model_task_{eval_model}'
        elif method == 'more_bw':
            test_model_name = f'/model_backupdate_{eval_model}'
        elif method == 'build':
            test_model_name = f'/buildv2_model_{eval_model}'
        c = 20
        
        t_train = train_data.make_dataset(eval_model)
        train_loaders.append(make_loader(t_train, args, train = 'train'))
        
        if hasattr(model, 'preprocess_task'):
            model.preprocess_task(names=train_data.task_list[eval_model][0],
                                labels=train_data.task_list[eval_model][1],
                                task_id=eval_model,
                                loader=train_loaders[-1])

        model_file = args.load_dir + test_model_name

        if os.path.exists(model_file):
            args.logger.print(f'Load a trained model from: {model_file}')
            state_dict = torch.load(model_file)  
            model.net.load_state_dict(state_dict)
            model.net.eval()
        else:
            raise NotImplementedError(f'{model_file} Load dir incorrect')

        base = {}
        # max_logit = {}
        react = {}
        dice = {}
        scale = {}
        cov, cov_inv, mean = {}, {}, {}

        if args.use_finetuned:
            with open(args.load_dir + f'/m{eval_model}_best_p.json', 'r') as p_file:
                args.logger.print(f'm{eval_model}: loading best p')
                p_dict = json.load(p_file)


        
        args.logger.print(f'###### Testing model: {eval_model}')

        for d in range(args.n_tasks):
            args.logger.print(f'#### using data id: {d}')
            t_test = test_data.make_dataset(d)
            test_loader = make_loader(t_test, args, train = 'test')

            for t in range(eval_model+1):
                # if "derpp" in args.model:
                #     args.logger.print(f'Using non-multi head, skipping head {t}')
                #     continue

                args.logger.print(f'### on head: {t}')

                # if method == 'more_bw':
                #     fc_file = args.load_dir + f'/bw_fc_layer_model_m{t}.npz'
                # else:
                #     fc_file = args.load_dir + f'/fc_layer_model_m{t}.npz'
                

                logits_list, label_list, sm_score_list, sm_pred_list, md_scores_list, smmd_pred_list, md_pred_list, maha_dist_list, activations_list = [], [], [], [], [], [], [], [], []

                base[d, t] = {}
                react[d, t] = {}
                dice[d, t] = {}
                scale[d, t] = {}
                
                # args.logger.print(f'loading fc layer from: {fc_file}')
                # fc_w, fc_b = get_fc_w_b(args, fc_file)
                
                head_name = f"head.{t}" if "more" in args.model else "head"
                fc_w = model.net.state_dict()[f'{head_name}.weight'].detach().cpu().numpy()
                fc_b = model.net.state_dict()[f'{head_name}.bias'].detach().cpu().numpy()
                cov = np.load(args.load_dir + f'/cov_task_{t}.npy')
                cov_inv = np.linalg.inv(cov)
                for y in range(t * args.num_cls_per_task, (t + 1) * args.num_cls_per_task):
                    mean_val = np.load(args.load_dir + f'/mean_label_{y}.npy')
                    mean[y] = mean_val

                for x, y, _, _, _ in test_loader:
                    x, y = x.to(args.device), y.to(args.device)
                    
                    label_list.append(y)
                    
                    with torch.no_grad():
                        model.net.eval()
                        if "derpp" in args.model or "pass" in args.model:
                            features = model.net.forward_features(x)
                            logits = model.net.forward_classifier(features)
                            if "pass" in args.model:
                                logits = logits[:, ::4]
                            # topk_scores = torch.softmax(logits, dim=1)[:, :args.num_cls_per_task]   # TODO: wrong
                            topk_scores = torch.softmax(logits, dim=1)[:, t * args.num_cls_per_task: (t + 1) * args.num_cls_per_task]
                            # this is to treat non-multi head models such as DER++ as multi-head. Slower but 100% compatible at least...
                        else:
                            features, _ = model.net.forward_features(t, x, s=args.smax)
                            logits = model.net.forward_classifier(t, features)
                            # logits = torch.matmul(features, fc_w.T.to(args.device)) + fc_b.to(args.device)
                            topk_scores = torch.softmax(logits, dim=1)[:, :args.num_cls_per_task] # QUESTIONABLE: why top k?
                            # topk_scores = torch.softmax(logits, dim=1)[:, t * args.num_cls_per_task: (t+1) * args.num_cls_per_task]  
                        sm_scores, sm_pred = torch.max(topk_scores, dim=1)

                        md_list, dist = [], 0
                        for y in range(t * args.num_cls_per_task, (t + 1) * args.num_cls_per_task):
                            mean_val = mean[y]
                            dist = md(features.detach().cpu().numpy(), mean_val, cov_inv, inverse=True)
                            # dist1 = [mahalanobis(feat, mean_val, cov_inv) for feat in features.detach().cpu().numpy()]
                            scores_md = c / dist           ### TODO: change c from 20, because dist in our case is much larger making scores_md very small
                            md_list.append(scores_md)
                        maha_dist = np.concatenate(md_list, axis = 1)
                        maha_dist, maha_pred = (torch.from_numpy(maha_dist).to(args.device)).max(1)
                        smmd_scores, smmd_pred = ((topk_scores.T * maha_dist).T).max(1)
                        # _, md_pred = md_scores.max(1)
                    
                    logits_list.append(logits)
                    activations_list.append(features)
                    sm_score_list.append(sm_scores)
                    sm_pred_list.append(sm_pred + args.num_cls_per_task * t)
                    md_scores_list.append(smmd_scores)
                    smmd_pred_list.append(smmd_pred + args.num_cls_per_task * t)
                    maha_dist_list.append(maha_dist)
                    md_pred_list.append(maha_pred + args.num_cls_per_task * t)  
                
                
                args.logger.print(f'--base scores')
                maha_dist_list = torch.cat(maha_dist_list)
                maha_pred_list = torch.cat(md_pred_list)
                logit_score, logit_pred = torch.max(torch.cat(logits_list), dim = 1)
                # maha_dist_list = maha_dist_list.reshape(maha_dist_list.shape[0])
                
                base[d, t]['logits'] = torch.cat(logits_list).detach().cpu().numpy()
                base[d, t]['gt'] = torch.cat(label_list).detach().cpu().numpy().astype(int)
                base[d, t]['maha_dist'] = maha_dist_list.detach().cpu().numpy()
                base[d, t]['maha_pred'] = maha_pred_list.detach().cpu().numpy().astype(int)
                base[d, t]['logit_scores'], base[d, t]['logit_pred'] = logit_score.detach().cpu().numpy(), (logit_pred + args.num_cls_per_task * t).detach().cpu().numpy().astype(int)     # max-logit scores
                base[d, t]['sm_scores'], base[d, t]['sm_pred'] = torch.cat(sm_score_list).detach().cpu().numpy(), torch.cat(sm_pred_list).detach().cpu().numpy()
                base[d, t]['smmd_scores'], base[d, t]['smmd_pred'] = torch.cat(md_scores_list).detach().cpu().numpy(), torch.cat(smmd_pred_list).detach().cpu().numpy()
                base[d, t]['en_scores'] = torch.logsumexp(torch.cat(logits_list), dim=1).detach().cpu().numpy()
                base[d, t]['enmd_scores'] = base[d, t]['en_scores'] * maha_dist_list.detach().cpu().numpy()   
                
                # max_logit
                # args.logger.print(f'--max logit scores')
                # _, maxlog_pred = torch.max(torch.cat(logits_list), dim = 1)
                # maxlog_pred = maxlog_pred + args.num_cls_per_task * t
                # maxlog_energy_score = torch.logsumexp(torch.cat(logits_list), dim=1)
                # maxlog_en_md_score = maxlog_energy_score * torch.from_numpy(maha_dist_list).to(args.device)
                # # _, maxlog_md_pred = maxlog_md_score.max(1)

                # max_logit[d, t]['pred'] = maxlog_pred.cpu().numpy().astype(int)
                # max_logit[d, t]['sm_scores'] = base[d, t]['sm_scores']      # softmax scores remain the same in max logit
                # max_logit[d, t]['en_scores'] = maxlog_energy_score.cpu().numpy()
                # max_logit[d, t]['smmd_scores'] = base[d, t]['smmd_scores']
                # max_logit[d, t]['enmd_scores'] = maxlog_en_md_score.cpu().numpy()
                

                # react
                args.logger.print(f'--react scores')
                if args.use_finetuned:
                    r_p = p_dict[str(t)]["react"]
                else:
                    r_p = 90
                args.logger.print(f'   p val for react: {r_p}')
                clip_thresh = np.load(args.load_dir + f'/m{t}_react_thresh_percentile_{r_p}.npy')
                react_logits, _ = get_react_logits(args, model, t, torch.cat(activations_list), clip_thresh) 
                # -- logit_scores
                react_logit_score, react_logit_pred = torch.max(react_logits, dim = 1)
                react_logit_pred = react_logit_pred + args.num_cls_per_task * t
                # --sm_scores
                sm_scores = torch.softmax(react_logits, dim = 1)[:, :args.num_cls_per_task]
                react_sm_score, react_sm_pred = torch.max(sm_scores, dim=1) 
                react_sm_pred = react_sm_pred + args.num_cls_per_task * t
                # -- smmd scores
                react_smmd_score, react_smmd_pred = ((sm_scores.T * maha_dist_list).T).max(1)
                react_smmd_pred = react_smmd_pred + args.num_cls_per_task * t
                # -- energy scores
                react_energy_score = torch.logsumexp(react_logits.data, dim=1)
                # --enmd scores
                react_enmd_score = react_energy_score * maha_dist_list

                react[d, t]['logit_scores'], react[d, t]['logit_pred'] = react_logit_score.detach().cpu().numpy(), react_logit_pred.detach().cpu().numpy().astype(int)
                react[d, t]['sm_scores'], react[d, t]['sm_pred'] = react_sm_score.detach().cpu().numpy(), react_sm_pred.detach().cpu().numpy().astype(int)
                react[d, t]['smmd_scores'], react[d, t]['smmd_pred'] = react_smmd_score.detach().cpu().numpy(), react_smmd_pred.detach().cpu().numpy().astype(int)
                react[d, t]['en_scores'] = react_energy_score.detach().cpu().numpy()
                react[d, t]['enmd_scores'] = react_enmd_score.detach().cpu().numpy()
                
                args.logger.print(f'--dice scores')
                if args.use_finetuned:
                    d_p = p_dict[str(t)]["dice"]
                else:
                    d_p = 85            # 75
                args.logger.print(f'   p val for dice: {d_p}')
                mean_act = np.load(args.load_dir + f'/m{t}_dice_mean_activation.npy')
                masked_w = dice_calculate_mask(args, mean_act, fc_w, d_p)
                vote = torch.cat(activations_list)[:, None, :] * masked_w
                dice_logits = (vote.sum(2) + torch.from_numpy(fc_b).to(args.device))
                # --dice logit scores 
                dice_logit_scores, dice_logit_pred = torch.max(dice_logits, dim = 1)
                dice_logit_pred = dice_logit_pred + args.num_cls_per_task * t
                # --sm_scores
                sm_scores = torch.softmax(dice_logits, dim=1)[:, :args.num_cls_per_task]
                dice_sm_scores, dice_sm_pred = torch.max(sm_scores, dim = 1)
                dice_sm_pred = dice_sm_pred + args.num_cls_per_task * t
                # -- smmd scores
                dice_smmd_scores, dice_smmd_pred = ((sm_scores.T * maha_dist_list).T).max(1)
                dice_smmd_pred = dice_smmd_pred + args.num_cls_per_task * t
                # --energy scores
                dice_energy_score = torch.logsumexp(dice_logits, dim=1)
                # --enmd scores
                dice_enmd_scores = dice_energy_score * maha_dist_list

                dice[d, t]['logit_scores'], dice[d, t]['logit_pred'] = dice_logit_scores.detach().cpu().numpy(), dice_logit_pred.detach().cpu().numpy().astype(int)
                dice[d, t]['sm_scores'], dice[d, t]['sm_pred'] = dice_sm_scores.detach().cpu().numpy(), dice_sm_pred.detach().cpu().numpy().astype(int)
                dice[d, t]['smmd_scores'], dice[d, t]['smmd_pred'] = dice_smmd_scores.detach().cpu().numpy(), dice_smmd_pred.detach().cpu().numpy().astype(int)
                dice[d, t]['en_scores'] = dice_energy_score.detach().cpu().numpy()
                dice[d, t]['enmd_scores'] = dice_enmd_scores.detach().cpu().numpy()

                # scale
                args.logger.print(f'--scale scores')
                if args.use_finetuned:
                    s_p = p_dict[str(t)]["scale"]
                else:
                    s_p = 85        # 80
                args.logger.print(f'   p val for scale: {s_p}')
                scale_logits = get_scale_logits(args, model, torch.cat(activations_list), s_p, t)
                scale_logits = scale_logits.to(args.device)
                # -- scale logit scores
                scale_logit_scores, scale_logit_pred = torch.max(scale_logits, dim = 1)
                scale_logit_pred = scale_logit_pred + args.num_cls_per_task * t
                # --sm scores
                sm_scores = torch.softmax(scale_logits, dim=1)[:, :args.num_cls_per_task]
                scale_sm_scores, scale_sm_pred = torch.max(sm_scores, dim=1)
                scale_sm_pred = scale_sm_pred + args.num_cls_per_task * t
                # --smmd_scores
                scale_smmd_scores, scale_smmd_pred = ((sm_scores.T * maha_dist_list).T).max(1)
                scale_smmd_pred = scale_smmd_pred + args.num_cls_per_task * t
                # --en scores
                scale_energy_scores = torch.logsumexp(scale_logits.data, dim=1)
                # --enmd scores
                scale_enmd_scores = scale_energy_scores * maha_dist_list
                scale[d, t]['logit_scores'], scale[d, t]['logit_pred'] = scale_logit_scores.detach().cpu().numpy(), scale_logit_pred.detach().cpu().numpy().astype(int)
                scale[d, t]['sm_scores'], scale[d, t]['sm_pred'] = scale_sm_scores.detach().cpu().numpy(), scale_sm_pred.detach().cpu().numpy().astype(int)
                scale[d, t]['smmd_scores'], scale[d, t]['smmd_pred'] = scale_smmd_scores.detach().cpu().numpy(), scale_smmd_pred.detach().cpu().numpy().astype(int)
                scale[d, t]['en_scores'] = scale_energy_scores.detach().cpu().numpy()
                scale[d, t]['enmd_scores'] = scale_enmd_scores.detach().cpu().numpy()
                
        np.savez(fname, 
                base = base,
                react = react,
                dice = dice,
                scale = scale
                )     
        args.logger.print(f'{args.method}: m{eval_model}: test ouptut values saved at {fname}') 


def rejection_vs_accuracy(scores, pred, labels):
    thresholds = np.linspace(np.min(scores), np.max(scores), 10)  # Generate 10 thresholds
    rejection_rates = []
    accuracies = []

    scores = np.array(scores)
    pred = np.array(pred)
    labels = np.array(labels)

    for thresh in thresholds:
        rejected = scores < thresh  # Reject samples with score below threshold
        accepted = ~rejected  # Keep the rest

        rejection_rate = np.mean(rejected)  # Fraction of rejected samples
        if np.sum(accepted) > 0:
            accuracy = np.mean(pred[accepted] == labels[accepted])  # Accuracy on accepted samples
        else:
            accuracy = 0  # If all are rejected, accuracy is undefined

        rejection_rates.append(rejection_rate)
        accuracies.append(accuracy)
    
    return rejection_rates, accuracies


def get_pred_label_score(eval_model, detector, method ):
    dir = os.path.abspath(os.getcwd())
    if method in ['more_fw', 'more_bw']:
        load_dir = os.path.join(dir + f'/logs/more_cifar10-5T/')
    elif method == 'build':
        load_dir = os.path.join(dir + f'/logs/build_cifar10-5T/')
    fname = os.path.join(load_dir + f'/m{eval_model}_test_scores_{method}.npz')
    print(f'Loading scores from file: {fname}')
    out = np.load(fname, allow_pickle=True)
    base, react, dice, scale = out['base'].tolist(), out['react'].tolist(), out['dice'].tolist(), out['scale'].tolist()

    if detector =='base':
        det = base
    elif detector == 'react':
        det = react
    elif detector == 'dice':
        det = dice
    elif detector == 'scale':
        det = scale

    scores, pred, labels, id_label, ood_label = [], [], [], [], []

    for d in range(5):
        int_scores, int_pred = [], []
        d_labels = base[d, 0]['gt']

        for t in range(eval_model +1):

            int_pred.append(base[d, t]['sm_pred'])
            int_scores.append(det[d, t]['enmd_scores'])

        stacked_scores = np.column_stack(int_scores)
        stacked_pred = np.column_stack(int_pred)
        row_ind = np.arange(stacked_scores.shape[0])
        task_id_pred = np.argmax(stacked_scores, axis = 1)
        class_pred = stacked_pred[row_ind, task_id_pred]
        final_scores = stacked_scores[row_ind, task_id_pred]

        if d<=t:
            id_label.append(d_labels)
        else:
            ood_label.append([-1]* row_ind.shape[0])
        
        scores.append(final_scores)
        pred.append(class_pred)
        labels = id_label + ood_label

    rej, acc = rejection_vs_accuracy(scores, pred, labels)
    return rej, acc