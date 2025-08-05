import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from models.base import BaseLearner
# from utils.inc_net import IncrementalNet
# from utils.toolkit import count_parameters, tensor2numpy
from apprs.basemodel import BaseModel
import copy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import math

EPSILON = 1e-8

"""
Args from the MORE code
# For PASS
parser.add_argument('--kd_weight', type=float, default=10.0)
parser.add_argument('--protoAug_weight', type=float, default=10.0)
parser.add_argument('--pass_ensemble', action='store_true')
"""


class PASS(BaseModel):
    """Prototype Augmentation and Self-Supervision for Incremental Learning. CVPR2021"""
    def __init__(self, args):
        super().__init__(args)
        self.device = args.device

        # Overwrite optimizer to only train adapters
        self.optimizer = torch.optim.SGD(self.net.adapter_parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.args = args
        # self._network = IncrementalNet(args, False)
        self._protos = []
        self._radius = 0
        self._radiuses = []

        self.tb_log_dir = f'{args.logger.dir()}runs/pass'

        now = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self.tb_log_dir = f"{self.tb_log_dir}_{now}"
        # self.writer = SummaryWriter(self.tb_path)
        self.iter = 0
        print("")

        # Here we can create some aliases

    def after_task(self, task_id):
        self.iter = 0
        # self.writer.close()
        # self.writer = SummaryWriter(log_dir=f"{self.tb_log_dir}pass_task{task_id}")
        self._known_classes = len(self._protos)
        self._old_net = copy.deepcopy(self.net)
        for param in self._old_net.parameters():
            param.requires_grad = False
        self._old_net.eval()
    
    def observe(self, inputs, labels, task_id=None):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
        inputs = inputs.view(-1, 3, 224, 224)
        labels = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)
        logits, loss_clf, loss_fkd, loss_proto = self._compute_pass_loss(inputs, labels, task_id)
        loss = loss_clf + loss_fkd + loss_proto
        # loss = loss_clf + loss_proto
        self.optimizer.zero_grad()
        loss.backward()

        # Monitor gradient norm...
        total_norm = 0.
        params = self.optimizer.param_groups[0]['params']
        for p in params:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if self.args.use_clip_grad:
            total_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=self.args.clip_grad)
        
        self.optimizer.step()


        _, preds = torch.max(logits, dim=1)

        correct = preds.eq(labels.expand_as(preds)).cpu().sum()
        self.correct += correct
        self.total += len(labels)

        with SummaryWriter(log_dir=self.tb_log_dir) as writer:
            writer.add_scalar(f"PASS-{task_id}/batch_acc", correct / len(labels), self.iter)
            writer.add_scalar(f"PASS-{task_id}/tot_loss", loss, self.iter)
            writer.add_scalar(f"PASS-{task_id}/loss_clf", loss_clf, self.iter)
            writer.add_scalar(f"PASS-{task_id}/loss_fkd", loss_fkd, self.iter)
            writer.add_scalar(f"PASS-{task_id}/loss_proto", loss_proto, self.iter)
            writer.add_scalar(f"PASS-{task_id}/grad_norm", total_norm, self.iter)

        self.iter += 1

        return loss.item()

    def _compute_pass_loss(self, inputs, targets, task_id):
        features = self.net.forward_features(inputs)
        logits = self.net.forward_classifier(features)
        # logits = self.net(inputs)

        loss_clf = self.criterion(logits / self.args.temp, targets)
        
        if task_id == 0:
            return logits, loss_clf, torch.tensor(0.), torch.tensor(0.)
        
        with torch.no_grad():
            features_old = self._old_net.forward_features(inputs)
        # features = self.net.forward_features(inputs)    # not efficient, but it's the original implementation...
        loss_fkd = self.args.kd_weight * torch.dist(features, features_old, 2)

        index = np.random.choice(range(self._known_classes),
                                 size=self.args.batch_size, # * int(known_classes / (total_classes - known_classes)),
                                 replace=True)

        proto_features = np.array(self._protos)[index]
        proto_targets = 4 * index
        proto_features = proto_features + np.random.normal(0, 1, proto_features.shape) * self._radius
        proto_features = torch.from_numpy(proto_features).float().to(self.device)
        proto_targets = torch.from_numpy(proto_targets).to(self.device)
        
        proto_logits = self.net.forward_classifier(proto_features) 
        loss_proto = self.args.protoAug_weight * self.criterion(proto_logits / self.args.temp, proto_targets)
        return logits, loss_clf, loss_fkd, loss_proto
        
    
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:,::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self.net(inputs)[:,::4]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        elif hasattr(self, '_protos'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._protos/np.linalg.norm(self._protos,axis=1)[:,None])
            nme_accy = self._evaluate(y_pred, y_true)            
        else:
            nme_accy = None

        return cnn_accy, nme_accy
    

    def save(self, **kwargs):
        """
            Save model specific elements required for resuming training
            kwargs: e.g. model state_dict, optimizer state_dict, epochs, etc.
            self._protos = []
            self._radius = 0
            self._radiuses = []

            maybe self._old_net ?
        """
        self.saving_buffer['_protos'] = self._protos
        self.saving_buffer['_radius'] = self._radius
        self.saving_buffer['_radiuses'] = self._radiuses

        for key in kwargs:
            self.saving_buffer[key] = kwargs[key]

        torch.save(self.saving_buffer, self.args.logger.dir() + 'saving_buffer')