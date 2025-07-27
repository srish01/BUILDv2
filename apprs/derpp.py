# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
from apprs.basemodel import BaseModel
import numpy as np
# from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

# device = "cuda:2" if torch.cuda.is_available() else "cpu"

class Derpp(BaseModel):
    """Continual learning via Dark Experience Replay++."""
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    # @staticmethod
    # def get_parser(parser) -> ArgumentParser:
    #     add_rehearsal_args(parser)
    #     parser.add_argument('--alpha', type=float, required=True, default=0.5,
    #                         help='Penalty weight.')
    #     parser.add_argument('--beta', type=float, required=True, default=0.5,
    #                         help='Penalty weight.')
    #     return parser

    # def __init__(self, backbone, loss, args, transform, dataset=None):
    #     super().__init__(backbone, loss, args, transform, dataset=dataset)
    def __init__(self, args):
        super(Derpp, self).__init__(args)
        self.device = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.buffer = Buffer(self.args.buffer_size, self.device)

        self.args.alpha = args.set_alpha if args.set_alpha is not None else args.alpha
        self.args.beta = args.set_beta if args.set_beta is not None else args.beta

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss += loss_mse

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss += loss_ce

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()
    
    def set_seed(self):
        seed=self.args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True