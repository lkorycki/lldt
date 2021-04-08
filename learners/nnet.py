from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import math

from core.clearn import ContinualLearner
from utils.nn_utils import NeuralNetUtils as nnu


class NeuralNet(ContinualLearner):

    def __init__(self, net: nn.Module, loss: _Loss, optimizer: Optimizer, scheduler: _LRScheduler=None, device='cpu'):
        super().__init__()
        self.net = net.to(device)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def predict(self, x_batch):
        return torch.max(self.predict_prob((x_batch.to(self.device))), 1)[1]

    def predict_prob(self, x_batch):
        self.net.eval()
        with torch.no_grad():
            return self.net(x_batch.to(self.device)).cpu()

    def update(self, x_batch, y_batch, **kwargs):
        y_batch = y_batch.long()

        self.net.train()
        self.optimizer.zero_grad()

        outputs = self.net(x_batch.to(self.device))
        loss = self.loss(outputs, y_batch.to(self.device))

        loss.backward()
        self.optimizer.step()

    def get_net(self):
        return self.net


