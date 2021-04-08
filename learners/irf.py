import math
import random
from collections import Counter

from skmultiflow.drift_detection import ADWIN
import numpy as np
import ray

from core.clearn import ContinualLearner
from learners.ht import HoeffdingTree
from utils.calc_utils import CalculationsUtils
from utils.coll_utils import CollectionUtils


class IncrementalRandomForest(ContinualLearner):

    def __init__(self, size: int, lambda_val=5.0, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05,
                 rnd=True, subspaces=False, att_split_est=False, log_prob=True, num_atts=0, num_cls=0, num_workers=1):
        super().__init__()
        self.size = size
        self.lambda_val = lambda_val
        self.tree_groups = []
        self.num_par_groups = num_workers
        self.par = self.num_par_groups > 1
        self.init_tree_groups([0, split_step, split_wait, hb_delta, tie_thresh, rnd, subspaces, att_split_est, log_prob,
                               num_atts, num_cls])

    def init_tree_groups(self, tree_params: list):
        trees_per_group, r = math.ceil(self.size / self.num_par_groups), self.size
        while r > 0:
            tree_params[0] = min(r, trees_per_group)
            self.tree_groups.append(RemoteTreeGroupWrapper(*tree_params) if self.par else TreeGroup(*tree_params))
            r -= trees_per_group

    def predict(self, x_batch):
        return np.array([np.argmax(ya) for ya in self.predict_prob(x_batch)])

    def predict_prob(self, x_batch):
        weights = self.fetch([tg.get_weights() for tg in self.tree_groups])
        ws = sum([sum(w) for w in weights])

        probs = self.fetch([tg.predict_prob(x_batch) for tg in self.tree_groups])
        trees_batch_probs = CollectionUtils.flatten_list(probs)
        probs_sum = [CalculationsUtils.sum_arrays([tree_probs[i] for tree_probs in trees_batch_probs]) for i in range(len(x_batch))]

        return np.array(probs_sum, dtype=object) / ws

    def update(self, x_batch, y_batch, **kwargs):
        weights = kwargs.get('weights', np.ones(len(y_batch)))

        for tree_group in self.tree_groups:
            tree_group.update_trees(x_batch, y_batch, self.lambda_val, weights)

    def get_tree_group(self, idx):
        return self.tree_groups[idx].get_trees()

    def fetch(self, obj):
        return obj if not self.par else ray.get(obj)


class TreeGroup:

    def __init__(self, size: int, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05, rnd=True, subspaces=False,
                 att_split_est=False, log_prob=True, num_atts=0, num_cls=0):
        self.trees = [
            ForestHoeffdingTree(split_step, split_wait, hb_delta, tie_thresh, rnd, subspaces, att_split_est, log_prob, num_atts, num_cls)
            for _ in range(size)
        ]

    def get_weights(self):
        return [tree.get_weight() for tree in self.trees]

    def predict_prob(self, x_batch):
        return [tree.predict_prob(x_batch) for tree in self.trees]

    def update_trees(self, x_batch, y_batch, lambda_val, weights):
        for tree in self.trees:
            k = np.random.poisson(lambda_val, len(x_batch))
            tree.update(x_batch, y_batch, weights=np.multiply(weights, k))

    def get_trees(self):
        return self.trees


class ForestHoeffdingTree(HoeffdingTree):

    def __init__(self, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05, rnd=True, subspaces=False, att_split_est=False,
                 log_prob=True, num_atts=0, num_cls=0):
        super().__init__(split_step, split_wait, hb_delta, tie_thresh, rnd, subspaces, att_split_est, log_prob, num_atts, num_cls)
        self.quality = ADWIN()

    def update(self, x_batch, y_batch, **kwargs):
        preds = super().predict(x_batch)
        for p, y in zip(preds, y_batch): self.quality.add_element(int(int(p) == int(y)))
        super().update(x_batch, y_batch, **kwargs)

    def predict_prob(self, x_batch):
        return self.get_weight() * np.array([self.find_leaf(x).predict_prob(x) for x in x_batch], dtype=object)

    def get_weight(self):
        return self.quality.estimation if self.quality.total > 0 else 1.0


@ray.remote
class RemoteTreeGroup(TreeGroup):

    def __init__(self, size: int, split_step=0.1, split_wait=100, hb_delta=0.01, tie_thresh=0.05, bag=False, subspaces=False,
                 att_split_est=False, log_prob=True, num_atts=0, num_cls=0):
        super().__init__(size, split_step, split_wait, hb_delta, tie_thresh, bag, subspaces, att_split_est, log_prob, num_atts, num_cls)


class RemoteTreeGroupWrapper:

    def __init__(self, *remote_tree_group_args):
        self.remote_tree_group = RemoteTreeGroup.remote(*remote_tree_group_args)

    def get_weights(self):
        return self.remote_tree_group.get_weights.remote()

    def predict_prob(self, x_batch):
        return self.remote_tree_group.predict_prob.remote(x_batch)

    def update_trees(self, x_batch, y_batch, lambda_val, weights):
        self.remote_tree_group.update_trees.remote(x_batch, y_batch, lambda_val, weights)

    def get_trees(self):
        return self.remote_tree_group.get_trees.remote()


