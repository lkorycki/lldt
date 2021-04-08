import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from skmultiflow.trees import HoeffdingTreeClassifier
from river.ensemble import AdaptiveRandomForestClassifier
from river.tree import HoeffdingTreeClassifier
from river.multiclass.ovr import OneVsRestClassifier

import data.data_collection as data_col
from data.stream import ClassStream
from eval.eval import ClassStreamEvaluator
from eval.experiment import Experiment
from learners.ht import HoeffdingTree
from learners.irf import IncrementalRandomForest
from utils.cls_utils import RiverWrapper
import ray


num_cores = multiprocessing.cpu_count()
ray.init(num_cpus=num_cores, ignore_reinit_error=True)


class ExperimentLifelongTree(Experiment):
    def prepare(self):
        logdir_root = 'runs/ll_tree'

        self.add_algorithm_creator('HT', lambda: HoeffdingTree(att_split_est=False))
        self.add_algorithm_creator('HT-s10', lambda: HoeffdingTree(att_split_est=False, split_wait=10))
        self.add_algorithm_creator('HT-ae', lambda: HoeffdingTree(att_split_est=True, log_prob=True))
        self.add_algorithm_creator('HT-ae-s10', lambda: HoeffdingTree(att_split_est=True, split_wait=10, log_prob=True))

        self.add_algorithm_creator('IRF40', lambda: IncrementalRandomForest(size=40, num_workers=10, att_split_est=False))
        self.add_algorithm_creator('IRF40-s10', lambda: IncrementalRandomForest(size=40, split_wait=10, num_workers=10, att_split_est=False))
        self.add_algorithm_creator('IRF40-ae', lambda: IncrementalRandomForest(size=40, num_workers=10, att_split_est=True))
        self.add_algorithm_creator('IRF40-ae-s10', lambda: IncrementalRandomForest(size=40, split_wait=10, num_workers=10, att_split_est=True))

        self.add_algorithm_creator('ARF40', lambda: RiverWrapper(AdaptiveRandomForestClassifier(split_confidence=0.01, n_models=40)))
        self.add_algorithm_creator('BAG40', lambda: IncrementalRandomForest(size=40, num_workers=10, att_split_est=False, rnd=False, subspaces=False))
        self.add_algorithm_creator('RSP40', lambda: IncrementalRandomForest(size=40, num_workers=10, att_split_est=False, rnd=False, subspaces=True))
        self.add_algorithm_creator('OVR', lambda: RiverWrapper(OneVsRestClassifier(HoeffdingTreeClassifier(split_confidence=0.01))))

        self.add_data_creator('MNIST-CI-FLAT',
                              lambda: ClassStream(data_col.get('MNIST-TRAIN-FLAT'), data_col.get('MNIST-TEST-FLAT'), class_size=1))
        self.add_data_creator('FASHION-CI-FLAT',
                              lambda: ClassStream(data_col.get('FASHION-TRAIN-FLAT'), data_col.get('FASHION-TEST-FLAT'), class_size=1))
        self.add_data_creator('SVHN-TENSOR-CI',
                              lambda: ClassStream(data_col.get('SVHN-TRAIN-TENSOR'), data_col.get('SVHN-TEST-TENSOR'), class_size=1, max_cls_num=4658))
        self.add_data_creator('CIFAR20C-TENSOR-CI',
                              lambda: ClassStream(data_col.get('CIFAR20C-TRAIN-TENSOR'), data_col.get('CIFAR20C-TEST-TENSOR'), class_size=1))
        self.add_data_creator('IMAGENET20A-TENSOR-CI',
                              lambda: ClassStream(data_col.get('IMAGENET20A-TRAIN-TENSOR'), data_col.get('IMAGENET20A-TEST-TENSOR'), class_size=1))
        self.add_data_creator('IMAGENET20B-TENSOR-CI',
                              lambda: ClassStream(data_col.get('IMAGENET20B-TRAIN-TENSOR'), data_col.get('IMAGENET20B-TEST-TENSOR'), class_size=1))

        self.add_evaluator_creator('IncEval-shallow', lambda: ClassStreamEvaluator(batch_size=256, shuffle=True, num_epochs=1, num_workers=8,
                                                               logdir_root=logdir_root, numpy=True, vis=False))


def run():
    seqs = ['MNIST-CI-FLAT', 'FASHION-CI-FLAT', 'SVHN-TENSOR-CI', 'CIFAR20C-TENSOR-CI']
    img_seqs = ['IMAGENET20A-TENSOR-CI', 'IMAGENET20B-TENSOR-CI']

    ExperimentLifelongTree().run(algorithms=['HT', 'IRF40'], streams=seqs, evaluators=['IncEval-shallow'])
    ExperimentLifelongTree().run(algorithms=['HT-s10', 'IRF40-s10'], streams=img_seqs, evaluators=['IncEval-shallow'])

    ExperimentLifelongTree().run(algorithms=['HT-ae', 'IRF40-ae'], streams=seqs, evaluators=['IncEval-shallow'])
    ExperimentLifelongTree().run(algorithms=['HT-ae-s10', 'IRF40-ae-s10'], streams=img_seqs, evaluators=['IncEval-shallow'])

    ExperimentLifelongTree().run(algorithms=['ARF40', 'BAG40', 'RSP40', 'OVR'], streams=seqs + img_seqs, evaluators=['IncEval-shallow'])
