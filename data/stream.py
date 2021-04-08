import random
from typing import Dict
import numpy as np
from numpy.random.mtrand import RandomState
from torch.utils.data import Dataset, Subset
from abc import ABC

from data.data_utils import IndexDataset, DataUtils
from utils.coll_utils import CollectionUtils


class Stream(ABC):
    def __init__(self, cls_names: list = None):
        self.cls_names = cls_names if cls_names is not None else []


class InstanceStream(Stream):

    def __init__(self, dataset: Dataset, order=None, frac=1.0, shuffle=False, cls_names: list = None, init_frac: float = 0.0):
        super().__init__(cls_names)
        if order:
            data_indices = order
        else:
            data_indices = list(RandomState(0).permutation(len(dataset))) if shuffle else np.arange(len(dataset))

        if frac < 1.0:
            indices = random.sample(range(len(data_indices)), int(frac * len(data_indices)))
            data_indices = [data_indices[i] for i in sorted(indices)]

        init_indices = []
        if init_frac > 0.0:
            f = int(init_frac * len(data_indices))
            init_indices, data_indices = data_indices[:f], data_indices[f:]

        self.init_data = Subset(dataset, init_indices)
        self.data = Subset(dataset, data_indices)

    def get_init_data(self):
        return self.init_data

    def get_data(self):
        return self.data

    def __len__(self):
        return len(self.data)


class ClassStream(Stream):

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset=None, class_size: int=1, class_frac: float=1.0,
                 class_batch_seq: list=None, test_frac: float=0.2, max_cls_num=-1, cls_names: list=None, init_data: Dict=None):
        super().__init__(cls_names)

        if not test_dataset:
            indices = list(RandomState(0).permutation(len(train_dataset)))
            f = int(test_frac * len(indices))
            test_dataset = Subset(train_dataset, indices[:f])
            train_dataset = Subset(train_dataset, indices[f:])

        train_class_batch = self.create_class_batches(train_dataset, class_size, class_batch_seq, max_cls_num)

        init_indices, init_class_concept_mapping = [], {}
        cf = class_frac
        random.seed(0)

        if init_data is not None:
            for class_batch_idx, frac in init_data.items():
                class_idx, class_batch_indices, class_concept_mapping = train_class_batch[class_batch_idx]

                f = int(frac * len(class_batch_indices))
                train_class_batch[class_batch_idx] = (class_idx, class_batch_indices[f:], class_concept_mapping)

                init_indices.extend(class_batch_indices[:f])
                init_class_concept_mapping.update(class_concept_mapping)

        self.init_data = (init_class_concept_mapping, Subset(train_dataset, init_indices if cf == 1.0 else random.sample(init_indices, int(cf * len(init_indices)))))

        self.train_data = [(class_idx, Subset(train_dataset, indices if cf == 1.0 else random.sample(indices, int(cf * len(indices)))), class_concept_mapping)
                           for class_idx, indices, class_concept_mapping in train_class_batch]

        test_class_batch = self.create_class_batches(test_dataset, class_size, class_batch_seq, -1)
        self.test_data = [(class_idx, Subset(test_dataset, indices if cf == 1.0 else random.sample(indices, int(cf * len(indices)))), class_concept_mapping)
                          for class_idx, indices, class_concept_mapping in test_class_batch]

    def get_init_data(self):
        return self.init_data

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    @staticmethod
    def create_class_batches(dataset, class_size, class_batch_seq, max_cls_num):
        indices_per_class = DataUtils.get_class_indices(IndexDataset(dataset))

        if not class_batch_seq:
            input_classes = sorted(list(indices_per_class.keys()))
            class_batch_seq = CollectionUtils.split_list(input_classes, class_size)

            for i, subclasses in enumerate(class_batch_seq):
                class_batch_seq[i] = (i, subclasses, {c: i for c in subclasses})

        indices_per_class_batch = [(class_idx, CollectionUtils.flatten_list([indices_per_class[cls] for cls in classes]), concept_mapping)
                                   for i, (class_idx, classes, concept_mapping) in enumerate(class_batch_seq)]

        if max_cls_num > -1:
            for i, b in enumerate(indices_per_class_batch):
                indices_per_class_batch[i] = (b[0], b[1][:max_cls_num], b[2])

        return indices_per_class_batch
