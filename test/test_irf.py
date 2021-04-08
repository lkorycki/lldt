import unittest
import numpy as np
import datetime
import multiprocessing
import ray

import learners.irf as irf

num_cores = multiprocessing.cpu_count()
ray.init(num_cpus=num_cores, ignore_reinit_error=True)


class IncrementalRandomForestTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_irf(self):
        num_atts, num_cls, size = 10, 2, 5
        c0_data = np.random.multivariate_normal(np.ones(num_atts) * 10.0, np.eye(num_atts) * 1.0, 10)
        c1_data = np.random.multivariate_normal(np.ones(num_atts) * -10.0, np.eye(num_atts) * 1.0, 10)

        forest = irf.IncrementalRandomForest(size, split_wait=19, num_atts=num_atts, num_cls=num_cls, num_workers=1)
        self.assertEqual(len(forest.tree_groups), 1)
        self.assertEqual(len(forest.tree_groups[0].trees), size)
        self.__test_forest(forest, c0_data, c1_data)

        forest = irf.IncrementalRandomForest(size, split_wait=19)

        pred_probs = forest.predict_prob(c0_data[:2])
        preds = forest.predict(c0_data[:2])
        self.assertListEqual(pred_probs.tolist(), [[0.0], [0.0]])
        self.assertListEqual(preds.tolist(), [0, 0])

        size, num_par_groups = 8, 3
        parallel_forest = irf.IncrementalRandomForest(8, split_wait=19, num_atts=num_atts, num_cls=num_cls, num_workers=3)
        self.assertEqual(len(parallel_forest.tree_groups), num_par_groups)
        self.assertEqual(len(ray.get(parallel_forest.get_tree_group(0))), 3)
        self.assertEqual(len(ray.get(parallel_forest.get_tree_group(1))), 3)
        self.assertEqual(len(ray.get(parallel_forest.get_tree_group(2))), 2)
        self.__test_forest(parallel_forest, c0_data, c1_data)

    def __test_forest(self, forest, c0_data, c1_data):
        pred_probs = forest.predict_prob(c0_data[:2])
        preds = forest.predict(c0_data[:2])
        self.assertListEqual(pred_probs.tolist(), [[0.0, 0.0], [0.0, 0.0]])
        self.assertListEqual(preds.tolist(), [0, 0])

        forest.update(c0_data, np.zeros(len(c0_data)), weights=np.ones(len(c0_data)))
        forest.update(c1_data, np.ones(len(c1_data)), weights=np.ones(len(c1_data)))
        pred_probs = forest.predict_prob(np.array([c0_data[0], c1_data[0]]))
        preds = forest.predict(np.array([c0_data[0], c1_data[0]]))
        self.assertGreater(pred_probs[0][0], pred_probs[0][1])
        self.assertGreater(pred_probs[1][1], pred_probs[1][0])
        self.assertListEqual(preds.tolist(), [0, 1])

    # @unittest.skip
    # def test_irf_perf(self):
    #     num_atts, num_cls, size = 1000, 2, 10
    #     forest = irf.IncrementalRandomForest(size)
    #
    #     c0_data = np.random.multivariate_normal(np.ones(num_atts) * 10.0, np.eye(num_atts) * 1.0, 150)
    #     c1_data = np.random.multivariate_normal(np.ones(num_atts) * -10.0, np.eye(num_atts) * 1.0, 150)
    #
    #     forest2 = sk.meta.AdaptiveRandomForestClassifier(10, drift_detection_method=None)
    #     start_time = datetime.datetime.now()
    #     forest.update(c0_data, np.zeros(len(c0_data)), weights=np.ones(len(c0_data)))
    #     # forest2.update(c0_data, np.zeros(len(c0_data)), [0, 1])
    #     print(datetime.datetime.now() - start_time)


