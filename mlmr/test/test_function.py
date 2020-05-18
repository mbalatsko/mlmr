import multiprocessing as mp
import numpy as np

from unittest import TestCase

from numpy.ma.testutils import assert_array_equal

from mlmr.function import calculate_pool_size, map_reduce_splits, map_reduce


def map1(data):
    return data + 1


def map2(data):
    return np.sum(data)


def reduce1(data):
    return np.concatenate(data)


def reduce2(data):
    return np.sum(data)


def split_data(data):
    return np.array_split(data, 3)


class MapReduceTest(TestCase):

    def test_calculate_pool_size(self):
        cores = mp.cpu_count()

        self.assertEqual(calculate_pool_size(-1), cores)
        self.assertEqual(calculate_pool_size(1), 1)
        self.assertEqual(calculate_pool_size(2), 2)
        self.assertEqual(calculate_pool_size(cores+1), cores)

    def test_map_reduce_splits(self):
        data = np.array(range(1, 100))
        data_splits = np.array_split(data, 3)

        result1 = map1(data)
        result2 = map2(data)

        for n_jobs in [-1, 1, 2]:
            assert_array_equal(result1, map_reduce_splits(data_splits, map1, reduce1, n_jobs))
            self.assertEqual(result2, map_reduce_splits(data_splits, map2, reduce2, n_jobs))

    def test_map_reduce(self):
        data = np.array(range(1, 100))

        result1 = map1(data)
        result2 = map2(data)

        for n_jobs in [-1, 1, 2]:
            assert_array_equal(result1, map_reduce(data, split_data, map1, reduce1, n_jobs))
            self.assertEqual(result2, map_reduce(data, split_data, map2, reduce2, n_jobs))



