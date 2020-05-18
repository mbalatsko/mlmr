import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator

from mlmr.function import transform_concat, map_reduce, calculate_pool_size


class BaseMapReduceTransformer(TransformerMixin, BaseEstimator):
    """
    Sklearn wrapper base class for `mlmr.function.transform_concat` function.
    Add implementation of `transform_part` function with your logic.
    """

    def __init__(self, n_jobs=1):
        """
        :param n_jobs: number of jobs to run in parallel. `-1` means using all processors.
        """
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return transform_concat(X, self.transform_part, self.n_jobs)

    def transform_part(self, X):
        """Transform function on data slice"""
        raise NotImplementedError()


class FunctionMapReduceTransformer(TransformerMixin, BaseEstimator):
    """
    Sklearn wrapper class for `mlmr.function.map_reduce` function.
    """

    def __init__(
        self,
        map_func,
        reduce_func=pd.concat,
        data_split_func=None,
        n_jobs=1
    ):
        """
        :param map_func: (Callable) - Map function.
                         Map function signature: func(DataSplit) -> Any.
                         Function can't be lambda or local function!
        :param reduce_func: (Callable) - Reduce function. Reduce function signature: func(Iterable[MapResult]) -> Any.
        :param data_split_func: (Callable) - function that would be used to split data to perform Map operation.
                                Data split function signature: func(Iterable) -> Iterable[Any]
                                If `None` even data split function will be used.
        :param n_jobs: number of jobs to run in parallel. `-1` means using all processors.
        """

        if data_split_func is None:
            data_split_func = self.get_even_data_split_func(n_jobs)
        self.data_split_func = data_split_func
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.n_jobs = n_jobs

    @staticmethod
    def get_even_data_split_func(n_jobs):
        pool_size = calculate_pool_size(n_jobs)

        def even_data_split(data):
            return np.array_split(data, pool_size)

        return even_data_split

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return map_reduce(
            X,
            self.data_split_func,
            self.map_func,
            self.reduce_func,
            self.n_jobs
        )
