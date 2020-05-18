import multiprocessing as mp
from typing import Iterable, Callable, Any

import numpy as np
import pandas as pd


def calculate_pool_size(n_jobs: int) -> int:
    """
    Calculate processes poll size from actual cpu count and input `n_jobs`
    :param n_jobs: number of jobs to run in parallel. `-1` means using all processors.
    :return: calculated pool size
    """
    cores = mp.cpu_count()
    if n_jobs <= -1:
        pool_size = cores
    elif n_jobs <= 1:
        return 1
    else:
        pool_size = min(n_jobs, cores)
    return pool_size


def map_reduce_splits(
    data_splits: Iterable[Iterable],
    map_func: Callable[[Iterable], Any],
    reduce_func: Callable[[Iterable[Iterable]], Any],
    n_jobs: int = 1
):
    """
    Base function for performing parallel MapReduce on `data_splits`.
    From `n_jobs` argument, number of processes to run in parallel is calculated.
    Then `map_func` is applied on each element of `data_splits` in parallel.
    After calculation is complete `reduce_func` is sequentially applied on list of `map_func` results.
    `reduce_func` result is returned. Data preserves initial ordering.

    :param data_splits: (Iterable) - data splits on which MapReduce would be performed.
    :param map_func: (Callable) - Map function.
                     Map function signature: func(DataSplit) -> Any.
                     Function can't be lambda or local function!
    :param reduce_func: (Callable) - Reduce function. Reduce function signature: func(Iterable[MapResult]) -> Any.
    :param n_jobs: number of jobs to run in parallel. `-1` means using all processors.
    :return: Transformed (MapReduced) data splits.
    """
    pool_size = calculate_pool_size(n_jobs)
    pool = mp.Pool(pool_size)

    transformed = reduce_func(
        pool.map(
            map_func,
            data_splits
        )
    )
    pool.close()
    pool.join()

    return transformed


def map_reduce(
    data: Iterable,
    data_split_func: Callable[[Iterable], Iterable[Iterable]],
    map_func: Callable[[Iterable], Any],
    reduce_func: Callable[[Iterable[Iterable]], Any],
    n_jobs=1
):
    """
    Base function for performing parallel MapReduce on data.
    Firstly data are splitted into data splits using `data_split_func` function.
    From `n_jobs` argument, number of processes to run in parallel is calculated.
    Then `map_func` is applied on each data split in parallel.
    After calculation is complete `reduce_func` is sequentially applied on list of `map_func` results.
    `reduce_func` result is returned.
    Data preserves initial ordering.

    :param data: (Iterable) - data on which MapReduce would be performed.
    :param data_split_func: (Callable) - function that would be used to split data to perform Map operation.
                            Data split function signature: func(Iterable) -> Iterable[Any]
    :param map_func: (Callable) - Map function.
                     Map function signature: func(DataSplit) -> Any.
                     Function can't be lambda or local function!
    :param reduce_func: (Callable) - Reduce function. Reduce function signature: func(Iterable[MapResult]) -> Any.
    :param n_jobs: number of jobs to run in parallel. `-1` means using all processors.
    :return: Transformed (MapReduced) data.
    """
    return map_reduce_splits(data_split_func(data), map_func, reduce_func, n_jobs)


def transform_concat(
    data: Iterable,
    transform_func: Callable[[Iterable], Any],
    n_jobs: int = 1
):
    """
    Function for performing parallel data transformations on `data` (pd.DataFrame, pd.Series).
    From `n_jobs` argument, number of processes to run in parallel is calculated.
    Data is evenly divided into number of processes slices.
    Then `transform_func` is applied on each slice in parallel.
    After calculation is complete all transformation results are flattened.
    Flattened result is returned.
    Data preserves initial ordering.

    :param data: (Iterable) - data on which transformation using MapReduce would be performed.
    :param transform_func: (Callable) - transformation function of a `data`.
                           Transform function signature: func(Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series].
                           Function can't be lambda or local function!
    :param n_jobs: number of jobs to run in parallel. `-1` means using all processors.
    :return: Transformed data.
    """
    pool_size = calculate_pool_size(n_jobs)
    data_splits = np.array_split(data, pool_size)
    return map_reduce_splits(data_splits, transform_func, pd.concat, n_jobs)


