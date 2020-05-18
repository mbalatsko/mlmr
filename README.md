# MLMR 
[![PyPI version](https://badge.fury.io/py/mlmr.svg)](https://badge.fury.io/py/mlmr)
[![Downloads](https://pepy.tech/badge/mlmr)](https://pepy.tech/project/mlmr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This library will help you easily parallelize your python code for all kind of data transformations. 
Core functions are built on Map-Reduce paradigm. In this library Map part is parallelized using native 
python `multiprocessing` module.

## Installation

```bash
pip install mlmr
```

## Usage

In order to find out library API specification and advanced usage I recommend you to start with these short tutorials:

1. [Functional API tutorial](https://github.com/mbalatsko/mlmr/blob/master/tutorials/Function%20tutorial.ipynb)
1. [Sklearn integration tutorial](https://github.com/mbalatsko/mlmr/blob/master/tutorials/Sklearn%20integration%20tutorial.ipynb)

Here I'll post several real world `mlmr` API applications.

### Sum of squares in MapReduce fashion example

```python
import numpy as np
from mlmr.function import map_reduce

arr = [1, 2, 3, 4, 5]

def squares_of_slice(arr_slice): # our map function, with partial reduction
    return sum(map(lambda x: x**2, arr_slice))

def get_split_data_func(n_slices): # wrapper function of split data function
    def split_data(data):
        return np.array_split(data, n_slices)
    return split_data

n_jobs = 2

result = map_reduce(
    data=arr,
    data_split_func=get_split_data_func(n_jobs), # split data into n_jobs slices
    map_func=squares_of_slice,
    reduce_func=sum,
    n_jobs=n_jobs
)
```

### Pandas apply parallelization in MapReduce fashion example

In this example function performs parallel data transformations on `df` (pd.DataFrame, pd.Series).
From `n_jobs` argument, number of processes to run in parallel is calculated. Data is evenly divided into number 
of processes slices. Then `our_transform_func` is applied on each slice in parallel (every process has its own slice).
After calculation is complete all transformation results are flattened. Flattened result is returned.

```python
from mlmr.function import transform_concat

def comutation_costly_transformation(*_):
    pass

def our_transform_func(df):
    return df.apply(cosly_computation_func)

df_transformed = transform_concat(df, transform_func=our_transform_func, n_jobs=-1)
```

### Sklearn MapReduce transformer integration into Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from mlmr.transformers import BaseMapReduceTransformer

def comutation_costly_text_transformation(df):
    pass

class TextPreprocessor(BaseMapReduceTransformer):
    
    def transform_part(self, X):
        return comutation_costly_text_transformation(X)

n_jobs = 4

text_classification_pipeline = Pipeline([
     ('text_preprocessor', TextPreprocessor(n_jobs=n_jobs)),
     ('vectorizer', TfidfVectorizer(analyzer = "word", max_features=10000)),
     ('classifier', RandomForestClassifier(n_estimators=100, n_jobs=n_jobs))
])
```

Alternative implementation:

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from mlmr.transformers import FunctionMapReduceTransformer

def get_split_data_func(n_slices): # wrapper function of split data function
    def split_data(data):
        return np.array_split(data, n_slices)
    return split_data

def comutation_costly_text_transformation(df):
    pass

n_jobs = 4

text_classification_pipeline = Pipeline([
     ('text_preprocessor', FunctionMapReduceTransformer(
         map_func=comutation_costly_text_transformation,
         reduce_func=pd.concat,
         data_split_func=get_split_data_func(n_jobs),
         n_jobs=n_jobs
     )),
     ('vectorizer', TfidfVectorizer(analyzer = "word", max_features=10000)),
     ('classifier', RandomForestClassifier(n_estimators=100, n_jobs=n_jobs))
])
```