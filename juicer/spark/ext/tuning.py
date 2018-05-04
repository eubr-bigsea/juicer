# coding=utf-8
from multiprocessing.pool import ThreadPool

from pyspark import keyword_only
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark.sql.functions import rand

import numpy as np


# noinspection PyPep8Naming
def _parallelFitTasks(est, train, eva, validation, epm):
    """
    Creates a list of callables which can be called from different threads to
    fit and evaluate
    an estimator in parallel. Each callable returns an `(index, metric)` pair.
    Copied from pyspark.ml.tuning.

    :param est: Estimator, the estimator to be fit.
    :param train: DataFrame, training data set, used for fitting.
    :param eva: Evaluator, used to compute `metric`
    :param validation: DataFrame, validation data set, used for evaluation.
    :param epm: Sequence of ParamMap, params maps to be used during fitting &
    evaluation.
    :return: (int, float), an index into `epm` and the associated metric value.
    """
    modelIter = est.fitMultiple(train, epm)

    # noinspection PyPep8Naming
    def singleTask():
        index, model = next(modelIter)
        metric = eva.evaluate(model.transform(validation, epm[index]))
        return index, metric

    return [singleTask] * len(epm)


# noinspection PyPep8Naming
class CustomCrossValidator(CrossValidator):
    """
    Adapted from Copied from pyspark.ml.tuning. All changes in base class must
    be replicated here.
    Save all intermediate models.
    """

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None,
                 numFolds=3, seed=None, parallelism=1):
        super(CrossValidator, self).__init__()
        self._setDefault(numFolds=3, parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)
        self.all_results = None

    @property
    def variance(self):
        return [np.var(v) for v in self.all_results]

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        h = 1.0 / nFolds
        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        metrics = [0.0] * numModels

        self.all_results = [[[]] * nFolds] * len(epm)
        pool = ThreadPool(processes=min(self.getParallelism(), numModels))

        for i in range(nFolds):
            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
            validation = df.filter(condition).cache()
            train = df.filter(~condition).cache()

            tasks = _parallelFitTasks(est, train, eva, validation, epm)
            for j, metric in pool.imap_unordered(lambda f: f(), tasks):
                metrics[j] += (metric / nFolds)
                self.all_results[j][i] = metric
            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics))
