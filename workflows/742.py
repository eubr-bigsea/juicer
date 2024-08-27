#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-generated Spark code from Lemonade Workflow
(c) Speed Labs - Departamento de Ciência da Computação
    Universidade Federal de Minas Gerais
More information about Lemonade to be provided.
"""
from pyspark.ml import (classification, regression, clustering,
                        evaluation, feature, tuning)
from pyspark.sql import functions, types, Row, DataFrame

from pyspark.ml import Pipeline, PipelineModel

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import itertools
import math
import operator
import random
import traceback
import sys
import time
import uuid

import numpy as np
import os
from juicer.util import dataframe_util
import dataclasses

# FIXME: Do not work


@dataclasses.dataclass(frozen=True)
class TrainingContext:
    index: int
    task_id: int
    task_name: str
    name: str
    details: str
    operation_id: int
    start: float
    time: float
    def replace(self, *args, start: float, time: float):
       return dataclasses.replace(
           self, start=start, time=time)

@dataclasses.dataclass(frozen=True)
class TrainingResult:
    status: str = None
    model: str = None
    details: str = None
    metric: str = None
    exception: str = None
    ctx: object = None
    feature_importance: str = None
    def __iter__(self):
        return iter(
            (self.status, self.model, self.details, self.metric,
            self.exception, self.ctx, self.feature_importance)
        )



def get_feature_importance_score(model, features):
    if isinstance(model, classification.LogisticRegressionModel):
        m = model.extractParamMap()
        if m.get(model.family) in ('multinomial', 'auto'):
            coef = model.coefficientMatrix
            mag_coef = []
            result = coef
        else:
            print(model.family, m.get(model.family))
            coef = list(model.coefficients)
            result = list(zip(features, coef))
        print(result)
    elif isinstance(model,
                    (classification.DecisionTreeClassificationModel,
                     classification.RandomForestClassificationModel)):
        return model.featureImportances.toArray().tolist()
    elif isinstance(model, (classification.NaiveBayesModel)):
        return [{
            'pi': model.pi.values.tolist(),
            'theta': model.theta.values.tolist(),
        }]
    else:
        print('>>>>>>>>>>>>>>>>>>', type(model))

class CustomTrainValidationSplit():
    """ Mimic the TrainValidationSplit from Spark, but emit information
    about the training process after each iteration."""
    __slots__ = ('estimator', 'evaluator', 'train_ratio', 'seed', 'params',
               'train_size', 'test_size', 'num_workers', 'strategy', 'folds',
               'fold_col')
    def __init__(self, estimator, evaluator, params,
                 train_ratio=0.7, seed=None,
                 num_workers=4, strategy='split', folds=None):

        self.estimator = estimator
        self.evaluator = evaluator
        self.train_ratio = train_ratio
        self.seed = seed
        self.params = params
        self.train_size = 0
        self.test_size = 0
        self.num_workers = 4
        self.strategy = strategy
        self.folds = folds
        self.fold_col = None

    def _perform_training(self, train, test, estimator, evaluator, params,
                          details, ctx):
        try:
            start = time.time()
            model = estimator.fit(train, params)
            df = model.transform(test, params)
            metric = evaluator.evaluate(df)
            new_ctx = ctx.replace(start=start, time=time.time() - start)
            return TrainingResult('OK', model, details, metric, ctx=new_ctx)
        except Exception as e:
            try:
                dataframe_util.handle_spark_exception(e)
            except ValueError as ve:
                return TrainingResult('ERROR', model=None, metric=None,
                                      details=details, exception=ve, ctx=ctx)

    def _extract_params(self, params):
        task_id, task_name, operation_id, estimator, details = (
            None, None, None, None, {})
        for param, value in params.items():
            if param.name == 'stages':
                task_id = value[-1].task_id
                task_name = value[-1].task_name
                operation_id = value[-1].operation_id
                estimator = value[-1].__class__.__name__
            else:
                details[param.name] = value
        return task_id, task_name, operation_id, estimator, details

    def _generate_folds(self, df):
        if self.strategy == 'split':
            return [df.randomSplit(
                [self.train_ratio, 1 - self.train_ratio], self.seed)]
        elif self.strategy == 'cross_validation':
            return self._k_fold()
        else:
            raise ValueError('Invalid train split: {}'.format(self.strategy))

    def _k_fold(self, df):
        """ Generates k-fold. Based on the PySpark implementation:
        https://github.com/apache/spark/blob/master/python/pyspark/ml/tuning.py
        """
        datasets = []

        if self.fold_col:
            # FIXME: Implement support to fold_col
            pass
        else:
            h = 1.0 / self.folds
            rand_col = f'{uuid.uuid5()}_rand'
            df = df.select("*", functions.rand(self.seed).alias(rand_col))
            for i in range(self.folds):
                validate_lower_bound = i * h
                validate_upper_bound = (i + 1) * h
                condition = ((df[rand_col] >= validate_lower_bound) &
                             (df[rand_col] < validate_upper_bound))
                validation = df.filter(condition)
                train = df.filter(~condition)
                datasets.append((train, validation))

            return datasets

    def fit(self, df, emit):
        results = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            folds = self._generate_folds(df)
            for train, test in folds:
                train.cache()
                test.cache()
                future_train = {}
                for counter, params in enumerate(self.params):
                    task_id, task_name, operation_id, estimator, details = (
                        self._extract_params(params))
                    # task_id = None
                    # task_name = None
                    # operation_id = None
                    # estimator = None
                    # details = {}
                    # for param, value in params.items():
                    #     if param.name == 'stages':
                    #         task_id = value[-1].task_id
                    #         task_name = value[-1].task_name
                    #         operation_id = value[-1].operation_id
                    #         name = value[-1].__class__.__name__
                    #     else:
                    #         details[param.name] = value

                    ctx = TrainingContext(counter, task_id, task_name,
                        estimator, details, operation_id, time.time(), 0)
                    submission = executor.submit(
                        self._perform_training, train, test, self.estimator,
                        self.evaluator, params, details, ctx)
                    future_train[submission] = ctx

                is_larger_better = self.evaluator.isLargerBetter()
                for future in as_completed(future_train):
                    result = future.result()
                    # future_train[future]
                    ctx = result.ctx
                    message = {'params': ctx.details, 't': ctx.time,
                        'index': ctx.index}
                    if result.metric is not None and math.isnan(result.metric):
                         message['error'] = 'Metric could not be computed'
                         emit(identifier=ctx.task_id, status='ERROR',
                                message=message, title=ctx.task_name,
                                operation_id=ctx.operation_id)

                    elif result.status == 'OK':
                        results.append(result)
                        metric_name = self.evaluator.extractParamMap().get(
                            self.evaluator.metricName)
                        message['metric'] = {
                            'name': metric_name,
                            'value': result.metric if not math.isnan(result.metric) else 1000,
                            'is_larger_better': is_larger_better
                        }
                        ml_model = result.model.stages[-1]
                        importance = get_feature_importance_score(ml_model, [])
                        message['feature_importance'] = importance

                        emit(identifier=ctx.task_id, status='COMPLETED',
                            message=message, title=ctx.task_name,
                            operation_id=ctx.operation_id, )
                    else:
                        message['error'] = str(result.exception)
                        emit(identifier=ctx.task_id, status='ERROR',
                                message=message, title=ctx.task_name,
                                operation_id=ctx.operation_id)


                self.train_size = train.count()
                self.test_size = test.count()
                # Unpersist training & validation set once all metrics have been produced
                train.unpersist()
                test.unpersist()

        return results


def emit_metric(emit_event):
    return functools.partial(
        emit_event, name='task result', type='METRIC')

def read_data(spark_session):
    """ Read input data."""

    schema_df = types.StructType()
    schema_df.add('sepal_length', types.DecimalType(5, 1), False)
    schema_df.add('sepal_width', types.DecimalType(5, 1), False)
    schema_df.add('petal_length', types.DecimalType(5, 1), False)
    schema_df.add('petal_width', types.DecimalType(5, 1), False)
    schema_df.add('species', types.StringType(), False)

    url = '/'.join(['hdfs:', '', 'spark01.ctweb.inweb.org.br:9000', 'limonero', 'data', '0d6b6e1cee844d9bac6b028b04ed9c38_iris.csv']) #protect 'Protected, please update'
    jvm = spark_session._jvm
    jvm.java.lang.System.setProperty("HADOOP_USER_NAME", "hadoop")
    os.environ['HADOOP_USER_NAME'] = 'hadoop'

    df = spark_session.read.option('nullValue', '').option(
        'treatEmptyValuesAsNulls', 'true').option(
        'wholeFile', True).option(
            'multiLine', True).option('escape',
                '"').option('timestampFormat', 'yyyy/MM/dd HH:mm:ss'
                ).csv(
            url, schema=schema_df,
            quote=None,
            ignoreTrailingWhiteSpace=True, # Handles 
            encoding='UTF-8',
            header=True, sep=',',
            inferSchema=False,
            mode='PERMISSIVE')
    df.cache()

    # Many Spark 2.x transformers, including Binarizer and Imputer,
    # work only with float or double types.
    # Decimal and even integer is not supported.
    # See https://issues.apache.org/jira/browse/SPARK-20604.
    # Current solution is to convert all unsupported numeric data types
    # to double
    for s in df.schema:
        if s.dataType.typeName() in ['integer', 'decimal']:
            df = df.withColumn(s.name, df[s.name].cast('double'))

    
    if df.schema['sepal_length'].dataType.typeName() == 'string':
        # Cast features marked as Numeric, but loaded as String
        df = df.withColumn('sepal_length', df['sepal_length'].cast('double'))
    if df.schema['sepal_width'].dataType.typeName() == 'string':
        # Cast features marked as Numeric, but loaded as String
        df = df.withColumn('sepal_width', df['sepal_width'].cast('double'))
    if df.schema['petal_length'].dataType.typeName() == 'string':
        # Cast features marked as Numeric, but loaded as String
        df = df.withColumn('petal_length', df['petal_length'].cast('double'))
    if df.schema['petal_width'].dataType.typeName() == 'string':
        # Cast features marked as Numeric, but loaded as String
        df = df.withColumn('petal_width', df['petal_width'].cast('double'))


    return df

def create_pipeline(label: str):
    """ Define a Spark pipeline model."""

    # Feature engineering
    features_stages = []

    

    features_names = ['sepal_length', 'sepal_length', 'sepal_width', 'sepal_width', 'petal_length', 'petal_length', 'petal_width', 'petal_width']

    # Assembling of all features
    vec_assembler = feature.VectorAssembler(
        handleInvalid='skip',
        outputCol='features',
        inputCols=['sepal_length',
            'sepal_length',
            'sepal_width',
            'sepal_width',
            'petal_length',
            'petal_length',
            'petal_width',
            'petal_width',
            
        ])

    # Algorithms definition
    common_params = {'predictionCol': 'prediction',
        'featuresCol': 'features'}
    kmeans = clustering.KMeans(**common_params)

    # Lemonade internal use
    kmeans.task_id = '54078035-18f8-4d7b-b00f-ad524eb35dfc'
    kmeans.task_name = 'Agrupamento K-Means'
    kmeans.operation_id = 2355
    bisecting_kmeans = clustering.BisectingKMeans(**common_params)

    # Lemonade internal use
    bisecting_kmeans.task_id = 'fd312ed2-9055-41d7-b284-6021c9355e9d'
    bisecting_kmeans.task_name = 'Bisecting k-means'
    bisecting_kmeans.operation_id = 2255

    common_stages = features_stages +  [vec_assembler]
    # CrossValidation/TrainValidationSplit with multiple pipelines in PySpark:
    # See https://stackoverflow.com/a/54992651/1646932
    pipeline = Pipeline(stages=[])

    # Grid parameters (Using: grid)
    grid_kmeans = (tuning.ParamGridBuilder()
        .baseOn({pipeline.stages: common_stages + [kmeans] })
        .addGrid(kmeans.k, [3, 2, 4, 6, 8])
        .addGrid(kmeans.tol, [0.0001])
        .addGrid(kmeans.initMode, ['k-means||'])
        .addGrid(kmeans.distanceMeasure, ['euclidean'])
        .addGrid(kmeans.seed, [1])
        .build()
    )
    
    grid_bisecting_kmeans = (tuning.ParamGridBuilder()
        .baseOn({pipeline.stages: common_stages + [bisecting_kmeans] })
        .addGrid(bisecting_kmeans.k, [3])
        .addGrid(bisecting_kmeans.maxIter , [100])
        .addGrid(bisecting_kmeans.seed, [1])
        .addGrid(bisecting_kmeans.minDivisibleClusterSize, [1.0])
        .addGrid(bisecting_kmeans.distanceMeasure, ['euclidean'])
        .build()
    )
    
    # Remove duplicated grid parameters
    candidate_grid =grid_kmeans + grid_bisecting_kmeans
    distinct_grid = set()
    grid = []
    for row in candidate_grid:
        key = tuple((p, v if not isinstance(v, list) else tuple(v))
            for p, v in row.items() if p.name != 'stages')
        if key not in distinct_grid:
            distinct_grid.add(key)
            grid.append(row)

    if len(grid) == 0:
        raise ValueError(
            gettext('No algorithm or algorithm parameter informed.'))
    
    return pipeline, grid, label

def main(spark_session: any, cached_state: dict, emit_event: callable):
    """ Run generated code """

    try:
        label = None
        df = read_data(spark_session)
        df = df.sample(False, fraction=float(min(1, 10000/df.count())), seed=183375460)

        # Missing data handling
        # nothing to handle

        # Pipeline definition
        pipeline, grid, label = create_pipeline(label)

        emit = emit_metric(emit_event)

        # Pipeline evaluator
        evaluator = evaluation.ClusteringEvaluator(
            metricName='silhouette', )
        evaluator.task_id = 'c8955105-8531-4bd3-923f-d8ae05577a25'
        evaluator.operation_id = 2351

        # Train/validation definition
        # It can be a simple train + validation or cross validation
        train_ratio = 0.8 # Between 0.01 and 0.99
        executor = CustomTrainValidationSplit(
            pipeline, evaluator, grid, train_ratio, seed=None,
            strategy='split')

        results = executor.fit(df, emit)
        results = sorted(results, key=lambda x: x.metric,
                         reverse=evaluator.isLargerBetter())

        if len(results):
            status, model, params, metric, ex, ctx, fi = results[0]
            metric_name = evaluator.extractParamMap().get(evaluator.metricName)
            message = {'best': params, 'metric_value': metric,
                'larger_better': evaluator.isLargerBetter(),
                'index': ctx.index, 'metric_name': metric_name,
                'estimator': ctx.name, 'task_name': ctx.task_name,
                'train_size': executor.train_size,
                'test_size': executor.test_size,
                'model_count': len(results)
                }

            emit(status='COMPLETED', type='OTHER',
                identifier=evaluator.task_id,
                operation_id=evaluator.operation_id,
                title='Avaliar modelo',
                message=message)
            # print(message)
        for status, model, params, metric, ex, ctx, fi in results:
            pass
            # print(model.stages[-1], [(p, v) for p, v in params.items()], metric)
            # ml_model = model.stages[-1]
            # features = model.stages[-2].getInputCols()
            # print('#' * 20)
            # print('Features', features)
            # importance = get_feature_importance_score(ml_model, features)
            # print('#' * 20)
            # print(feature_assembler.metadata)
            # print(ml_model.metadata["ml_attr"]["attrs"].values())


    except Exception as e:
        spark_session.sparkContext.cancelAllJobs()
        traceback.print_exc(file=sys.stderr)
        if not dataframe_util.handle_spark_exception(e):
            raise