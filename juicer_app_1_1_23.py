#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-generated Spark code from Lemonade Workflow
(c) Speed Labs - Departamento de Ciência da Computação
    Universidade Federal de Minas Gerais
More information about Lemonade to be provided
"""
from concurrent.futures import ThreadPoolExecutor
import collections
import functools
import datetime
import itertools
import json
import os
import re
import simplejson
import string
import sys
import time
import threading
import traceback
import unicodedata

from textwrap import dedent
from timeit import default_timer as timer

from pyspark.ml import classification, evaluation, feature, tuning, clustering
from pyspark.sql import functions, types, Row, DataFrame
from pyspark.sql.utils import IllegalArgumentException
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import *
from pyspark.ml.clustering import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.ml.tuning import *
from pyspark.ml.recommendation import *
from pyspark.ml.regression import *
from pyspark.mllib.evaluation import *
from juicer import privaaas
from juicer.util import dataframe_util, get_emitter
from juicer.spark.reports import *
from juicer.spark.util import assemble_features_pipeline_model
from juicer.spark.ml_operation import ModelsEvaluationResultList
from juicer.spark.custom_library import *

executor = ThreadPoolExecutor(max_workers=2)
submission_lock = threading.Lock()
task_futures = {}



# noinspection PyUnusedLocal
def data_reader_0(spark_session, cached_state, emit_event):
    """
    Operation a0a7d521-c3d6-4676-b5d5-b16ad9f829d7
    Task hash: 9048694122acdb8644b58305e2b4c805313ad78b.
    """
    task_id = 'a0a7d521-c3d6-4676-b5d5-b16ad9f829d7'
    emit_task_running(task_id, spark_session, emit_event)

    start = timer()
    # First we verify whether this task's result is cached.
    results =  None
    if results is None:
        # --- Begin operation code ---- #
        schema_df_0 = types.StructType()
        schema_df_0.add('entity_id', types.IntegerType(), False)
        schema_df_0.add('score', types.IntegerType(), False)
        schema_df_0.add('label_value', types.IntegerType(), False)
        schema_df_0.add('race', types.StringType(), False)
        schema_df_0.add('sex', types.StringType(), False)
        schema_df_0.add('age_cat', types.StringType(), False)

        url = 'file:/srv/storage/srv/storage/limonero/data/41140177755a4eb48b8180e563c7fdf4_compas_for_aequitas.csv' #protect 'Protected, please update'

        df_0 = spark_session.read.option('nullValue', '').option(
            'treatEmptyValuesAsNulls', 'true').option(
            'wholeFile', True).option(
                'multiLine', True).option('escape',
                    '"').option('timestampFormat', 'yyyy/MM/dd HH:mm:ss'
                    ).csv(
                url, schema=schema_df_0,
                quote=None,
                ignoreTrailingWhiteSpace=True, # Handles 
                encoding='UTF-8',
                header=True, sep=',',
                inferSchema=False,
                mode='PERMISSIVE')
        df_0.cache()
        # --- End operation code ---- #

        results = {
            'execution_date': datetime.datetime.utcnow(),
            'task_name': 'Ler dados 1',
            'dados de saída': df_0,
             '__first__': df_0,
        }
    df_types = (DataFrame, dataframe_util.LazySparkTransformationDataframe)
    outputs = [(name, out) for name, out in results.items()
        if isinstance(out, df_types) and name[0:2] != '__']
    for name, out in outputs:
        dataframe_util.emit_sample(task_id, out, emit_event, name)
    emit_task_completed(task_id, spark_session, emit_event)

    results['time'] = timer() - start
    return results


# noinspection PyUnusedLocal
def fairness_evaluator_1(spark_session, cached_state, emit_event):
    """
    Operation 9684218b-4f74-4f37-9823-7c292144afae
    Task hash: 7fba23ea2d36c69eb9f35d26189b5649d84b9505.
    """
    task_id = '9684218b-4f74-4f37-9823-7c292144afae'

    # If the task's result is not cached, we submit its dependencies first
    parent_id = 'a0a7d521-c3d6-4676-b5d5-b16ad9f829d7'
    with submission_lock:
        if parent_id not in task_futures:
            task_futures[parent_id] = executor.submit(
                lambda: data_reader_0(spark_session, cached_state, emit_event))
    # Next we wait for the dependencies to complete
    parent_result = task_futures['a0a7d521-c3d6-4676-b5d5-b16ad9f829d7'].result()
    df_0 = parent_result['dados de saída']
    ts_df_0 = parent_result['time']
    
    emit_task_running(task_id, spark_session, emit_event)

    start = timer()
    # First we verify whether this task's result is cached.
    results =  None
    if results is None:
        # --- Begin operation code ---- #
        # from juicer.service import caipirinha_service
        from juicer.spark.vis_operation import HtmlVisualizationModel
        from juicer.spark.ext import FairnessEvaluatorTransformer
        from juicer.spark.reports import FairnessBiasReport
        from juicer.spark.reports import SimpleTableReport

        #Testing fairness evaluation sql code
        from juicer.spark.ext import FairnessEvaluatorSql
        #from pyspark.sql import SparkSession

        evaluator = FairnessEvaluatorSql(sensitive_column='race', score_column='score', 
                                         label_column='label_value', baseline_column='Caucassian', 
                                         range_column=[0.8,1.25], type_fairness_sql='list_all_groups_and_metrics',
                                         percentage_group_size=10, type_disparity='disparity_by_group'  
                                        )
        
        df_0.createOrReplaceTempView(evaluator.TABLE)
        sql_result = spark_session.sql(evaluator.get_fairness_sql())
        out_task_1 = sql_result

        baseline = 'Caucasian'

        sensitive_column_dt = df_0.schema[str('race')].dataType
        if isinstance(sensitive_column_dt, types.FractionalType):
            baseline = float(baseline)
        elif isinstance(sensitive_column_dt, types.IntegralType):
            baseline = int(baseline)
        elif not isinstance(sensitive_column_dt, types.StringType):
            raise ValueError(gettext('Invalid column type: {}').format(
            sensitive_column_dt))

        display_text = True

        headers = ['Group', 'Acceptable', 'Value']
        
        #import pdb; pdb.set_trace()
        #rows = out_task_1.select('race' , 'pred_pos_ratio_k_parity',
        #    functions.round('pred_pos_ratio_k_disparity', 2)).collect()

        '''
          pred_pos_ratio_k_parity e demais métricas referem-se as 
          colunas do registro retonados. Você precisa padronizar sql com 
          aequitas. 
        '''
        rows = out_task_1.select('race', 'positive', 'negative', 'predicted_positive', 'predicted_negative', 
                                 'group_label_positive', 'group_label_negative', 'true_negative', 'false_positive', 
                                 'false_negative', 'true_positive', 'group_size', 'accuracy', 'precision_ppv', 'recall', 
                                 'f1_score', 'group_prevalence', 'false_omission_rate', 'false_discovery_rate', 
                                 'false_positive_rate', 'false_negative_rate', 'true_negative_rate', 'negative_predictive', 
                                 'informedness', 'markedness', 'positive_likelihood_ratio', 'negative_likelihood_ratio', 
                                 'prevalence_threshold', 'jaccard_index', 'fowlkes_mallows_index', 
                                 'matthews_correlation_coefficient', 'diagnostic_odds_ratio', 'predicted_positive_rate_k', 
                                 'predicted_positive_rate_g').collect()
        #rows = out_task_1.select('race', 'predicted_positive_rate_k').collect()

        content = SimpleTableReport(
            'table table-striped table-bordered table-sm w-auto',
            headers, rows)

        emit_event(
            'update task', status='COMPLETED',
            identifier='9684218b-4f74-4f37-9823-7c292144afae',
            message='Equal Parity' + content.generate(),
            type='HTML', title='Equal Parity',
            task={'id': '9684218b-4f74-4f37-9823-7c292144afae'},
            operation={'id': 103},
            operation_id=103)

        # Records metric value
        props = ['group', 'acceptable', 'value']
        msg = json.dumps({
                'metric': 'EP',
                'workflow_id': '1',
                'values': [dict(zip(props, x)) for x in rows]
            })
        emit_event(
            'task result', status='COMPLETED',
            identifier='9684218b-4f74-4f37-9823-7c292144afae',
            content=msg,
            message=msg,
            type='METRIC', title='Equal Parity',
            task={'id': '9684218b-4f74-4f37-9823-7c292144afae'},
            operation={'id': 103},
            operation_id=103)
        '''    
          Continuar depurando daqui, pois tem erro no código. 
        '''
        if display_text:
            #html=""
            '''
              Eu preciso passar a métrica de justiça para gerar o relatório.
              Não está encontrando uma métrica.
              Não está encontrando a chave pred_pos_ratio_k_parity

              As colunas do registro do SQL da query precisam existir para 
              gerar o relatório. O código está certo, mas precisa padronizar 
              as métricas dentro do FairnessBiasReport. 
            '''
            html = FairnessBiasReport(out_task_1,
                        'race', baseline).generate()

            visualization = {
                'job_id': '0', 'task_id': '9684218b-4f74-4f37-9823-7c292144afae',
                'title': 'Bias/Fairness Report',
                'type': {'id': 1, 'name': 'HTML'},
                'model': HtmlVisualizationModel(title='Bias/Fairness Report'),
                'data': json.dumps({
                    'html': html,
                    'xhtml': '''
                        <a href="" target="_blank">
                        Bias/Fairness Report (click to open)
                        </a>'''
                }),
            }

            emit_event(
                        'update task', status='COMPLETED',
                        identifier='9684218b-4f74-4f37-9823-7c292144afae',
                        message=html,
                        type='HTML', title='Bias/Fairness Report',
                        task={'id': '9684218b-4f74-4f37-9823-7c292144afae'},
                        operation={'id': 103},
                        operation_id=103)

            # Basic information to connect to other services
            config = {
                'juicer': {
                    'services': {
                        'limonero': {
                            'url': 'http://limonero:23402',
                            'auth_token': '123456'
                        },
                        'caipirinha': {
                            'url': 'http://caipirinha:23401',
                            'auth_token': '123456',
                            'storage_id': 2
                        },
                    }
                }
            }
            # emit_event(
            #             'update task', status='COMPLETED',
            #             identifier='9684218b-4f74-4f37-9823-7c292144afae',
            #             message=base64.b64encode(fig_file.getvalue()),
            #             type='IMAGE', title='Bias/Fairness Report',
            #             task={'id': '9684218b-4f74-4f37-9823-7c292144afae'},
            #             operation={'id': 103},
            #             operation_id=103)

            # caipirinha_service.new_visualization(
            #     config,
            #     {'id': 1, 'name': 'Admin ', 'login': 'admin@lemonade.org.br'},
            #     1, 0, '9684218b-4f74-4f37-9823-7c292144afae',
            #     visualization, emit_event)
         #--- End operation code ---- #

        results = {
            'execution_date': datetime.datetime.utcnow(),
            'task_name': 'Avaliador de justiça 0',
            'dados de saída': out_task_1,
             '__first__': out_task_1,
        }
    df_types = (DataFrame, dataframe_util.LazySparkTransformationDataframe)
    outputs = [(name, out) for name, out in results.items()
        if isinstance(out, df_types) and name[0:2] != '__']
    for name, out in outputs:
        dataframe_util.emit_sample(task_id, out, emit_event, name)
    emit_task_completed(task_id, spark_session, emit_event)

    results['time'] = timer() - start
    return results


def emit_task_running(task_id, spark_session, emit_event):
    emit_event(name='update task', message=_('Task running'), status='RUNNING',
               identifier=task_id)

def emit_task_completed(task_id, spark_session, emit_event):
    emit_event(name='update task', message=_('Task completed'),
               status='COMPLETED', identifier=task_id)


def get_results(_task_futures, task_id):
    return _task_futures[task_id].result() if task_id in _task_futures else None

def get_cached_state(task_id, cached_state, emit_event, spark_session,
                     task_hash):
    results = None
    if task_id in cached_state:
        cached, _hash = cached_state.get(task_id)
        if _hash == task_hash:
            emit_event(name='update task',
                message=_('Task running (cached data)'), status='RUNNING',
                identifier=task_id)
            emit_cache = cached_state.get('emit_cache')
            if emit_cache and task_id in emit_cache:
                args, kwargs = emit_cache[task_id]
                emit_event(*args, **kwargs)
            results = cached
    return results

def main(spark_session, cached_state, emit_event):
    """ Run generated code """

    emit_cache = {}
    def cached_emit_event(*args, **kwargs):
        if 'type' in kwargs:
            msg_type = kwargs['type']
            if msg_type in ['HTML', 'IMAGE', 'JSON']: # can be cached
                task_id = kwargs.get('task').get('id')
                emit_cache[task_id] = (args, kwargs)

        emit_event(*args, **kwargs)
            
        
    try:
        task_futures['a0a7d521-c3d6-4676-b5d5-b16ad9f829d7'] = executor.submit(
            lambda: data_reader_0(spark_session, cached_state, cached_emit_event))
        task_futures['9684218b-4f74-4f37-9823-7c292144afae'] = executor.submit(
            lambda: fairness_evaluator_1(spark_session, cached_state, cached_emit_event))
        task_futures['a0a7d521-c3d6-4676-b5d5-b16ad9f829d7'].result()
        task_futures['9684218b-4f74-4f37-9823-7c292144afae'].result()

        return {
            'status': 'OK',
            'message': 'Execution defined',
            'emit_cache': emit_cache,
            'a0a7d521-c3d6-4676-b5d5-b16ad9f829d7':
                [get_results(task_futures,
                'a0a7d521-c3d6-4676-b5d5-b16ad9f829d7'),
                '9048694122acdb8644b58305e2b4c805313ad78b'],
            '9684218b-4f74-4f37-9823-7c292144afae':
                [get_results(task_futures,
                '9684218b-4f74-4f37-9823-7c292144afae'),
                '7fba23ea2d36c69eb9f35d26189b5649d84b9505'],
        }
    except Exception as e:
        spark_session.sparkContext.cancelAllJobs()
        traceback.print_exc(file=sys.stderr)
        if not dataframe_util.handle_spark_exception(e):
            raise
