# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import ast
import json
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.runner import configuration


from tests import compare_ast, format_code_comparison



from juicer.scikit_learn.clustering_operation import \
    ClusteringModelOperation, \
    LdaClusteringOperation, KMeansClusteringOperation, \
    GaussianMixtureClusteringOperation

"""
    K-Means tests
"""


def test_kmeans_clustering_operation_random_type_minibatch_success():
    params = {
        KMeansClusteringOperation.N_CLUSTERS_PARAM: 10,
        KMeansClusteringOperation.MAX_ITER_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAM:
            KMeansClusteringOperation.TYPE_PARAM_MB,
        KMeansClusteringOperation.INIT_PARAM:
            KMeansClusteringOperation.INIT_PARAM_RANDOM,
        KMeansClusteringOperation.TOLERANCE_PARAM: 0.001,
        KMeansClusteringOperation.SEED_PARAM: 15
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    name = "clustering.KMeans"

    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        from sklearn.cluster import MiniBatchKMeans
        {output} = MiniBatchKMeans(n_clusters={k}, init='{init}',
                max_iter={max_iter}, tol={tol}, random_state={seed})
        """.format(output=named_outputs['algorithm'],
                   k=params[KMeansClusteringOperation.N_CLUSTERS_PARAM],
                   init=params[KMeansClusteringOperation.INIT_PARAM],
                   max_iter=params[KMeansClusteringOperation.MAX_ITER_PARAM],
                   seed=params[KMeansClusteringOperation.SEED_PARAM],
                   tol=params[KMeansClusteringOperation.TOLERANCE_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_kmeans_clustering_operation_random_type_kmeans_success():
    params = {
        KMeansClusteringOperation.N_CLUSTERS_PARAM: 10,
        KMeansClusteringOperation.MAX_ITER_PARAM: 20,
        KMeansClusteringOperation.TYPE_PARAM:
            KMeansClusteringOperation.TYPE_PARAM_KMEANS,
        KMeansClusteringOperation.INIT_PARAM:
            KMeansClusteringOperation.INIT_PARAM_RANDOM,
        KMeansClusteringOperation.TOLERANCE_PARAM: 0.001,
        KMeansClusteringOperation.SEED_PARAM: 15
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    name = "clustering.KMeans"
    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        from sklearn.cluster import KMeans
        {output} = KMeans(n_clusters={k}, init='{init}',
                max_iter={max_iter}, tol={tol}, random_state={seed})
        """.format(output=named_outputs['algorithm'],
                   k=params[KMeansClusteringOperation.N_CLUSTERS_PARAM],
                   init=params[KMeansClusteringOperation.INIT_PARAM],
                   max_iter=params[KMeansClusteringOperation.MAX_ITER_PARAM],
                   seed=params[KMeansClusteringOperation.SEED_PARAM],
                   tol=params[KMeansClusteringOperation.TOLERANCE_PARAM]))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_kmeans_clustering_minimum_success():
    params = {
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    name = "clustering.KMeans"

    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
            from sklearn.cluster import KMeans
            clustering_algo_1 = KMeans(n_clusters=8, init='k-means++',
                    max_iter=300, tol=0.001, random_state=None)
            """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_kmeans_clustering_operation_random_type_failure():
    params = {
        KMeansClusteringOperation.N_CLUSTERS_PARAM: -10,

    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    with pytest.raises(ValueError):
        KMeansClusteringOperation(params, named_inputs={},
                                  named_outputs=named_outputs)

























# def test_clustering_model_operation_success():
#     params = {
#
#         ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',
#         'task_id': 232,
#         'operation_id': 343
#
#     }
#     named_inputs = {'algorithm': 'df_1',
#                     'train input data': 'df_2'}
#     named_outputs = {'output data': 'output_1',
#                      'model': 'output_2'}
#     outputs = ['output_1']
#
#     instance = ClusteringModelOperation(params, named_inputs=named_inputs,
#                                         named_outputs=named_outputs)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("""
#         {algorithm}.setFeaturesCol('{features}')
#         if hasattr(df_1, 'setPredictionCol'):
#             df_1.setPredictionCol('prediction')
#         {model} = {algorithm}.fit({input})
#         setattr({model}, 'features', '{features}')
#
#         # Lazy execution in case of sampling the data in UI
#         def call_transform(df):
#             return output_2.transform(df)
#         output_1 = dataframe_util.LazySparkTransformationDataframe(
#              output_2, df_2, call_transform)
#
#         summary = getattr(output_2, 'summary', None)
#         def call_clusters(df):
#             if hasattr(output_2, 'clusterCenters'):
#                 return spark_session.createDataFrame(
#                     [center.tolist()
#                         for center in output_2.clusterCenters()])
#             else:
#                 return spark_session.createDataFrame([],
#                     types.StructType([]))
#
#         centroids_task_1 = dataframe_util.LazySparkTransformationDataframe(
#             output_2, df_2, call_clusters)
#
#         if summary:
#             summary_rows = []
#             for p in dir(summary):
#                 if not p.startswith('_') and p != "cluster":
#                     try:
#                         summary_rows.append(
#                             [p, getattr(summary, p)])
#                     except Exception as e:
#                         summary_rows.append([p, e.message])
#             summary_content = SimpleTableReport(
#                 'table table-striped table-bordered', [],
#                 summary_rows,
#                 title='Summary')
#             emit_event('update task', status='COMPLETED',
#                 identifier='232',
#                 message=summary_content.generate(),
#                 type='HTML', title='Clustering result',
#                 task={{'id': '{task_id}' }},
#                 operation={{'id': {operation_id} }},
#                 operation_id={operation_id})
#         """.format(algorithm=named_inputs['algorithm'],
#                    input=named_inputs['train input data'],
#                    model=named_outputs['model'],
#                    output=outputs[0],
#                    operation_id=params['operation_id'],
#                    task_id=params['task_id'],
#                    features=params[
#                        ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM]))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_clustering_model_operation_missing_features_failure():
#     params = {}
#     named_inputs = {'algorithm': 'df_1',
#                     'train input data': 'df_2'}
#     named_outputs = {'output data': 'output_1',
#                      'model': 'output_2'}
#
#     with pytest.raises(ValueError):
#         ClusteringModelOperation(params,
#                                  named_inputs=named_inputs,
#                                  named_outputs=named_outputs)
#
#
# def test_clustering_model_operation_missing_input_failure():
#     params = {
#         ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',
#     }
#     named_inputs = {'algorithm': 'df_1'}
#     named_outputs = {'output data': 'output_1', 'model': 'output_2'}
#
#     clustering = ClusteringModelOperation(params,
#                                           named_inputs=named_inputs,
#                                           named_outputs=named_outputs)
#     assert not clustering.has_code
#
#
# def test_clustering_model_operation_missing_output_success():
#     params = {
#         ClusteringModelOperation.FEATURES_ATTRIBUTE_PARAM: 'f',
#     }
#     named_inputs = {'algorithm': 'df_1', 'train input data': 'df_2'}
#     named_outputs = {'model': 'output_2'}
#
#     clustering = ClusteringModelOperation(params,
#                                           named_inputs=named_inputs,
#                                           named_outputs=named_outputs)
#     assert clustering.has_code
#
#
# def test_clustering_operation_success():
#     # This test its not very clear, @CHECK
#     params = {}
#     named_outputs = {'algorithm': 'clustering_algo_1'}
#
#     name = 'BaseClustering'
#     set_values = []
#     instance = ClusteringOperation(params, named_inputs={},
#                                    named_outputs=named_outputs)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("{output} = {name}()".format(
#         output=named_outputs['algorithm'],
#         name=name))
#
#     settings = (['{0}.set{1}({2})'.format(named_outputs['model'], name, v)
#                  for name, v in set_values])
#     settings = "\n".join(settings)
#
#     expected_code += "\n" + settings
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_lda_clustering_operation_optimizer_online_success():
#     params = {
#         LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
#         LdaClusteringOperation.OPTIMIZER_PARAM: 'online',
#         LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
#         LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
#         LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
#     }
#     named_outputs = {'algorithm': 'clustering_algo_1'}
#
#     name = "clustering.LDA"
#
#     set_values = [
#         ['K', params[LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM]],
#         ['MaxIter', params[LdaClusteringOperation.MAX_ITERATIONS_PARAM]],
#         ['Optimizer',
#          "'{}'".format(params[LdaClusteringOperation.OPTIMIZER_PARAM])],
#         ['DocConcentration', [0.25]],
#         ['TopicConcentration',
#          params[LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM]]
#     ]
#
#     instance = LdaClusteringOperation(params, named_inputs={},
#                                       named_outputs=named_outputs)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("{output} = {name}()".format(
#         output=named_outputs['algorithm'],
#         name=name))
#
#     settings = (['{0}.set{1}({2})'.format(
#         named_outputs['algorithm'], name,
#         v if not isinstance(v, list) else json.dumps(v))
#                  for name, v in set_values])
#     settings = "\n".join(settings)
#
#     expected_code += "\n" + settings
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# # def test_lda_clustering_operation_optimizer_em_success():
# #     params = {
# #         LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
# #         LdaClusteringOperation.OPTIMIZER_PARAM: 'em',
# #         LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
# #         LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
# #         LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
# #         LdaClusteringOperation.ONLINE_OPTIMIZER: '',
# #         LdaClusteringOperation.EM_OPTIMIZER: ''
# #
# # }
# # inputs = ['df_1', 'df_2']
# # outputs = ['output_1']
# #
# # instance = LdaClusteringOperation(params, inputs,
# #                                   outputs,
# #                                   named_inputs={},
# #                                   named_outputs={})
# #
# # code = instance.generate_code()
# #
# # expected_code = dedent("""
# #     {input_2}.setLabelCol('{label}').setFeaturesCol('{features}')
# #     {output} = {input_2}.fit({input_1})
# #     """.format(output=outputs[0],
# #                input_1=inputs[0],
# #                input_2=inputs[1],
# #                features=params[ClassificationModel.FEATURES_ATTRIBUTE_PARAM],
# #                label=params[ClassificationModel.LABEL_ATTRIBUTE_PARAM]))
# #
# # result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
# #
# # assert result, msg + debug_ast(code, expected_code)
#
#
# def test_lda_clustering_operation_failure():
#     params = {
#         LdaClusteringOperation.NUMBER_OF_TOPICS_PARAM: 10,
#         LdaClusteringOperation.OPTIMIZER_PARAM: 'xXx',
#         LdaClusteringOperation.MAX_ITERATIONS_PARAM: 10,
#         LdaClusteringOperation.DOC_CONCENTRATION_PARAM: 0.25,
#         LdaClusteringOperation.TOPIC_CONCENTRATION_PARAM: 0.1
#     }
#     named_outputs = {'algorithm': 'clustering_algo_2'}
#     with pytest.raises(ValueError):
#         LdaClusteringOperation(params, named_inputs={},
#                                named_outputs=named_outputs)
#

# def test_gaussian_mixture_clustering_operation_success():
#     params = {
#         GaussianMixtureClusteringOperation.K_PARAM: 10,
#         GaussianMixtureClusteringOperation.MAX_ITERATIONS_PARAM: 10,
#         GaussianMixtureClusteringOperation.TOLERANCE_PARAMETER: 0.001
#     }
#     named_outputs = {'algorithm': 'clustering_algo_1'}
#     name = "clustering.GaussianMixture"
#
#     set_values = [
#         ['MaxIter',
#          params[GaussianMixtureClusteringOperation.MAX_ITERATIONS_PARAM]],
#         ['K', params[GaussianMixtureClusteringOperation.K_PARAM]],
#         ['Tol', params[GaussianMixtureClusteringOperation.TOLERANCE_PARAMETER]],
#     ]
#
#     instance = GaussianMixtureClusteringOperation(params, named_inputs={},
#                                                   named_outputs=named_outputs)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("{output} = {name}()".format(
#         output=named_outputs['algorithm'],
#         name=name))
#
#     settings = (['{0}.set{1}({2})'.format(named_outputs['algorithm'], name, v)
#                  for name, v in set_values])
#     settings = "\n".join(settings)
#
#     expected_code += "\n" + settings
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
#     assert result, msg + format_code_comparison(code, expected_code)
#
