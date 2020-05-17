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
    AgglomerativeClusteringOperation,\
    ClusteringModelOperation, \
    DBSCANClusteringOperation,\
    LdaClusteringOperation, KMeansClusteringOperation, \
    GaussianMixtureClusteringOperation


"""
    Agglomerative Clustering tests
"""


def test_agglomerative_clustering_success():
    params = {
        AgglomerativeClusteringOperation.ALIAS_PARAM: 'ALIAS',
        AgglomerativeClusteringOperation.FEATURES_PARAM: ['f'],
        AgglomerativeClusteringOperation.AFFINITY_PARAM:
            AgglomerativeClusteringOperation.AFFINITY_PARAM_COS,
        AgglomerativeClusteringOperation.LINKAGE_PARAM:
            AgglomerativeClusteringOperation.AFFINITY_PARAM_L2
    }

    named_inputs = {'input data': 'df1'}
    named_outputs = {'output data': 'df2'}

    instance = AgglomerativeClusteringOperation(params,
                                                named_inputs=named_inputs,
                                                named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        df2 = df1.copy()

        X = df2['f'].values.tolist()
        clt = AgglomerativeClustering(n_clusters=2, 
            linkage='l2', affinity='cosine')
        df2['ALIAS'] = clt.fit_predict(X)
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_agglomerative_clustering_minimum_success():
    params = {
        AgglomerativeClusteringOperation.FEATURES_PARAM: ['f'],
    }
    named_inputs = {'input data': 'df1'}
    named_outputs = {'output data': 'df2'}

    instance = AgglomerativeClusteringOperation(params,
                                                named_inputs=named_inputs,
                                                named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        df2 = df1.copy()

        X = df2['f'].values.tolist()
        clt = AgglomerativeClustering(n_clusters=2,
             linkage='ward', affinity='euclidean')
        df2['cluster'] = clt.fit_predict(X)
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_agglomerative_clustering_failure():
    params = {

    }
    named_inputs = {'input data': 'df1'}
    named_outputs = {'output data': 'df2'}

    with pytest.raises(ValueError):
        AgglomerativeClusteringOperation(params, named_inputs=named_inputs,
                                         named_outputs=named_outputs)


"""
    DBSCAN tests
"""


def test_dbscan_clustering_success():
    params = {
        DBSCANClusteringOperation.FEATURES_PARAM: ['f'],
        DBSCANClusteringOperation.ALIAS_PARAM: 'alias',
        DBSCANClusteringOperation.EPS_PARAM: 0.15,
        DBSCANClusteringOperation.MIN_SAMPLES_PARAM: 20,

    }

    named_inputs = {'input data': 'df1'}
    named_outputs = {'output data': 'df2'}

    instance = DBSCANClusteringOperation(params, named_inputs=named_inputs,
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        df2 = df1.copy()
         
        X = df2['f'].values.tolist()
        clt = DBSCAN(eps=0.15, min_samples=20)
        df2['alias'] = clt.fit_predict(X)
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_dbscan_clustering_minimum_success():
    params = {
        DBSCANClusteringOperation.FEATURES_PARAM: ['f'],
    }
    named_inputs = {'input data': 'df1'}
    named_outputs = {'output data': 'df2'}

    instance = DBSCANClusteringOperation(params, named_inputs=named_inputs,
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        df2 = df1.copy()
         
        X = df2['f'].values.tolist()
        clt = DBSCAN(eps=0.5, min_samples=5)
        df2['cluster'] = clt.fit_predict(X)
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_dbscan_clustering_failure():
    params = {

    }
    named_inputs = {'input data': 'df1'}
    named_outputs = {'output data': 'df2'}

    with pytest.raises(ValueError):
        AgglomerativeClusteringOperation(params, named_inputs=named_inputs,
                                         named_outputs=named_outputs)


"""
    Gaussian Mixture tests
"""


def test_gaussian_mixture_clustering_success():
    params = {
        GaussianMixtureClusteringOperation.MAX_ITER_PARAM: 15,
        GaussianMixtureClusteringOperation.TOLERANCE_PARAM: 0.11,
        GaussianMixtureClusteringOperation.N_CLUSTERS_PARAM: 11,

    }

    named_outputs = {'algorithm': 'clustering_algo_1'}

    instance = GaussianMixtureClusteringOperation(params, named_inputs={},
                                                  named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        clustering_algo_1 = GaussianMixture(n_components=11, 
            max_iter=15, tol=0.11)
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_gaussian_mixture_clustering_minimum_success():
    params = {

    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    instance = GaussianMixtureClusteringOperation(params, named_inputs={},
                                                  named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        clustering_algo_1 = GaussianMixture(n_components=1, 
        max_iter=100, tol=0.001)
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_gaussian_mixture_clustering_failure():
    params = {
        GaussianMixtureClusteringOperation.N_CLUSTERS_PARAM: -10
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    with pytest.raises(ValueError):
        GaussianMixtureClusteringOperation(params, named_inputs={},
                                           named_outputs=named_outputs)


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

    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
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

    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
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

    instance = KMeansClusteringOperation(params, named_inputs={},
                                         named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
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


"""
    LDA Clustering tests
"""


def test_lda_clustering_success():
    params = {
        LdaClusteringOperation.N_COMPONENTES_PARAM: 10,
        LdaClusteringOperation.ALPHA_PARAM: 0.5,
        LdaClusteringOperation.SEED_PARAM: 11,
        LdaClusteringOperation.MAX_ITER_PARAM: 100,
        LdaClusteringOperation.ETA_PARAM: 0.5,
        LdaClusteringOperation.LEARNING_METHOD_PARAM:
            LdaClusteringOperation.LEARNING_METHOD_ON,

    }

    named_outputs = {'algorithm': 'clustering_algo_1'}

    instance = LdaClusteringOperation(params, named_inputs={},
                                      named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        clustering_algo_1 = LatentDirichletAllocation(n_components=10, 
         doc_topic_prior=0.5, topic_word_prior=0.5, 
         learning_method='online', max_iter=100)
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_lda_clustering_minimum_success():
    params = {

    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    instance = LdaClusteringOperation(params, named_inputs={},
                                      named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        clustering_algo_1 = LatentDirichletAllocation(n_components=10, 
        doc_topic_prior=None, topic_word_prior=None, 
        learning_method='online', max_iter=10)
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_lda_clustering_failure():
    params = {
        LdaClusteringOperation.N_COMPONENTES_PARAM: -10
    }
    named_outputs = {'algorithm': 'clustering_algo_1'}

    with pytest.raises(ValueError):
        LdaClusteringOperation(params, named_inputs={},
                               named_outputs=named_outputs)


"""
    Clustering Model tests
"""


def test_clustering_operation_success():
    params = {

        ClusteringModelOperation.FEATURES_PARAM: ['f'],

    }
    named_inputs = {'algorithm': 'algo',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1'}

    instance = ClusteringModelOperation(params, named_inputs=named_inputs,
                                        named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        X = df_2['f'].values.tolist()
        model_output_1 = algo.fit(X)
         
        y = algo.predict(X)
        output_1 = df_2
        output_1['prediction'] = y
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_clustering_with_model_operation_success():
    params = {

        ClusteringModelOperation.FEATURES_PARAM: ['f'],

    }
    named_inputs = {'algorithm': 'algo',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}

    instance = ClusteringModelOperation(params, named_inputs=named_inputs,
                                        named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        X = df_2['f'].values.tolist()
        output_2 = algo.fit(X)

        y = algo.predict(X)
        output_1 = df_2
        output_1['prediction'] = y
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_clustering_model_operation_success():
    params = {

        ClusteringModelOperation.FEATURES_PARAM: ['f'],

    }
    named_inputs = {'algorithm': 'algo',
                    'train input data': 'df_2'}
    named_outputs = {'model': 'output_2'}

    instance = ClusteringModelOperation(params, named_inputs=named_inputs,
                                        named_outputs=named_outputs)

    code = instance.generate_code()

    expected_code = dedent("""
        X = df_2['f'].values.tolist()
        output_2 = algo.fit(X)
        
        task_1 = None
        """.format())

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_clustering_model_operation_missing_features_failure():
    params = {}
    named_inputs = {'algorithm': 'df_1',
                    'train input data': 'df_2'}
    named_outputs = {'output data': 'output_1',
                     'model': 'output_2'}

    with pytest.raises(ValueError):
        ClusteringModelOperation(params,
                                 named_inputs=named_inputs,
                                 named_outputs=named_outputs)


def test_clustering_model_operation_missing_input_failure():
    params = {
        ClusteringModelOperation.FEATURES_PARAM: ['f']
    }
    named_inputs = {'algorithm': 'df_1'}
    named_outputs = {'output data': 'output_1', 'model': 'output_2'}

    clustering = ClusteringModelOperation(params,
                                          named_inputs=named_inputs,
                                          named_outputs=named_outputs)
    assert not clustering.has_code


def test_clustering_model_operation_missing_output_success():
    params = {
        ClusteringModelOperation.FEATURES_PARAM: ['f']
    }
    named_inputs = {'algorithm': 'df_1', 'train input data': 'df_2'}
    named_outputs = {'model': 'output_2'}

    clustering = ClusteringModelOperation(params,
                                          named_inputs=named_inputs,
                                          named_outputs=named_outputs)
    assert clustering.has_code

