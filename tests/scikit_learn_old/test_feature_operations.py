# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import ast
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.runner import configuration


from tests import compare_ast, format_code_comparison


from juicer.scikit_learn.feature_operation import \
    FeatureAssemblerOperation, \
    MinMaxScalerOperation, \
    MaxAbsScalerOperation, \
    StandardScalerOperation, \
    OneHotEncoderOperation,\
    PCAOperation,\
    QuantileDiscretizerOperation


'''
 FeatureAssembler tests
'''


def test_feature_assembler_operation_success():
    params = {
        FeatureAssemblerOperation.ATTRIBUTES_PARAM: ['col'],
        FeatureAssemblerOperation.ALIAS_PARAM: 'c'
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = FeatureAssemblerOperation(params, named_inputs=n_in,
                                         named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        cols = {cols}
        {output} = {input}
        {output}['{alias}'] = {input}[cols].values.tolist()
        """.format(cols=params[FeatureAssemblerOperation.ATTRIBUTES_PARAM],
                   alias=params[FeatureAssemblerOperation.ALIAS_PARAM],
                   output=out, input=in1))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_feature_assembler_operation_success():
    params = {
        FeatureAssemblerOperation.ATTRIBUTES_PARAM: ['col'],
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = FeatureAssemblerOperation(params, named_inputs=n_in,
                                         named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        cols = {cols}
        {output} = {input}
        {output}['FeatureField'] = {input}[cols].values.tolist()
        """.format(cols=params[FeatureAssemblerOperation.ATTRIBUTES_PARAM],
                   output=out, input=in1))

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_feature_assembler_operation_failure():
    params = {
        FeatureAssemblerOperation.ALIAS_PARAM: 'c'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        FeatureAssemblerOperation(params, named_inputs=n_in,
                                  named_outputs=n_out)


'''
    Min-Max Scaler tests
'''


def test_minmaxscaler_operation_success():
    params = {
        MinMaxScalerOperation.ALIAS_PARAM: 'result',
        MinMaxScalerOperation.ATTRIBUTE_PARAM: ['col_1'],
        MinMaxScalerOperation.MAX_PARAM: 2,
        MinMaxScalerOperation.MIN_PARAM: -2
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = MinMaxScalerOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-2,2))
        X_train = input_1['col_1'].values.tolist()
        output_1['result'] = scaler.fit_transform(X_train).tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_minmaxscaler_minimum_operation_success():
    params = {
        MinMaxScalerOperation.ATTRIBUTE_PARAM: ['col'],
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = MinMaxScalerOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train = input_1['col'].values.tolist()
        output_1['col_norm'] = scaler.fit_transform(X_train).tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_minmaxscaler_operation_failure():
    params = {
        MinMaxScalerOperation.ALIAS_PARAM: 'c'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        MinMaxScalerOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Max-Abs Scaler tests
'''


def test_maxabsscaler_operation_success():
    params = {
        MaxAbsScalerOperation.ALIAS_PARAM: 'result',
        MaxAbsScalerOperation.ATTRIBUTE_PARAM: ['col_1']
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = MaxAbsScalerOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        X_train = input_1['col_1'].values.tolist()
        output_1['result'] = scaler.fit_transform(X_train).tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_maxabsscaler_minimum_operation_success():
    params = {
        MaxAbsScalerOperation.ATTRIBUTE_PARAM: ['col'],
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = MaxAbsScalerOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        X_train = input_1['col'].values.tolist()
        output_1['col_norm'] = scaler.fit_transform(X_train).tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_maxabsscaler_operation_failure():
    params = {
        MaxAbsScalerOperation.ALIAS_PARAM: 'c'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        MaxAbsScalerOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Standard Scaler tests
'''


def test_standardscaler_operation_success():
    params = {
        StandardScalerOperation.ALIAS_PARAM: 'result',
        StandardScalerOperation.ATTRIBUTE_PARAM: ['col_1']
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = StandardScalerOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = input_1['col_1'].values.tolist()
        output_1['result'] = scaler.fit_transform(X_train).tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_standardscaler_minimum_operation_success():
    params = {
        StandardScalerOperation.ATTRIBUTE_PARAM: ['col'],
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = StandardScalerOperation(params, named_inputs=n_in,
                                       named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = input_1['col'].values.tolist()
        output_1['col_norm'] = scaler.fit_transform(X_train).tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_standardscaler_operation_failure():
    params = {
        StandardScalerOperation.ALIAS_PARAM: 'c'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        StandardScalerOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    OneHot Encoder tests
'''


def test_onehot_encoder_operation_success():
    params = {
        OneHotEncoderOperation.ALIAS_PARAM: 'result',
        OneHotEncoderOperation.ATTRIBUTE_PARAM: ['col_1']
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = OneHotEncoderOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        X_train = input_1['col_1'].values.tolist()
        output_1['result'] = enc.fit_transform(X_train).toarray().tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_onehot_encoder_minimum_operation_success():
    params = {
        OneHotEncoderOperation.ATTRIBUTE_PARAM: ['col'],
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = OneHotEncoderOperation(params, named_inputs=n_in,
                                      named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder()
        X_train = input_1['col'].values.tolist()
        output_1['col_norm'] = enc.fit_transform(X_train).toarray().tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_onehot_encoder_operation_failure():
    params = {
        OneHotEncoderOperation.ALIAS_PARAM: 'c'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        OneHotEncoderOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
 PCA Operation tests
'''


def test_pca_operation_success():
    params = {
        PCAOperation.ATTRIBUTE_PARAM: ['col'],
        PCAOperation.ALIAS_PARAM: 'feature',
        PCAOperation.N_COMPONENTS: 3
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = PCAOperation(params, named_inputs=n_in, named_outputs=n_out)

    code = instance.generate_code()

    expected_code = dedent("""
        output_1 = input_1
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X_train = input_1['col'].values.tolist()
        output_1['feature'] = pca.fit_transform(X_train).tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_pca_operation_failure():
    params = {
        PCAOperation.N_COMPONENTS: -1
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        PCAOperation(params, named_inputs=n_in, named_outputs=n_out)


'''
    Quantile Discretizer tests
'''


def test_quantile_discretizer_operation_success():
    params = {
        QuantileDiscretizerOperation.ALIAS_PARAM: 'result',
        QuantileDiscretizerOperation.ATTRIBUTE_PARAM: ['col_1'],
        QuantileDiscretizerOperation.DISTRIBUITION_PARAM:
            QuantileDiscretizerOperation.DISTRIBUITION_PARAM_NORMAL,
        QuantileDiscretizerOperation.SEED_PARAM: 19,
        QuantileDiscretizerOperation.N_QUANTILES_PARAM: 500
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}
    in1 = n_in['input data']
    out = n_out['output data']

    instance = QuantileDiscretizerOperation(params, named_inputs=n_in,
                                            named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=500,
            output_distribution='normal', random_state=19)
        X_train = input_1['col_1'].values.tolist()
        output_1['result'] = qt.fit_transform(X_train).toarray().tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_quantile_discretizer_minimum_operation_success():
    params = {
        QuantileDiscretizerOperation.ATTRIBUTE_PARAM: ['col'],
    }

    n_in = {'input data': 'input_1'}
    n_out = {'output data': 'output_1'}

    instance = QuantileDiscretizerOperation(params, named_inputs=n_in,
                                            named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        output_1 = input_1
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=1000,
            output_distribution='uniform', random_state=None)
        X_train = input_1['col'].values.tolist()
        output_1['col_norm'] = qt.fit_transform(X_train).toarray().tolist()
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))

    assert result, msg + format_code_comparison(code, expected_code)


def test_quantile_discretizer_operation_failure():
    params = {
        QuantileDiscretizerOperation.ALIAS_PARAM: 'c'
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1'}
        n_out = {'output data': 'output_1'}
        QuantileDiscretizerOperation(params, named_inputs=n_in,
                                     named_outputs=n_out)



'''
    FeatureIndexer tests
'''


# def test_feature_indexer_operation_success():
#     params = {
#         FeatureIndexerOperation.TYPE_PARAM: 'string',
#         FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
#         FeatureIndexerOperation.ALIAS_PARAM: 'c',
#         FeatureIndexerOperation.MAX_CATEGORIES_PARAM: '20',
#     }
#     n_in = {'input data': 'input_1'}
#     n_out = {'output data': 'output_1'}
#     in1 = n_in['input data']
#     out = n_out['output data']
#
#     instance = FeatureIndexerOperation(params, named_inputs=n_in,
#                                        named_outputs=n_out)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("""
#         col_alias = dict(tuple({alias}))
#         indexers = [feature.StringIndexer(inputCol=col, outputCol=alias,
#                             handleInvalid='skip')
#                     for col, alias in col_alias.items()]
#
#         # Use Pipeline to process all attributes once
#         pipeline = Pipeline(stages=indexers)
#         models_task_1 = dict([(c, indexers[i].fit({in1}))
#                   for i, c in enumerate(col_alias.values())])
#
#         # labels = [model.labels for model in models.itervalues()]
#         # Spark ML 2.0.1 do not deal with null in indexer.
#         # See SPARK-11569
#
#         # input_1_without_null = input_1.na.fill('NA', subset=col_alias.keys())
#         {in1}_without_null = {in1}.na.fill('NA', subset=col_alias.keys())
#         {out} = pipeline.fit({in1}_without_null).transform({in1}_without_null)
#
#         if 'indexer' not in cached_state:
#             cached_state['indexers'] = {{}}
#         for name, model in models_task_1.items():
#             cached_state['indexers'][name] = model
#
#         """.format(attr=params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
#                    in1=in1,
#                    out=out,
#                    alias=json.dumps(
#                        zip(params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
#                            params[FeatureIndexerOperation.ALIAS_PARAM]))))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_feature_indexer_string_type_param_operation_failure():
#     params = {
#         FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
#         FeatureIndexerOperation.TYPE_PARAM: 'XxX',
#         FeatureIndexerOperation.ALIAS_PARAM: 'c',
#         FeatureIndexerOperation.MAX_CATEGORIES_PARAM: '20',
#     }
#
#     with pytest.raises(ValueError):
#         n_in = {'input data': 'input_1'}
#         n_out = {'output data': 'output_1'}
#
#         indexer = FeatureIndexerOperation(params, named_inputs=n_in,
#                                           named_outputs=n_out)
#         indexer.generate_code()
#
#
# def test_feature_indexer_string_missing_attribute_param_operation_failure():
#     params = {
#         FeatureIndexerOperation.TYPE_PARAM: 'string',
#         FeatureIndexerOperation.ALIAS_PARAM: 'c',
#         FeatureIndexerOperation.MAX_CATEGORIES_PARAM: '20',
#     }
#
#     with pytest.raises(ValueError):
#         n_in = {'input data': 'input_1'}
#         n_out = {'output data': 'output_1'}
#         FeatureIndexerOperation(params, named_inputs=n_in,
#                                 named_outputs=n_out)
#
#
# def test_feature_indexer_vector_missing_attribute_param_operation_failure():
#     params = {
#         FeatureIndexerOperation.TYPE_PARAM: 'string',
#         FeatureIndexerOperation.ALIAS_PARAM: 'c',
#         FeatureIndexerOperation.MAX_CATEGORIES_PARAM: '20',
#     }
#     with pytest.raises(ValueError):
#         n_in = {'input data': 'input_1'}
#         n_out = {'output data': 'output_1'}
#         FeatureIndexerOperation(params, named_inputs=n_in,
#                                 named_outputs=n_out)
#
#
# def test_feature_indexer_vector_operation_success():
#     params = {
#
#         FeatureIndexerOperation.TYPE_PARAM: 'vector',
#         FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
#         FeatureIndexerOperation.ALIAS_PARAM: 'c',
#         FeatureIndexerOperation.MAX_CATEGORIES_PARAM: 20,
#     }
#
#     n_in = {'input data': 'input_1'}
#     n_out = {'output data': 'output_1'}
#     in1 = n_in['input data']
#     out = n_out['output data']
#
#     instance = FeatureIndexerOperation(params, named_inputs=n_in,
#                                        named_outputs=n_out)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("""
#             col_alias = dict({3})
#             indexers = [feature.VectorIndexer(maxCategories={4},
#                             inputCol=col, outputCol=alias)
#                             for col, alias in col_alias.items()]
#
#             # Use Pipeline to process all attributes once
#             pipeline = Pipeline(stages=indexers)
#             models_task_1 = dict([(col, indexers[i].fit({1})) for i, col in
#                         enumerate(col_alias.values())])
#             labels = None
#
#             # Spark ML 2.0.1 do not deal with null in indexer.
#             # See SPARK-11569
#             {1}_without_null = {1}.na.fill('NA', subset=col_alias.keys())
#
#             {2} = pipeline.fit({1}_without_null).transform({1}_without_null)
#             if 'indexer' not in cached_state:
#                 cached_state['indexers'] = {{}}
#             for name, model in models_task_1.items():
#                 cached_state['indexers'][name] = model
#
#             """.format(params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
#                        in1,
#                        out,
#                        json.dumps(
#                            zip(params[FeatureIndexerOperation.ATTRIBUTES_PARAM],
#                                params[FeatureIndexerOperation.ALIAS_PARAM])),
#                        params[FeatureIndexerOperation.MAX_CATEGORIES_PARAM]))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_feature_indexer_vector_operation_failure():
#     params = {
#
#         FeatureIndexerOperation.TYPE_PARAM: 'vector',
#         FeatureIndexerOperation.ATTRIBUTES_PARAM: ['col'],
#         FeatureIndexerOperation.ALIAS_PARAM: 'c',
#         FeatureIndexerOperation.MAX_CATEGORIES_PARAM: -1,
#     }
#
#     with pytest.raises(ValueError):
#         n_in = {'input data': 'input_1'}
#         n_out = {'output data': 'output_1'}
#         FeatureIndexerOperation(params, named_inputs=n_in,
#                                 named_outputs=n_out)
#
#
# def test_feature_indexer_operation_failure():
#     params = {}
#     with pytest.raises(ValueError):
#         n_in = {'input data': 'input_1'}
#         n_out = {'output data': 'output_1'}
#         FeatureIndexerOperation(params, named_inputs=n_in,
#                                 named_outputs=n_out)
#

#
# '''
#  ApplyModel tests
# '''
#
#
# def test_apply_model_operation_success():
#     params = {}
#
#     n_in = {'input data': 'input_1', 'model': 'model1'}
#     n_out = {'output data': 'output_1', 'model': 'model1'}
#
#     in1 = n_in['input data']
#     model = n_in['model']
#     out = n_out['output data']
#
#     instance = ApplyModelOperation(params, named_inputs=n_in,
#                                    named_outputs=n_out)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("""
#     # params = {{'predictionCol': 'prediction'}}
#     params = {{}}
#     {output_1} = {model}.transform({input_1}, params)
#     """.format(
#         output_1=out, input_1=in1, model=model))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# '''
#     EvaluateModel tests
# '''
#
#
# @pytest.mark.skip
# def test_evaluate_model_operation_success():
#     params = {
#
#         EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'c',
#         EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: 'c',
#         EvaluateModelOperation.METRIC_PARAM: 'f1',
#         'task_id': '2323-afffa-343bdaff',
#         'workflow_id': 203,
#         'workflow_name': 'test',
#         'job_id': 34,
#         'user': {
#             'id': 12,
#             'name': 'admin',
#             'login': 'admin'
#         },
#         'operation_id': 2793,
#         'task': {
#             'forms': {}
#         }
#     }
#     configuration.set_config(
#         {
#             'juicer': {
#                 'services': {
#                     'limonero': {
#                         'url': 'http://localhost',
#                         'auth_token': 'FAKE',
#                     },
#                     'caipirinha': {
#                         'url': 'http://localhost',
#                         'auth_token': 'FAKE',
#                         'storage_id': 343
#                     }
#                 }
#             }
#         }
#     )
#     n_in = {'input data': 'input_1', 'model': 'df_model'}
#     n_out = {'metric': 'metric_value', 'evaluator': 'df_evaluator'}
#     instance = EvaluateModelOperation(params, named_inputs=n_in,
#                                       named_outputs=n_out)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("""
#         metric_value = 0.0
#         display_text = True
#         display_image = True
#
#         metric = '{metric}'
#         if metric in ['areaUnderROC', 'areaUnderPR']:
#             evaluator = evaluation.BinaryClassificationEvaluator(
#                 predictionCol='{predic_atr}',
#                 labelCol='{label_atr}',
#                 metricName=metric)
#             metric_value = evaluator.evaluate({input_1})
#             if display_text:
#                 result = '<h4>{{}}: {{}}</h4>'.format('{metric}',
#                     metric_value)
#
#                 emit_event(
#                    'update task', status='COMPLETED',
#                    identifier='{task_id}',
#                    message=result,
#                    type='HTML', title='Evaluation result',
#                    task={{'id': '{task_id}'}},
#                    operation={{'id': {operation_id}}},
#                    operation_id={operation_id})
#
#         elif metric in ['f1', 'weightedPrecision', 'weightedRecall',
#                 'accuracy']:
#             label_prediction = input_1.select(
#                 '{label_atr}', '{predic_atr}')
#             evaluator = MulticlassMetrics(label_prediction.rdd)
#             if metric == 'f1':
#                 metric_value = evaluator.weightedFMeasure()
#             elif metric == 'weightedPrecision':
#                 metric_value = evaluator.weightedPrecision
#             elif metric == 'weightedRecall':
#                 metric_value = evaluator.weightedRecall
#             elif metric == 'accuracy':
#                 metric_value = evaluator.accuracy
#
#             if display_image:
#                 classes = ['c: {{}}'.format(x[0]) for x in
#                     label_prediction.select(
#                         'c').distinct().sort(
#                             'c', ascending=True).collect()]
#
#                 content = ConfusionMatrixImageReport(
#                     cm=evaluator.confusionMatrix().toArray(),
#                     classes=classes,)
#
#                 emit_event(
#                    'update task', status='COMPLETED',
#                    identifier='{task_id}',
#                    message=content.generate(),
#                    type='IMAGE', title='Evaluation result',
#                    task={{'id': '{task_id}'}},
#                    operation={{'id': {operation_id}}},
#                    operation_id={operation_id})
#
#
#             if display_text:
#                 headers = ['Metric', 'Value']
#                 rows = [
#                     ['F1', evaluator.weightedFMeasure()],
#                     ['Weighted Precision', evaluator.weightedPrecision],
#                     ['Weighted Recall', evaluator.weightedRecall],
#                     ['Accuracy', evaluator.accuracy],
#                 ]
#
#                 content = SimpleTableReport(
#                         'table table-striped table-bordered table-sm',
#                         headers, rows, title='Evaluation result')
#
#                 emit_event(
#                    'update task', status='COMPLETED',
#                    identifier='{task_id}',
#                    message=content.generate(),
#                    type='HTML', title='Evaluation result',
#                    task={{'id': '{task_id}'}},
#                    operation={{'id': {operation_id}}},
#                    operation_id={operation_id})
#
#         from juicer.spark.ml_operation import ModelsEvaluationResultList
#         model_task_1 = ModelsEvaluationResultList(
#             [df_model], df_model, '{metric}', metric_value)
#         f1 = metric_value
#         model_task_1 = None
#             """.format(output=n_out['metric'],
#                        evaluator_out=n_out['evaluator'],
#                        input_2=n_in['model'],
#                        input_1=n_in['input data'],
#                        task_id=params['task_id'],
#                        operation_id=params['operation_id'],
#                        predic_atr=params[
#                            EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM],
#                        label_atr=params[
#                            EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM],
#                        metric=params[EvaluateModelOperation.METRIC_PARAM],
#                        evaluator=EvaluateModelOperation.METRIC_TO_EVALUATOR[
#                            params[EvaluateModelOperation.METRIC_PARAM]][0],
#                        predic_col=EvaluateModelOperation.METRIC_TO_EVALUATOR[
#                            params[EvaluateModelOperation.METRIC_PARAM]][1]))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#     assert result, msg + format_code_comparison(code,
#                                                 expected_code)
#
#
# def test_evaluate_model_operation_wrong_metric_param_failure():
#     params = {
#
#         EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'c',
#         EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: 'c',
#         EvaluateModelOperation.METRIC_PARAM: 'mist',
#     }
#     n_in = {'input data': 'input_1', 'model': 'df_model'}
#     n_out = {'metric': 'df_metric', 'evaluator': 'df_evaluator'}
#     with pytest.raises(ValueError):
#         EvaluateModelOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#
# def test_evaluate_model_operation_missing_metric_param_failure():
#     params = {
#
#         EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'c',
#         EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: 'c',
#         EvaluateModelOperation.METRIC_PARAM: '',
#     }
#     n_in = {'input data': 'input_1', 'model': 'df_model'}
#     n_out = {'metric': 'df_metric', 'evaluator': 'df_evaluator'}
#     with pytest.raises(ValueError):
#         EvaluateModelOperation(params, named_inputs=n_in,
#                                named_outputs=n_out)
#
#
# def test_evaluate_model_operation_missing_prediction_attribute_failure():
#     params = {
#
#         EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: '',
#         EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: 'c',
#         EvaluateModelOperation.METRIC_PARAM: 'f1',
#     }
#     n_in = {'input data': 'input_1', 'model': 'df_model'}
#     n_out = {'metric': 'df_metric', 'evaluator': 'df_evaluator'}
#     with pytest.raises(ValueError):
#         EvaluateModelOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#
# def test_evaluate_model_operation_missing_label_attribute_failure():
#     params = {
#
#         EvaluateModelOperation.PREDICTION_ATTRIBUTE_PARAM: 'c',
#         EvaluateModelOperation.LABEL_ATTRIBUTE_PARAM: '',
#         EvaluateModelOperation.METRIC_PARAM: 'f1',
#     }
#     n_in = {'input data': 'input_1', 'model': 'df_model'}
#     n_out = {'metric': 'df_metric', 'evaluator': 'df_evaluator'}
#     with pytest.raises(ValueError):
#         EvaluateModelOperation(params, named_inputs=n_in, named_outputs=n_out)
#
#
# '''
#     CrossValidationOperation Tests
# '''
#
#
# def test_cross_validation_partial_operation_success():
#     params = {
#         'task_id': 232,
#         'operation_id': 1,
#         CrossValidationOperation.NUM_FOLDS_PARAM: 3,
#         CrossValidationOperation.EVALUATOR_PARAM: 'accuracy',
#         CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: ['label'],
#
#     }
#     n_in = {'algorithm': 'df_1', 'input data': 'df_2', 'evaluator': 'xpto'}
#     n_out = {'scored data': 'output_1', 'evaluation': 'eval_1'}
#     outputs = ['output_1']
#
#     instance = CrossValidationOperation(params, named_inputs=n_in,
#                                         named_outputs=n_out, )
#
#     code = instance.generate_code()
#
#     expected_code = dedent("""
#             grid_builder = tuning.ParamGridBuilder()
#             estimator, param_grid, metric = {algorithm}
#
#             for param_name, values in param_grid.items():
#                 param = getattr(estimator, param_name)
#                 grid_builder.addGrid(param, values)
#
#             evaluator = evaluation.MulticlassClassificationEvaluator(
#                 predictionCol='prediction',
#                 labelCol='{label}',
#                 metricName='{metric}')
#
#             estimator.setLabelCol('{label}')
#             estimator.setPredictionCol('prediction')
#
#             cross_validator = tuning.CrossValidator(
#                  estimator=estimator,
#                  estimatorParamMaps=grid_builder.build(),
#                  evaluator=evaluator, numFolds=3)
#
#
#             cv_model = cross_validator.fit({input_data})
#             fit_data = cv_model.transform({input_data})
#             best_model_1  = cv_model.bestModel
#             metric_result = evaluator.evaluate(fit_data)
#             # {evaluation} = metric_result
#             {output} = fit_data
#             models_task_1 = None
#
#             """.format(
#         algorithm=n_in['algorithm'],
#         input_data=n_in['input data'],
#         evaluator=n_in['evaluator'],
#         evaluation='eval_1',
#         output=outputs[0],
#         metric=params[CrossValidationOperation.EVALUATOR_PARAM],
#         label=params[CrossValidationOperation.LABEL_ATTRIBUTE_PARAM][0],
#         folds=params[CrossValidationOperation.NUM_FOLDS_PARAM]))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_cross_validation_complete_operation_success():
#     params = {
#
#         CrossValidationOperation.NUM_FOLDS_PARAM: 3,
#         'task_id': '2323-afffa-343bdaff',
#         'operation_id': 2793,
#         CrossValidationOperation.EVALUATOR_PARAM: 'weightedRecall',
#         CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: ['label']
#
#     }
#     n_in = {'algorithm': 'algo1', 'input data': 'df_1', 'evaluator': 'ev_1'}
#     n_out = {'evaluation': 'output_1', 'scored data': 'output_1'}
#     outputs = ['output_1']
#
#     instance = CrossValidationOperation(
#         params, named_inputs=n_in, named_outputs=n_out)
#
#     code = instance.generate_code()
#
#     expected_code = dedent("""
#             grid_builder = tuning.ParamGridBuilder()
#             estimator, param_grid, metric = {algorithm}
#
#             for param_name, values in param_grid.items():
#                 param = getattr(estimator, param_name)
#                 grid_builder.addGrid(param, values)
#
#             evaluator = evaluation.MulticlassClassificationEvaluator(
#                 predictionCol='prediction',
#                 labelCol='{label}',
#                 metricName='{metric}')
#
#             estimator.setLabelCol('{label}')
#             estimator.setPredictionCol('prediction')
#
#             cross_validator = tuning.CrossValidator(
#                 estimator=estimator, estimatorParamMaps=grid_builder.build(),
#                 evaluator=evaluator, numFolds={folds})
#             cv_model = cross_validator.fit({input_data})
#             fit_data = cv_model.transform({input_data})
#             best_model_1  = cv_model.bestModel
#             metric_result = evaluator.evaluate(fit_data)
#             # {output} = metric_result
#             {output} = fit_data
#             models_task_1 = None
#
#             """.format(
#         algorithm=n_in['algorithm'],
#         input_data=n_in['input data'],
#         evaluator=n_in['evaluator'],
#         output=outputs[0],
#         label=params[CrossValidationOperation.LABEL_ATTRIBUTE_PARAM][0],
#         metric=params[CrossValidationOperation.EVALUATOR_PARAM],
#         folds=params[CrossValidationOperation.NUM_FOLDS_PARAM]))
#
#     result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
#
#     assert result, msg + format_code_comparison(code, expected_code)
#
#
# def test_cross_validation_complete_operation_missing_input_failure():
#     params = {
#         CrossValidationOperation.EVALUATOR_PARAM: 'f1',
#         CrossValidationOperation.NUM_FOLDS_PARAM: 3,
#         CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: 'label'
#
#     }
#     n_in = {'algorithm': 'algo1'}
#     n_out = {'evaluation': 'output_1'}
#
#     instance = CrossValidationOperation(params, named_inputs=n_in,
#                                         named_outputs=n_out)
#     assert not instance.has_code
#
#
