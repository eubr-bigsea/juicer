# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import ast
from textwrap import dedent

import pytest
# Import Operations to test
from juicer.runner import configuration


from tests import compare_ast, format_code_comparison

from juicer.scikit_learn.model_operation import CrossValidationOperation


'''
 CrossValidation tests
'''


def test_crossvalidation_operation_success():
    params = {
        CrossValidationOperation.PREDICTION_ATTRIBUTE_PARAM: 'predict',
        CrossValidationOperation.EVALUATOR_PARAM: 'f1',
        CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: ['label'],
        CrossValidationOperation.FEATURE_ATTRIBUTE_PARAM: ['Feature'],
        CrossValidationOperation.NUM_FOLDS_PARAM: 7,
        CrossValidationOperation.SEED_PARAM: 88,
    }

    n_in = {'input data': 'input_1', 'algorithm': 'algo1'}
    n_out = {'scored data': 'output_1'}
    instance = CrossValidationOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        kf = KFold(n_splits=7, random_state=88,  shuffle=True) 
        X_train = input_1['Feature'].values        
        y = input_1['label'].values                                
                                                                         
        scores = cross_val_score(algo1, X_train.tolist(),            
                                 y.tolist(), cv=kf, scoring='f1')  
                                                                         
        best_score = np.argmax(scores)                                          
                                                                              
        models = None                                                         
        train_index, test_index = list(kf.split(X_train))[best_score]       
        Xf_train, Xf_test = X_train[train_index], X_train[test_index]        
        yf_train, yf_test = y[train_index],  y[test_index]              
        best_model_1 = algo1.fit(Xf_train.tolist(), yf_train.tolist())     
                                                                         
        metric_result = scores[best_score]                          
        output_1 = input_1.copy()                                       
        output_1['predict'] = best_model_1.predict(X_train.tolist())
        models_task_1 = models
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_crossvalidation_operation_models_success():
    params = {
        CrossValidationOperation.PREDICTION_ATTRIBUTE_PARAM: 'predict',
        CrossValidationOperation.EVALUATOR_PARAM: 'f1',
        CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: ['label'],
        CrossValidationOperation.FEATURE_ATTRIBUTE_PARAM: ['Feature'],
        CrossValidationOperation.NUM_FOLDS_PARAM: 7,
        CrossValidationOperation.SEED_PARAM: 88,
    }

    n_in = {'input data': 'input_1', 'algorithm': 'algo1'}
    n_out = {'scored data': 'output_1',
             'models': 'models1', 'best model': 'BestModel'}
    instance = CrossValidationOperation(params, named_inputs=n_in,
                                        named_outputs=n_out)

    code = instance.generate_code()
    expected_code = dedent("""
        kf = KFold(n_splits=7, random_state=88,  shuffle=True) 
        X_train = input_1['Feature'].values        
        y = input_1['label'].values                                

        scores = cross_val_score(algo1, X_train.tolist(),            
                                 y.tolist(), cv=kf, scoring='f1')  

        best_score = np.argmax(scores)                                          

        models = []                                                           
        for train_index, test_index in kf.split(X_train):      
            Xf_train, Xf_test = X_train[train_index], X_train[test_index]   
            yf_train, yf_test = y[train_index],  y[test_index]   
            algo1.fit(Xf_train.tolist(), yf_train.tolist())                         
            models.append(algo1)                 
                               
        BestModel = models[best_score]     

        metric_result = scores[best_score]                          
        output_1 = input_1.copy()                                       
        output_1['predict'] = BestModel.predict(X_train.tolist())
        models1 = models
        """)

    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)


def test_crossvalidation_operation_failure():
    params = {
        CrossValidationOperation.LABEL_ATTRIBUTE_PARAM: ['label'],
        CrossValidationOperation.FEATURE_ATTRIBUTE_PARAM: ['Feature'],
    }
    with pytest.raises(ValueError):
        n_in = {'input data': 'input_1', 'algorithm': 'algo1'}
        n_out = {'scored data': 'output_1'}
        CrossValidationOperation(params, named_inputs=n_in, named_outputs=n_out)


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
