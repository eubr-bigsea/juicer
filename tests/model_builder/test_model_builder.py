import ast
from io import StringIO
import sys
from textwrap import dedent
from juicer.meta.meta_minion import MetaMinion
from juicer.meta.transpiler import (
    ModelBuilderTemplateParams as ModelBuilderParams)
from juicer.workflow.workflow import Workflow
from tests import compare_ast, format_code_comparison
from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SortOperation
import pytest
from juicer.transpiler import GenerateCodeParams
from mock import patch, MagicMock
from .fixtures import *  # Must be * in order to import fixtures
import hashlib

# region Test Sample Operation


def test_sample_percent_success(builder_params: ModelBuilderParams):
    frac = 0.5
    seed = 777
    builder_params.sample.parameters['fraction'] = frac
    builder_params.sample.parameters['seed'] = seed  # Must be set
    builder_params.sample.parameters['type'] = 'percent'
    assert (
        f'df = df.sample(withReplacement=False, fraction={frac}, seed={seed})'
        == builder_params.sample.model_builder_code())


def test_sample_head_success(builder_params: ModelBuilderParams):
    n = 120
    builder_params.sample.parameters['value'] = n
    builder_params.sample.parameters['type'] = 'head'
    assert f'df = df.limit({n})' == builder_params.sample.model_builder_code()


def test_sample_fixed_number_success(builder_params: ModelBuilderParams):
    n = 300
    seed = 123
    builder_params.sample.parameters['value'] = n
    builder_params.sample.parameters['seed'] = seed
    builder_params.sample.parameters['type'] = 'value'
    assert (f'df = df.sample(False, fraction={n}/df.count(), seed={seed})' ==
            builder_params.sample.model_builder_code())


def test_sample_no_type_informed_failure(builder_params: ModelBuilderParams):
    builder_params.sample.parameters['type'] = None
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert "Parameter 'type' must be informed" in str(ve)


def test_sample_invalid_type_informed_failure(builder_params: ModelBuilderParams):
    builder_params.sample.parameters['type'] = 'invalid value'
    with pytest.raises(ValueError) as ve:
        builder_params.sample.model_builder_code()
    assert "Invalid value for parameter 'type'" in str(ve)

# endregion

# region Test Split Operation


def test_split(builder_params: ModelBuilderParams):
    # for k in ['strategy', 'seed', 'ratio']:
    #    print(k, builder_params.split.parameters.get(k))

    builder_params.split.strategy = 'cross_validation'
    builder_params.split.seed = 302324
    builder_params.split.ratio = .7
    # Test method parameter:
    # - split:
    # - cross_validation: Not implemented :(

    print(builder_params.split.model_builder_code())


def test_split_split_strategy_success(builder_params: ModelBuilderParams):
    ratio = .8
    seed = 232
    builder_params.split.strategy = 'split'
    builder_params.split.seed = seed
    builder_params.split.ratio = ratio

    code = builder_params.split.model_builder_code()
    expected_code = dedent(f"""
    train_ratio = {ratio}
    executor = CustomTrainValidationSplit(pipeline, evaluator, grid, 
        train_ratio, seed={seed}, strategy='split')
    """)
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_split_cross_val_strategy_success(builder_params: ModelBuilderParams):
    ratio = .8
    seed = 232
    builder_params.split.strategy = 'cross_validation'
    builder_params.split.seed = seed
    builder_params.split.ratio = ratio

    code = builder_params.split.model_builder_code()
    expected_code = dedent(f"""
    train_ratio = {ratio}
    executor = CustomTrainValidationSplit(pipeline, evaluator, grid, 
        train_ratio, seed={seed}, strategy='cross_validation')
    """)
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

# endregion

# region Evaluator operation

@pytest.mark.parametrize('metric', ['accuracy', 'weightedRecall',
                                    'weightedPrecision', 'f1'])
def test_evaluator_multiclass_success(builder_params: ModelBuilderParams,
                                      metric: str):
    builder_params.evaluator.task_type = 'multiclass-classification'
    builder_params.evaluator.metric = metric
    expected_code = dedent(f"""
    evaluator = evaluation.MulticlassClassificationEvaluator(
        metricName='{metric}', labelCol=label)
    evaluator.task_id = '3d4beb3f-5cba-4656-a228-9064b687cd0b'
    evaluator.operation_id = 2351
    """)
    code = builder_params.evaluator.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('metric', ['areaUnderROC', 'areaUnderPR'])
def test_evaluator_binclass_success(builder_params: ModelBuilderParams,
                                    metric: str):
    builder_params.evaluator.task_type = 'binary-classification'
    builder_params.evaluator.metric = metric
    expected_code = dedent(f"""
    evaluator = evaluation.BinaryClassificationEvaluator(
        metricName='{metric}', labelCol=label)
    evaluator.task_id = '3d4beb3f-5cba-4656-a228-9064b687cd0b'
    evaluator.operation_id = 2351
    """)
    code = builder_params.evaluator.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('metric', ['rmse', 'mse', 'r2', 'mae', 'mape'])
def test_evaluator_regression_success(builder_params: ModelBuilderParams,
                                      metric: str):
    builder_params.evaluator.task_type = 'regression'
    builder_params.evaluator.metric = metric
    expected_code = dedent(f"""
    evaluator = evaluation.RegressionEvaluator(
        metricName='{metric}', labelCol=label)
    evaluator.task_id = '3d4beb3f-5cba-4656-a228-9064b687cd0b'
    evaluator.operation_id = 2351
    """)
    code = builder_params.evaluator.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('metric', ['silhouette'])
def test_evaluator_clustering_success(builder_params: ModelBuilderParams,
                                      metric: str):
    builder_params.evaluator.task_type = 'clustering'
    builder_params.evaluator.metric = metric
    expected_code = dedent(f"""
    evaluator = evaluation.ClusteringEvaluator(metricName='{metric}')
    evaluator.task_id = '3d4beb3f-5cba-4656-a228-9064b687cd0b'
    evaluator.operation_id = 2351
    """)
    code = builder_params.evaluator.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)

# endregion

# region Features Operation
@pytest.mark.parametrize('task_type', ['regression', 'classification'])
def test_features_supervisioned_with_no_label_failure(
        builder_params: ModelBuilderParams, task_type: str):
    """Test if type is regression or classification and label informed"""

    builder_params.features.task_type = task_type
    builder_params.features.features_and_label = (
        builder_params.features.features_and_label[:-1])
    builder_params.features.label = None
    with pytest.raises(ValueError) as ve:
        builder_params.features.process_features_and_label()
    assert 'Missing required parameter: label' in str(ve)


@pytest.mark.parametrize('task_type', ['clustering'])
def test_features_unsupervisioned_with_no_label_success(
        builder_params: ModelBuilderParams, task_type: str):
    """Test if type is unsupervisioned and label doesn't matter"""

    builder_params.features.task_type = task_type
    builder_params.features.features_and_label = (
        builder_params.features.features_and_label[:-1])
    builder_params.features.label = None
    builder_params.features.model_builder_code()

@pytest.mark.parametrize('task_type', ['regression', 'classification',
                                       'clustering'])
def test_features_supervisioned_with_no_features_failure(
        builder_params: ModelBuilderParams, task_type: str):

    builder_params.features.task_type = task_type
    builder_params.features.features_and_label = [builder_params.features.label]
    with pytest.raises(ValueError) as ve:
        builder_params.features.process_features_and_label()
    assert 'Missing required parameter: features' in str(ve)


def test_features_invalid_feature_type_failure(
        builder_params: ModelBuilderParams):
    builder_params.features.label['feature_type'] = 'invalid_type'
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert 'Invalid feature type' in str(ve)


def test_features_invalid_feature_usage_failure(
        builder_params: ModelBuilderParams):
    builder_params.features.label['usage'] = 'invalid_usage'
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert 'Invalid feature usage' in str(ve)


def test_features_categorical_invalid_transform_failure(
        builder_params: ModelBuilderParams):
    builder_params.features.label['transform'] = 'invalid_transformer'
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert 'Invalid transformation' in str(ve)


def test_features_categorical_success(builder_params: ModelBuilderParams):
    """ 
    Expect default label setup: remove nulls and use StringIndexer (dummy)
    """
    # sepal_length, sepal_width, petal_length, petal_width = (
    #     builder_params.features.features)
    # print(sepal_length['name'], sepal_width['name'], petal_length['name'],
    #       petal_width['name'])
    builder_params.features.label['transform'] = 'string_indexer'
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (class) IS NOT NULL')
        features_stages.append(class_del_nulls)
        class_inx = feature.StringIndexer(
            inputCol='class', outputCol='class_inx', handleInvalid='skip'
        )
        features_stages.append(class_inx)
        label = 'class_inx'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_categorical_transform_one_hot_success(
        builder_params: ModelBuilderParams):
    builder_params.features.label['transform'] = 'one_hot_encoder'
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (class) IS NOT NULL')
        features_stages.append(class_del_nulls)
        class_inx = feature.StringIndexer(
            inputCol='class', outputCol='class_inx', handleInvalid='skip'
        )
        features_stages.append(class_inx)
        class_ohe = feature.OneHotEncoder(inputCol='class_inx',
            outputCol='class_inx_ohe')
        features_stages.append(class_ohe)
        label = 'class_inx_ohe'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_categorical_transform_flag_not_null_success(
        builder_params: ModelBuilderParams):
    builder_params.features.label['transform'] = 'not_null'
    expected_code = dedent("""
        class_na = feature.SQLTransformer(
            statement='SELECT *, INT(ISNULL(class)) AS class_na FROM __THIS__')
        features_stages.append(class_na)
        label = 'class_na'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_categorical_handle_null_success(
        builder_params: ModelBuilderParams):
    builder_params.features.label['missing_data'] = 'remove'
    expected_code = dedent("""
        class_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (class) IS NOT NULL')
        features_stages.append(class_del_nulls)
        class_inx = feature.StringIndexer(
            inputCol='class', outputCol='class_inx', handleInvalid='skip'
        )
        features_stages.append(class_inx)
        label = 'class_inx'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('constant', ['Iris-virginica', 'Iris-setosa'])
def test_features_categorical_handle_null_with_constant_success(
        builder_params: ModelBuilderParams, constant: str):
    builder_params.features.label['missing_data'] = 'constant'
    builder_params.features.label['constant'] = constant
    expected_code = dedent(f"""
        class_na = feature.SQLTransformer(
            statement="SELECT *, COALESCE(class, '{constant}') "
                "AS class_na FROM __THIS__")
        features_stages.append(class_na)
        class_inx = feature.StringIndexer(
            inputCol='class_na', outputCol='class_na_inx', handleInvalid='skip'
        )
        features_stages.append(class_inx)
        label = 'class_na_inx'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_categorical_handle_null_with_invalid_constant_failure(
        builder_params: ModelBuilderParams):
    builder_params.features.label['missing_data'] = 'constant'
    builder_params.features.label['constant'] = None
    with pytest.raises(ValueError) as ve:
        builder_params.features.model_builder_code()
    assert 'Missing constant value' in str(ve)


def test_features_numerical_success(builder_params: ModelBuilderParams):
    """ 
    Change task_type to regression and set sepallength as label
    Expect default label setup: remove nulls? 
    """
    builder_params.features.task_type = 'regression'
    builder_params.features.features[0]['usage'] = 'label'
    builder_params.features.features_and_label.pop()  # remove old label
    builder_params.features.process_features_and_label()

    expected_code = dedent("""
        label = 'sepallength'
    """)
    code = builder_params.features.model_builder_code()
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


# def test_features_numerical_keep_as_is_success(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


# def test_features_numerical_binarize_success(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


# def test_features_numerical_transform_quantis_success(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


# def test_features_numerical_transform_buckets_success(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


# def test_features_numerical_transform_invalid_failure(
#         builder_params: ModelBuilderParams):
#     assert False, 'Implement'


@pytest.mark.parametrize('action, algo',
                         [('min_max', 'MinMaxScaler'),
                          ('max_abs', 'MaxAbsScaler'),
                          ('standard', 'StandardScaler')])
def test_features_numerical_scaler_success(
        builder_params_no_label: ModelBuilderParams, action: str, algo: str):

    # sepallength
    builder_params_no_label.features.features[0]['scaler'] = action
    assert(
        builder_params_no_label.features.features[0]['name'] == 'sepallength')

    expected_code = dedent(f"""
        sepallength_asm = feature.VectorAssembler(
            handleInvalid='skip',
            inputCols=['sepallength'],
            outputCol='sepallength_asm')
        features_stages.append(sepallength_asm)
        sepallength_scl = feature.{algo}(
            inputCol='sepallength_asm',
            outputCol='sepallength_scl')
        features_stages.append(sepallength_scl)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert ({'sepallength_scl', 'sepalwidth', 'petallength', 'petalwidth'} ==
            set(builder_params_no_label.features.features_names))
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('constant', [3.14, 1, -123])
def test_features_numerical_constant_if_null_success(
        builder_params_no_label: ModelBuilderParams, constant: float):
    # sepallength
    builder_params_no_label.features.features[0]['missing_data'] = 'constant'

    builder_params_no_label.features.features[0]['constant'] = constant

    expected_code = dedent(f"""
        sepallength_na = feature.SQLTransformer(
            statement=('SELECT *, COALESCE(sepallength, {constant}) '
                       'AS sepallength_na FROM __THIS__'))
        features_stages.append(sepallength_na)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert ({'sepallength_na', 'sepalwidth', 'petallength', 'petalwidth'} ==
            set(builder_params_no_label.features.features_names))
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


@pytest.mark.parametrize('action', ['media', 'median'])
def test_features_numerical_media_or_median_if_null_success(
        builder_params_no_label: ModelBuilderParams, action: str):
    # petallength
    builder_params_no_label.features.features[2]['missing_data'] = action

    expected_code = dedent(f"""
        petallength_imp = feature.Imputer(
            strategy='{action}', inputCols=['petallength'], 
            outputCols=['petallength_na'])
        features_stages.append(petallength_na)
    """)
    code = builder_params_no_label.features.model_builder_code()
    assert ({'sepallength', 'sepalwidth', 'petallength_na', 'petalwidth'} ==
            set(builder_params_no_label.features.features_names))
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_median_if_null_multiple_success(
        builder_params: ModelBuilderParams):
    # sepallength
    builder_params.features.features[0]['missing_data'] = 'median'
    builder_params.features.features[1]['missing_data'] = 'median'

    # Remove the original label to simplify the test
    builder_params.features.task_type = 'clustering'
    builder_params.features.features_and_label.pop()
    builder_params.features.process_features_and_label()

    expected_code = dedent("""
        sepallength_imp = feature.Imputer(
            strategy='median', inputCols=['sepallength'], 
            outputCols=['sepallength_na'])
        features_stages.append(sepallength_na)
        sepalwidth_imp = feature.Imputer(
            strategy='median', inputCols=['sepalwidth'], 
            outputCols=['sepalwidth_na'])
        features_stages.append(sepalwidth_na)
    """)
    code = builder_params.features.model_builder_code()
    assert ({'sepallength_na', 'sepalwidth_na', 'petallength', 'petalwidth'} ==
            set(builder_params.features.features_names))

    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


def test_features_numerical_remove_if_null_success(
        builder_params: ModelBuilderParams):

    # sepallength
    builder_params.features.features[0]['missing_data'] = 'remove'

    # Remove the original label to simplify the test
    builder_params.features.task_type = 'clustering'
    builder_params.features.features_and_label.pop()
    builder_params.features.process_features_and_label()

    expected_code = dedent("""
        sepallength_del_nulls = feature.SQLTransformer(
            statement='SELECT * FROM __THIS__ WHERE (sepallength) IS NOT NULL')
        features_stages.append(sepallength_del_nulls)
    """)
    code = builder_params.features.model_builder_code()
    assert ({'sepallength', 'sepalwidth', 'petallength', 'petalwidth'} ==
            set(builder_params.features.features_names))
    result, msg = compare_ast(ast.parse(expected_code), ast.parse(code))
    assert result, format_code_comparison(expected_code, code, msg)


# endregion


def test_grid(builder_params: ModelBuilderParams):
    for k in ['strategy', 'random_grid', 'seed', 'max_iterations',
              'max_search_time', 'parallelism']:
        print(k, builder_params.grid.parameters.get(k))

    builder_params.grid.has_code = True
    builder_params.grid.strategy = 'random'
    builder_params.grid.max_iterations = 30
    # Test strategy parameter:
    # - grid: ok
    # - random: ok

    print(builder_params.grid.model_builder_code())


def test_reduction(builder_params: ModelBuilderParams):
    for k in ['method', 'number', 'k']:
        print(k, builder_params.reduction.parameters.get(k))
    print(builder_params.reduction.parameters.keys())

    builder_params.reduction.method = 'pca'
    builder_params.reduction.has_code = True
    builder_params.reduction.k = 30
    # Test method parameter:
    # - disable: no reduction
    # - pca: Uses pca

    print(builder_params.reduction.model_builder_code())


def test_estimators(builder_params: ModelBuilderParams):

    print('Estimators',  builder_params.estimators)
    for estimator in builder_params.estimators:
        for k in ['method', 'number', 'k']:
            print(k, estimator.parameters.get(k))
            # print(estimator.parameters.keys())

        estimator.has_code = True
        # Test method parameter:
        # - disable: no estimator
        # - pca: Uses pca
        print('-'*20)
        estimator.grid_info.get('value')['strategy'] = 'random'
        estimator.grid_info.get('value')['max_iterations'] = 32
        print(estimator.model_builder_code())
        print(estimator.generate_hyperparameters_code())
        # print(estimator.generate_random_hyperparameters_code())


def test_builder_params(sample_workflow: dict, builder_params: ModelBuilderParams):

    loader = Workflow(sample_workflow, config, lang='en')
    instances = loader.workflow['tasks']

    minion = MetaMinion(None, config=config, workflow_id=sample_workflow['id'],
                        app_id=sample_workflow['id'])

    job_id = 1
    opt = GenerateCodeParams(loader.graph, job_id, None, {},
                             ports={}, state={}, task_hash=hashlib.sha1(),
                             workflow=loader.workflow,
                             tasks_ids=list(loader.graph.nodes.keys()))
    instances, _ = minion.transpiler.get_instances(opt)

    builder_params = minion.transpiler.prepare_model_builder_parameters(
        ops=instances.values())

    print(dir(builder_params))
    print(builder_params.read_data.model_builder_code())
    print(builder_params.sample.model_builder_code())
    print(builder_params.split.model_builder_code())
    print(builder_params.evaluator.model_builder_code())
    print(builder_params.features.model_builder_code())
    print(builder_params.reduction.model_builder_code())
    print(builder_params.grid.model_builder_code())


def xtest_generate_run_code_success(sample_workflow: dict, builder_params: ModelBuilderParams):

    job_id = 1
    estimator = builder_params.estimators[0]
    estimator.has_code = True
    estimator.grid_info.get('value')['strategy'] = 'random'
    estimator.grid_info.get('value')['strategy'] = 'grid'
    estimator.grid_info.get('value')['max_iterations'] = 32

    # import pdb; pdb.set_trace()
    loader = Workflow(sample_workflow, config, lang='en')

    loader.handle_variables({'job_id': job_id})
    out = StringIO()

    minion = MetaMinion(None, config=config, workflow_id=sample_workflow['id'],
                        app_id=sample_workflow['id'])

    minion.transpiler.transpile(loader.workflow, loader.graph,
                                config, out, job_id,
                                persist=False)
    out.seek(0)
    code = out.read()
    with open('/tmp/juicer_app_1_1_1.py', 'w') as f:
        f.write(code)

    print(code, file=sys.stderr)

    result = util.execute(code, {'df': 'df'})
    print('Executed')
    # assert not result['out'].equals(test_df)
    # assert """out = df.sort_values(by=['sepalwidth'], ascending=[False])""" == \
    #        instance.generate_code()
